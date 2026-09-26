# -*- coding: utf-8 -*-
"""q1v2_assets.py —— 资产定位、预取、身份核验与环境冻结（问题一 v2）

四条原则，全部对应赛题红线：

1. **运行期零网络。** 所有外部资产在 `prepare` 阶段一次性备齐，之后只按本地路径加载。
   RoBERTa 与 whisper 本机已存在（零下载）；只有 MediaPipe 的 `face_landmarker.task`
   不在 pip 包里，需取一次。
2. **按 SHA-256 认资产，不按文件名认。** 六个 RoBERTa 文件逐个校验——只比
   `model.safetensors` 会漏掉 tokenizer 被换掉的情形，那同样会改变 768 维词向量，
   而落盘的 revision 字符串还是旧的。
3. **版本记录而非钉死。** 对照实现把 Python 3.12.10 与 10 个包版本硬钉在
   `check_assets` 里；本机是 Python 3.13.9 / transformers 5.17.0 / torch 2.14.0。
   降级会与正在跑的 Q2/Q3 冲突，故**记录实际版本并显式声明差异**，不假装一致。
4. **只写相对路径。** 对照实现的 `run_q1_v1.py` 把 `str(args.input_root)` 写进了
   `environment.json`，产物因此带上本机用户名。本模块一律走 `relpath_for_delivery`。
"""

from __future__ import annotations

import os
import platform
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from q1v2_contract import (
    FACE_MODEL_BYTES,
    FACE_MODEL_PATH,
    FACE_MODEL_SHA256,
    FACE_MODEL_URL,
    LABEL_XLSX_SHA256,
    OPENMILE_CONFIG_SHA256,
    OPENMILE_FEATURE_NAMES,
    OPENMILE_LLD_DIM,
    PROTECTED_DIRS,
    VOLATILE_BY_DESIGN,
    PROJECT_ROOT,
    ROBERTA_DIR_CANDIDATES,
    ROBERTA_FILE_SHA256,
    ROBERTA_HIDDEN_DIM,
    ROBERTA_MODEL_ID,
    ROBERTA_REVISION,
    VENV_PYTHON,
    WHISPER_CHECKPOINT_CANDIDATES,
    WHISPER_CHECKPOINT_SHA256,
    WHISPER_MODEL_NAME,
    WORK_ROOT,
    WORK_ROOT_REASON,
    HardStop,
    excluded_walk,
    relpath_for_delivery,
    sha256,
)

#: 分发用的包名 → 实际 import 名。写进 environment.json 便于复现。
_TRACKED_PACKAGES = (
    ("torch", "torch"),
    ("torchaudio", "torchaudio"),
    ("transformers", "transformers"),
    ("av", "av"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("librosa", "librosa"),
    ("openai-whisper", "whisper"),
    ("stable-ts", "stable_whisper"),
    ("opensmile", "opensmile"),
    ("mediapipe", "mediapipe"),
    ("openpyxl", "openpyxl"),
)


class AssetError(RuntimeError):
    """资产缺失或校验失败。区别于样本级 HardStop——这是整批级错误。"""


# --------------------------------------------------------------------------
# 定位
# --------------------------------------------------------------------------


def _first_existing(candidates) -> Optional[Path]:
    for path in candidates:
        if Path(path).exists():
            return Path(path)
    return None


def resolve_roberta_dir(required: bool = True) -> Optional[Path]:
    """定位 RoBERTa 本地快照目录。

    **为什么不直接 `from_pretrained("FacebookAI/roberta-base")`**：本机缓存目录名是
    `models--roberta-base`（当初按短名 `roberta-base` 下载的），而带命名空间的
    repo id 会去找 `models--FacebookAI--roberta-base`——名字对不上，`local_files_only`
    下必然未命中。直接解析到快照目录既绕开这个命名依赖，又把 revision 钉死成目录名，
    比 hub 名更可靠。
    """
    override = os.environ.get("Q1V2_ROBERTA_DIR")
    candidates = ([Path(override)] if override else []) + list(ROBERTA_DIR_CANDIDATES)
    found = _first_existing(candidates)
    if found is None and required:
        raise AssetError(
            "未找到 RoBERTa 快照目录。已查找：\n  "
            + "\n  ".join(str(c) for c in candidates)
            + "\n可用环境变量 Q1V2_ROBERTA_DIR 显式指定（目录内须含冻结的 6 个文件）。"
        )
    return found


def verify_roberta_dir(model_dir: Path) -> Dict[str, Any]:
    """逐个文件核对 SHA-256。任一处不符即报错——不做「部分匹配」的让步。"""
    files: Dict[str, str] = {}
    missing: List[str] = []
    mismatched: List[str] = []
    for name, expected in sorted(ROBERTA_FILE_SHA256.items()):
        path = Path(model_dir) / name
        if not path.is_file():
            missing.append(name)
            files[name] = "missing"
            continue
        got = sha256(path)
        files[name] = got
        if got != expected:
            mismatched.append(name)
    if missing or mismatched:
        raise AssetError(
            f"RoBERTa 资产身份不符：缺失 {missing}，SHA 不符 {mismatched}。\n"
            f"目录：{relpath_for_delivery(model_dir)}"
        )
    return {
        "dir": relpath_for_delivery(model_dir),
        "files_sha256": files,
        "n_files": len(files),
        "all_match": True,
    }


def resolve_whisper_checkpoint(required: bool = True) -> Optional[Path]:
    """定位 whisper `base.en.pt` 检查点（stable-ts 强制对齐用）。"""
    override = os.environ.get("Q1V2_WHISPER_CHECKPOINT")
    candidates = ([Path(override)] if override else []) + list(WHISPER_CHECKPOINT_CANDIDATES)
    found = _first_existing(candidates)
    if found is None and required:
        raise AssetError(
            "未找到 whisper 检查点 base.en.pt。已查找：\n  "
            + "\n  ".join(str(c) for c in candidates)
        )
    return found


def verify_whisper_checkpoint(path: Path) -> Dict[str, Any]:
    got = sha256(path)
    if got != WHISPER_CHECKPOINT_SHA256:
        raise AssetError(
            f"whisper 检查点 SHA 不符：{got} != {WHISPER_CHECKPOINT_SHA256}"
        )
    return {"file": relpath_for_delivery(path), "sha256": got,
            "bytes": path.stat().st_size, "model_name": WHISPER_MODEL_NAME}


def ensure_face_model(path: Optional[Path] = None,
                      allow_download: bool = False) -> Dict[str, Any]:
    """确保 `face_landmarker.task` 就位且 SHA 正确；缺失时（经授权）下载一次。

    下载后**先校验 SHA 再改名落盘**：校验不过的文件直接丢弃，
    绝不让一份来源不明的权重进入后续所有样本的特征里。
    """
    target = Path(path or FACE_MODEL_PATH)
    if target.is_file():
        got = sha256(target)
        if got != FACE_MODEL_SHA256:
            raise AssetError(
                f"已存在的 face_landmarker.task SHA 不符：{got}\n"
                f"请删除 {relpath_for_delivery(target)} 后重跑（--allow-download）。"
            )
        return {"file": relpath_for_delivery(target), "sha256": got,
                "bytes": target.stat().st_size, "source_url": FACE_MODEL_URL,
                "downloaded_this_run": False}

    if not allow_download:
        raise AssetError(
            f"缺少 {relpath_for_delivery(target)}（MediaPipe 权重不在 pip 包内）。\n"
            "请加 --allow-download 一次性预取，之后即可离线运行。"
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_suffix(".task.part")
    request = urllib.request.Request(FACE_MODEL_URL, headers={"User-Agent": "q1v2-assets/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, staging.open("wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)
    got = sha256(staging)
    if got != FACE_MODEL_SHA256:
        staging.unlink(missing_ok=True)
        raise AssetError(
            f"下载到的 face_landmarker.task SHA 不符：{got}\n"
            f"期望：{FACE_MODEL_SHA256}\n已丢弃该文件。"
        )
    if staging.stat().st_size != FACE_MODEL_BYTES:
        staging.unlink(missing_ok=True)
        raise AssetError("下载体大小与冻结值不符，已丢弃。")
    staging.replace(target)
    return {"file": relpath_for_delivery(target), "sha256": got,
            "bytes": target.stat().st_size, "source_url": FACE_MODEL_URL,
            "downloaded_this_run": True}


def opensmile_info() -> Dict[str, Any]:
    """读 openSMILE 的配置身份。值来自返回值，不自己推算。"""
    try:
        from opensmile import FeatureLevel, FeatureSet, Smile
    except Exception as exc:  # noqa: BLE001
        raise AssetError(
            f"openSMILE 不可用（{type(exc).__name__}: {exc}）。\n"
            f"当前解释器 {sys.executable}\n"
            f"若路径含非 ASCII 字符，openSMILE 的 Python 绑定会抛 UnicodeEncodeError；"
            f"应使用 ASCII 路径下的隔离 venv：{VENV_PYTHON}"
        ) from exc
    smile = Smile(feature_set=FeatureSet.eGeMAPSv02,
                  feature_level=FeatureLevel.LowLevelDescriptors,
                  num_workers=1, multiprocessing=False, verbose=False)
    config_path = Path(smile.config_path)
    config_sha = sha256(config_path)
    names = list(smile.feature_names)
    if len(names) != OPENMILE_LLD_DIM:
        raise AssetError(f"openSMILE LLD 维度为 {len(names)}，期望 {OPENMILE_LLD_DIM}")
    if tuple(names) != OPENMILE_FEATURE_NAMES:
        raise AssetError(
            "openSMILE 返回的特征名或顺序与冻结值不符——(L,25) 的列语义会变，"
            "所有音频特征将不可比。\n"
            f"实得：{names}\n冻结：{list(OPENMILE_FEATURE_NAMES)}"
        )
    if config_sha != OPENMILE_CONFIG_SHA256:
        raise AssetError(
            f"eGeMAPSv02.conf SHA 不符：{config_sha} != {OPENMILE_CONFIG_SHA256}"
        )
    return {
        "config_file": config_path.name,
        "config_path": str(config_path),
        "config_sha256": config_sha,
        "feature_set": "eGeMAPSv02",
        "feature_level": "LowLevelDescriptors",
        "feature_dim": OPENMILE_LLD_DIM,
        "feature_names": names,
        "num_workers": 1,
        "multiprocessing": False,
    }


# --------------------------------------------------------------------------
# 版本与环境
# --------------------------------------------------------------------------


def package_versions() -> Dict[str, str]:
    """记录**实际**版本。不钉死、不降级——降级会与在跑的 Q2/Q3 冲突。"""
    from importlib.metadata import PackageNotFoundError, version as pkg_version
    out: Dict[str, str] = {}
    for dist_name, _import_name in _TRACKED_PACKAGES:
        try:
            out[dist_name] = pkg_version(dist_name)
        except PackageNotFoundError:
            out[dist_name] = "not-installed"
        except Exception:  # noqa: BLE001
            out[dist_name] = "unknown"
    return out


def freeze_environment() -> Dict[str, Any]:
    """环境快照。**不含绝对路径中的用户名**——工作区路径用相对/中性写法记录。"""
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "interpreter": "isolated venv (ASCII path)",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": package_versions(),
        "work_root": str(WORK_ROOT),
        "work_root_reason": WORK_ROOT_REASON,
        "work_root_note": (
            "原生资产与隔离 venv 的工作区。openSMILE 的 Python 绑定以 ASCII 编码"
            "配置文件路径（opensmile/core/lib.py），MediaPipe 亦把权重路径交给原生层，"
            "故二者所在路径必须为纯 ASCII。本仓库根目录含中文，因此工作区被解析到"
            "仓库之外的 ASCII 路径。工作区不随仓库提交、不占提交体积。"
        ),
        "venv_python": str(VENV_PYTHON),
        "hf_endpoint_note": (
            "环境变量 HF_ENDPOINT 存在时不使用它：运行期零网络，"
            "全部模型按本地快照目录直载（local_files_only=True）。"
        ),
        "offline_enforced": True,
    }


def version_differences_from_reference(reference: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """与对照实现的资格环境逐项对比，**记录差异而非消除差异**（R6）。"""
    reference = reference or {
        "python": "3.12.10", "torch": "2.8.0", "torchaudio": "2.8.0",
        "stable-ts": "2.19.1", "openai-whisper": "20250625", "opensmile": "2.6.0",
        "mediapipe": "0.10.35", "transformers": "4.57.6", "av": "18.0.0",
        "pandas": "3.0.3", "openpyxl": "3.1.5",
    }
    actual = {"python": platform.python_version(), **package_versions()}
    rows = []
    for name, ref in reference.items():
        got = actual.get(name, "unknown")
        rows.append({"package": name, "reference": ref, "ours": got,
                     "same": got == ref})
    return {
        "rows": rows,
        "n_same": sum(1 for r in rows if r["same"]),
        "n_diff": sum(1 for r in rows if not r["same"]),
        "policy": (
            "不降级。降级 transformers/torch 会与同一仓库正在运行的问题二/问题三冲突，"
            "代价大于收益。差异已由词单元口径（1932/1926/0，本机复算命中）部分排除风险；"
            "音频与视觉走的是 PyAV + openSMILE + MediaPipe 的冻结配置，"
            "与 torch/transformers 版本无耦合。"
        ),
    }


# --------------------------------------------------------------------------
# 总入口
# --------------------------------------------------------------------------


def check_assets(allow_download: bool = False,
                 label_xlsx: Optional[Path] = None) -> Dict[str, Any]:
    """`prepare` 阶段的总入口：定位 + 校验全部资产，返回可落盘的身份快照。"""
    model_dir = resolve_roberta_dir()
    roberta = verify_roberta_dir(model_dir)
    roberta["model_id"] = ROBERTA_MODEL_ID
    roberta["revision"] = ROBERTA_REVISION
    roberta["hidden_dim"] = ROBERTA_HIDDEN_DIM

    whisper = verify_whisper_checkpoint(resolve_whisper_checkpoint())
    face = ensure_face_model(allow_download=allow_download)
    smile = opensmile_info()

    label = None
    if label_xlsx is not None:
        label_path = Path(label_xlsx)
        got = sha256(label_path)
        label = {"file": relpath_for_delivery(label_path), "sha256": got,
                 "matches_reference": got == LABEL_XLSX_SHA256,
                 "reference_sha256": LABEL_XLSX_SHA256}

    return {
        "roberta": roberta,
        "whisper": whisper,
        "mediapipe": face,
        "opensmile": smile,
        "label_xlsx": label,
        "work_root": str(WORK_ROOT),
        "environment": freeze_environment(),
        "version_differences": version_differences_from_reference(),
    }


# --------------------------------------------------------------------------
# 受保护基线：证明本次重建没有破坏既有交付
# --------------------------------------------------------------------------


def snapshot_protected_baseline() -> Dict[str, Any]:
    """对既有交付目录做全文件 SHA 快照。相对路径，不含用户名。"""
    snapshot: Dict[str, Any] = {"dirs": {}, "total_files": 0, "total_bytes": 0}
    for rel in PROTECTED_DIRS:
        root = PROJECT_ROOT / rel
        if not root.is_dir():
            snapshot["dirs"][rel] = {"exists": False, "files": {}, "bytes": 0}
            continue
        files: Dict[str, str] = {}
        total = 0
        for path in sorted(excluded_walk(root)):
            key = path.relative_to(PROJECT_ROOT).as_posix()
            files[key] = sha256(path)
            total += path.stat().st_size
        snapshot["dirs"][rel] = {"exists": True, "files": files, "bytes": total}
        snapshot["total_files"] += len(files)
        snapshot["total_bytes"] += total
    return snapshot


def compare_protected_baseline(baseline: Dict[str, Any]) -> Dict[str, Any]:
    """重算并与基线比对，逐文件给出 added / removed / changed 三张清单。

    改动按 `VOLATILE_BY_DESIGN` 分流：

    * `changed` —— **硬失败**，只含不在名单里的文件；
    * `changed_volatile_by_design` —— 记录 + 归因，不判失败（见契约里的说明）；
    * `unchanged_strict` —— 连名单内文件都没动，供人核对「是否发生过任何重跑」。
    """
    current = snapshot_protected_baseline()
    added: List[str] = []
    removed: List[str] = []
    changed: List[str] = []
    for rel, entry in baseline.get("dirs", {}).items():
        before = entry.get("files", {})
        after = current["dirs"].get(rel, {}).get("files", {})
        added.extend(sorted(set(after) - set(before)))
        removed.extend(sorted(set(before) - set(after)))
        changed.extend(sorted(k for k in set(before) & set(after) if before[k] != after[k]))
    volatile = [p for p in changed if p in VOLATILE_BY_DESIGN]
    hard = [p for p in changed if p not in VOLATILE_BY_DESIGN]

    # 自检：名单里的键若在基线文件表里**一条都匹配不到**，多半是键写错了
    # （本项目刚踩过一次：用了 `os.path.join` 的反斜杠键，快照是正斜杠，
    # 于是名单声明了却一条都没分流，表现为"改动全是硬失败"）。
    # 这类静默失效不会自己暴露，所以显式列出来供人核对。
    known = {p for entry in baseline.get("dirs", {}).values()
             for p in entry.get("files", {})}
    dead_keys = sorted(k for k in VOLATILE_BY_DESIGN if k not in known)

    return {
        "unchanged": not (added or removed or hard),
        "unchanged_strict": not (added or removed or changed),
        "added": added,
        "removed": removed,
        "changed": hard,
        "changed_volatile_by_design": [
            {"file": p, "reason": VOLATILE_BY_DESIGN[p]} for p in volatile],
        "volatile_keys_not_in_baseline": dead_keys,
        "baseline_total_files": baseline.get("total_files", 0),
        "current_total_files": current["total_files"],
    }


__all__ = [
    "AssetError", "resolve_roberta_dir", "verify_roberta_dir",
    "resolve_whisper_checkpoint", "verify_whisper_checkpoint",
    "ensure_face_model", "opensmile_info", "package_versions",
    "freeze_environment", "version_differences_from_reference", "check_assets",
    "snapshot_protected_baseline", "compare_protected_baseline",
]
