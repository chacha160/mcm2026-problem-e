# -*- coding: utf-8 -*-
"""run_q1v2_all.py —— 问题一 v2 全流程编排

    prepare → text → align → media → tables → verify

## 逐样本隔离策略：媒体阶段用子进程，文本/对齐阶段用批内 try/except

计划里三个阶段都写的是「逐样本子进程」。实现时按**风险来源**拆成两种，理由如下：

* **media**：PyAV 解码 + MediaPipe(TF Lite) + openSMILE(C 扩展) 三个原生栈叠在一起，
  段错误会**直接带走整个进程**——这类故障 try/except 拦不住，只有子进程能隔离。
  故媒体阶段默认逐样本起子进程（超时 900s、最多 2 次重试）。
* **text / align**：纯 PyTorch / stable-ts，无自定义原生扩展，段错误风险可忽略；
  而每个子进程都要重新加载 RoBERTa（~5 s）或 Whisper（~3 s），100 条就是
  白付 8~13 分钟。故改为**批内逐样本 try/except + 即时落盘**：一条样本抛
  `HardStop` 只记该条失败并继续下一条，缓存也已在磁盘上，中断后可续跑。
  这与子进程方案在「HardStop 不拖停整批」（R11）上等效，只对段错误不等效——
  而那正是这两阶段不存在的风险。

两种模式都可用 `--isolate` / `--no-isolate` 强制覆盖。

## 断点续跑

每个阶段**逐样本即写即落**，重跑时默认跳过已存在的产物（`--overwrite` 反选）。
故任何中断都可原样重跑，只补缺失的部分。媒体子进程写到 `.tmp` 再 `replace`，
不会留下半截 NPZ 被当作成品跳过。
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import numpy as np

import q1v2_contract as C

OUTPUT_ROOT = C.OUTPUT_ROOT
CACHE_ROOT = C.CACHE_ROOT
LOGS_DIR = OUTPUT_ROOT / "logs"
FEATURES_DIR = OUTPUT_ROOT / "features"
METADATA_DIR = OUTPUT_ROOT / "metadata"
REPORTS_DIR = OUTPUT_ROOT / "reports"
AUDIT_DIR = OUTPUT_ROOT / "audit"

STAGES = ("prepare", "text", "align", "media", "tables", "verify")


# --------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    """numpy → JSON 可序列化。用于把缓存写成 JSON。"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_log(stem: str, stage: str, payload: Dict[str, Any]) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    path = LOGS_DIR / f"{stem}.{stage}.json"
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(_jsonable(payload), ensure_ascii=False,
                               indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(path)


def _read_log(stem: str, stage: str) -> Optional[Dict[str, Any]]:
    path = LOGS_DIR / f"{stem}.{stage}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _child_python() -> str:
    """子进程解释器：优先用装了 mediapipe/opensmile 的那个 venv。"""
    if C.VENV_PYTHON.is_file():
        return str(C.VENV_PYTHON)
    return sys.executable


def _child_env() -> Dict[str, str]:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONPATH"] = str(_CODE_DIR)
    env["Q1V2_WORK_ROOT"] = str(C.WORK_ROOT)
    return env


def ffprobe_format_duration_sec(video_path: Path) -> float:
    """容器 format 层时长（秒）。**外部交叉核对用，不参与任何时间轴定义。**

    与解码覆盖本就不必相等（A 实证 6.800 vs 6.8667）——容器声明的是标称时长，
    解码拿到的是实际 presentation 覆盖。两者差异是 V6 的核对对象，不是错误。
    """
    ffprobe = "ffprobe"
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(video_path)],
            capture_output=True, text=True, timeout=60,
        )
        return float(out.stdout.strip())
    except Exception:
        return float("nan")


def _role_from_durations(durations: Sequence[float]) -> List[str]:
    """按时长分位给样本打标签：最短/最长十分位为 short/long，其余 ordinary。

    这个标签**只用于分组阅读**，不参与任何筛选或删除——
    异常阈值永不作为删除样本的理由。
    """
    arr = np.asarray(durations, dtype=np.float64)
    finite = np.isfinite(arr)
    if finite.sum() < 3:
        return ["ordinary_median_duration_proxy"] * len(arr)
    lo = np.nanpercentile(arr[finite], 10)
    hi = np.nanpercentile(arr[finite], 90)
    roles = []
    for value in arr:
        if not np.isfinite(value):
            roles.append("duration_unavailable")
        elif value <= lo:
            roles.append("short_duration_proxy")
        elif value >= hi:
            roles.append("long_duration_proxy")
        else:
            roles.append("ordinary_median_duration_proxy")
    return roles


# --------------------------------------------------------------------------
# prepare
# --------------------------------------------------------------------------


def stage_prepare(args: argparse.Namespace) -> Dict[str, Any]:
    """S1：资产预检 / 冻结、样本登记、受保护基线快照、路由说明书。"""
    import q1v2_assets as A
    import q1v2_route as R
    import unaligned_common as U

    for d in (OUTPUT_ROOT, CACHE_ROOT, LOGS_DIR, FEATURES_DIR,
              METADATA_DIR, REPORTS_DIR, AUDIT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print("[prepare] 资产预检 ...")
    label_xlsx = Path(U.config.ATTACHMENT1_DIR) / U.config.LABEL_XLSX_NAME
    assets = A.check_assets(allow_download=bool(args.allow_download),
                            label_xlsx=label_xlsx)

    # 显式逐项比对冻结值。不做这一步的话，"资产检查通过"只是说文件在，
    # 而不是说文件**是那一份**——后者才是可复现性的前提。
    asset_checks = {
        "roberta_model_sha256": (
            assets["roberta"]["files_sha256"].get("model.safetensors"),
            C.ROBERTA_FILE_SHA256["model.safetensors"]),
        "mediapipe_model_sha256": (
            assets["mediapipe"].get("sha256"), C.FACE_MODEL_SHA256),
        "opensmile_config_sha256": (
            assets["opensmile"].get("config_sha256"), C.OPENMILE_CONFIG_SHA256),
        "whisper_checkpoint_sha256": (
            assets["whisper"].get("sha256"), C.WHISPER_CHECKPOINT_SHA256),
    }
    asset_mismatches = {
        name: {"got": got, "expected": expected}
        for name, (got, expected) in asset_checks.items() if got != expected
    }
    if asset_mismatches:
        raise C.HardStop(f"冻结资产身份不符：{json.dumps(asset_mismatches, ensure_ascii=False)}")
    assets["all_ok"] = True
    assets["verified_against_frozen_constants"] = sorted(asset_checks)

    print("[prepare] 环境快照 ...")
    environment = dict(assets.get("environment") or A.freeze_environment())
    environment["version_differences_from_reference"] = assets.get(
        "version_differences", A.version_differences_from_reference())
    C.write_json(METADATA_DIR / "environment.json", environment)

    # config_snapshot.json —— config_digest() 就是它的 SHA，用来标「哪份配置产出」
    C.write_json(OUTPUT_ROOT / "config_snapshot.json", {
        "schema_version": C.SCHEMA_VERSION,
        "align_params": C.ALIGN_PARAMS,
        "align_source": C.ALIGN_SOURCE,
        "aligner_name": C.ALIGNER_NAME,
        "roberta_model_id": C.ROBERTA_MODEL_ID,
        "roberta_revision": C.ROBERTA_REVISION,
        "roberta_hidden_dim": C.ROBERTA_HIDDEN_DIM,
        "face_model_dim": C.FACE_MODEL_DIM,
        "opensmile_lld_dim": C.OPENMILE_LLD_DIM,
        "opensmile_feature_names": list(C.OPENMILE_FEATURE_NAMES),
        "asset_policy": {"runtime_network": False},
    })

    print("[prepare] 样本登记 ...")
    label_sha = C.sha256(label_xlsx)
    if label_sha != C.LABEL_XLSX_SHA256:
        raise C.HardStop(
            f"label-100.xlsx SHA 不符：{label_sha} != {C.LABEL_XLSX_SHA256}")

    raw = U.list_samples()
    samples: List[Dict[str, Any]] = []
    for row in raw:
        video_path = Path(str(row["video_path"]))
        text = str(row["text"])
        samples.append({
            "sample_key": C.sample_key_of(str(row["video_id"]), str(row["clip_id"])),
            "video_id": str(row["video_id"]),
            "clip_id": str(row["clip_id"]),
            "official_text": text,
            "official_text_sha256": C.sha256_text(text),
            "source_video_relpath": C.relpath_for_delivery(video_path),
            "source_video_sha256": C.sha256(video_path),
            "source_label_xlsx_sha256": label_sha,
            "ffprobe_format_duration_sec": ffprobe_format_duration_sec(video_path),
        })

    roles = _role_from_durations([s["ffprobe_format_duration_sec"] for s in samples])
    for sample, role in zip(samples, roles):
        sample["role"] = role
        sample["modalities"] = "text|audio|vision"

    registry = {
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "schema_version": C.SCHEMA_VERSION,
        "n_samples": len(samples),
        "label_xlsx_relpath": C.relpath_for_delivery(label_xlsx),
        "label_xlsx_sha256": label_sha,
        "config_sha256": C.config_digest(),
        "role_policy": (
            "按时长十分位分组，仅供阅读；不参与筛选，不删除任何样本"
        ),
        "samples": samples,
    }
    C.write_json(METADATA_DIR / "registry.json", registry)

    print("[prepare] 冻结模型资产身份 ...")
    C.write_json(METADATA_DIR / "model_assets.json", assets)

    print("[prepare] 路由政策说明书 ...")
    C.write_json(METADATA_DIR / "route_policy.json", R.route_policy_document())

    print("[prepare] 受保护基线快照 ...")
    baseline = A.snapshot_protected_baseline()
    baseline["generated_at_utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    C.write_json(METADATA_DIR / "protected_baseline.json", baseline)

    summary = {
        "n_samples": len(samples),
        "roles": {r: roles.count(r) for r in sorted(set(roles))},
        "assets_ok": bool(assets.get("all_ok", True)),
        "protected_dirs": {k: len(v.get("files", {}))
                           for k, v in baseline.get("dirs", {}).items()},
        "protected_total_bytes": baseline.get("total_bytes", 0),
        "config_sha256": registry["config_sha256"],
    }
    C.write_json(REPORTS_DIR / "prepare_summary.json", summary)
    return summary


def load_registry() -> List[Dict[str, Any]]:
    path = METADATA_DIR / "registry.json"
    if not path.is_file():
        raise C.HardStop("缺 metadata/registry.json，请先跑 --stage=prepare")
    return json.loads(path.read_text(encoding="utf-8"))["samples"]


def select_samples(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows = load_registry()
    if getattr(args, "sample_key", None):
        wanted = {norm for norm in str(args.sample_key).split(",") if norm.strip()}
        rows = [r for r in rows if r["sample_key"] in wanted]
        missing = wanted - {r["sample_key"] for r in rows}
        if missing:
            raise C.HardStop(f"样本键不在登记表里：{sorted(missing)}")
    if getattr(args, "limit", None):
        rows = rows[: int(args.limit)]
    return rows


# --------------------------------------------------------------------------
# text
# --------------------------------------------------------------------------


def stage_text(args: argparse.Namespace) -> Dict[str, Any]:
    """S3a：官方文本 → 词单元 + RoBERTa 词级特征，落 `.cache/text/<stub>.npz`。"""
    import q1v2_assets as A
    import q1v2_build as B
    import q1v2_text as T

    rows = select_samples(args)
    out_dir = CACHE_ROOT / "text"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_dir = A.resolve_roberta_dir()
    A.verify_roberta_dir(model_dir)
    print(f"[text] RoBERTa 本地目录 {model_dir.name}  样本 {len(rows)}")
    tokenizer, model, _identity_sha = T.load_roberta(model_dir)
    model_sha = C.sha256(model_dir / "model.safetensors")

    failures: List[Dict[str, Any]] = []
    n_done = n_skip = 0
    total_words = total_chunks = 0
    max_tokens = 0
    started = time.time()

    for sample in rows:
        stem = C.safe_stem(sample["sample_key"])
        out = out_dir / f"{stem}.npz"
        previous = _read_log(stem, "text")
        done = bool(previous and previous.get("status") == "PASS")
        # 续跑判据要求**产物与成功日志同时在**。只认产物是不够的：
        # 若上一轮在「写完 NPZ、写日志前」中断，磁盘上会留下一个没有成功记录的
        # 产物，而它可能是旧版代码按旧字段布局写出来的——当成成品跳过就再也不会被修正。
        if out.is_file() and done and not args.overwrite:
            n_skip += 1
            total_words += int(previous["n_words"])
            total_chunks += int(previous["n_chunks"])
            max_tokens = max(max_tokens, int(previous["input_length"]))
            continue
        if out.is_file() and not done:
            print(f"     重建（产物在但无成功日志）：{sample['sample_key']}")
        try:
            text = sample["official_text"]
            words = T.source_words(text)
            chunks = T.whitespace_chunks(text)
            result = T.extract_text_features(text, words, tokenizer, model)
            # `tokenizer_failures` 是 dict 列表，直接 np.asarray 会得到 object 数组，
            # 存进 npz 就要求 pickle——与本套代码 allow_pickle=False 的约定冲突。
            # 落盘只放计数，明细去 JSON 日志（日志本来就是文本）。
            payload = {k: np.asarray(v) for k, v in result.items()
                       if k not in ("features", "tokenizer_failures")}
            payload["features"] = np.asarray(result["features"], dtype=np.float32)
            payload["tokenizer_failure_count"] = np.asarray(
                len(result["tokenizer_failures"]), dtype=np.int32)
            payload["words"] = np.asarray([w["text"] for w in words])
            payload["word_char_start"] = np.asarray(
                [w["char_start"] for w in words], dtype=np.int32)
            payload["word_char_end"] = np.asarray(
                [w["char_end"] for w in words], dtype=np.int32)
            payload["n_chunks"] = np.asarray(len(chunks), dtype=np.int32)
            payload["chunk_char_start"] = np.asarray(
                [c["char_start"] for c in chunks], dtype=np.int32)
            payload["chunk_char_end"] = np.asarray(
                [c["char_end"] for c in chunks], dtype=np.int32)
            payload["roberta_model_sha256"] = np.asarray(model_sha)
            B.write_npz_atomic(out, payload)

            total_words += len(words)
            total_chunks += len(chunks)
            max_tokens = max(max_tokens, int(result["input_length"]))
            n_done += 1
            _write_log(stem, "text", {
                "sample_key": sample["sample_key"], "status": "PASS",
                "n_words": len(words), "n_chunks": len(chunks),
                "input_length": int(result["input_length"]),
                "tokenizer_failure_count": len(result["tokenizer_failures"]),
                "tokenizer_failures": result["tokenizer_failures"][:20],
                "valid_words": int(np.count_nonzero(result["valid"])),
            })
        except Exception as exc:  # noqa: BLE001 —— 一条样本失败不拖停整批
            failures.append({"sample_key": sample["sample_key"],
                             "error": f"{type(exc).__name__}: {exc}"})
            _write_log(stem, "text", {
                "sample_key": sample["sample_key"], "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}"})
            print(f"  !! {sample['sample_key']}  {type(exc).__name__}: {exc}")

    summary = {
        "stage": "text", "n_selected": len(rows), "n_written": n_done,
        "n_skipped_existing": n_skip, "n_failed": len(failures),
        "sum_word_units": total_words, "sum_whitespace_chunks": total_chunks,
        "max_input_tokens": max_tokens,
        "runtime_sec": round(time.time() - started, 1),
        "failures": failures,
    }
    C.write_json(REPORTS_DIR / "stage_text.json", summary)
    return summary


# --------------------------------------------------------------------------
# align
# --------------------------------------------------------------------------


def stage_align(args: argparse.Namespace) -> Dict[str, Any]:
    """S3b：官方文本强制对齐 → 词单元半开区间，落 `.cache/align/<stub>.json`。"""
    import q1v2_align as AL
    import q1v2_assets as A
    import q1v2_media as M
    import q1v2_text as T

    rows = select_samples(args)
    out_dir = CACHE_ROOT / "align"
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = A.resolve_whisper_checkpoint()
    A.verify_whisper_checkpoint(checkpoint)
    print(f"[align] 载入对齐器 ...")
    aligner = AL.load_aligner(checkpoint)

    failures: List[Dict[str, Any]] = []
    n_done = n_skip = 0
    total_slots = total_words = total_mapping_fail = 0
    total_zero_words = 0
    n_samples_zero_words = 0
    started = time.time()

    for sample in rows:
        stem = C.safe_stem(sample["sample_key"])
        out = out_dir / f"{stem}.json"
        previous = _read_log(stem, "align")
        done = bool(previous and previous.get("status") == "PASS")
        if out.is_file() and done and not args.overwrite:
            n_skip += 1
            cached = json.loads(out.read_text(encoding="utf-8"))
            diag = cached["diagnostics"]
            total_slots += int(diag["n_aligner_slots"])
            total_words += int(diag["n_official_words"])
            total_mapping_fail += int(diag["mapping_fail_count"])
            total_zero_words += int(diag["zero_duration_word_count"])
            n_samples_zero_words += int(diag["zero_duration_word_count"] > 0)
            continue
        try:
            path = C.PROJECT_ROOT / sample["source_video_relpath"]
            text = sample["official_text"]
            words = T.source_words(text)
            chunks = T.whitespace_chunks(text)
            chunk_to_word = T.chunk_to_word_index(chunks, words)

            probe = M.probe_presentation_starts(path)
            waveform, audio_info = M.decode_audio_16k(
                path, float(probe["audio_stream_start_sec"]))
            align_audio, _conv = AL.prepare_alignment_audio(waveform)
            result = AL.align_official_text(aligner, align_audio, text)

            a_start = float(audio_info["resampled_start_sec"])
            a_end = float(audio_info["resampled_end_sec"])
            intervals = AL.chunks_to_word_intervals(
                result["slots"], chunks, chunk_to_word, len(words), a_start, a_end)
            diag = AL.structural_diagnostics(intervals, a_start, a_end)
            status = AL.mapping_status(diag)

            payload = {
                "sample_key": sample["sample_key"],
                "alignment_source": C.ALIGN_SOURCE,
                "alignment_params": dict(C.ALIGN_PARAMS),
                "aligner_name": C.ALIGNER_NAME,
                # 跑次指纹：由槽内容本身决定，重跑同一条应得到同一个值
                "run_sha256": C.sha256_text(json.dumps(
                    _jsonable(result["slots"]), ensure_ascii=False, sort_keys=True)),
                "n_segments": int(result["n_segments"]),
                "intervals": {
                    "word_start_sec": intervals["word_start_sec"],
                    "word_end_sec": intervals["word_end_sec"],
                    "word_time_valid": intervals["word_time_valid"],
                    "slot_to_word": intervals["slot_to_word"],
                    "slot_to_chunk": intervals["slot_to_chunk"],
                    "n_slots": intervals["n_slots"],
                    "n_words": intervals["n_words"],
                    "mapping_method": intervals["mapping_method"],
                    "mapping_diagnostics": intervals["mapping_diagnostics"],
                },
                "diagnostics": diag,
                "mapping_status": status,
                "audio_coverage_sec": [a_start, a_end],
            }
            temp = out.with_suffix(".json.tmp")
            temp.write_text(json.dumps(_jsonable(payload), ensure_ascii=False,
                                       indent=1, sort_keys=True), encoding="utf-8")
            temp.replace(out)

            total_slots += int(diag["n_aligner_slots"])
            total_words += int(diag["n_official_words"])
            total_mapping_fail += int(diag["mapping_fail_count"])
            total_zero_words += int(diag["zero_duration_word_count"])
            n_samples_zero_words += int(diag["zero_duration_word_count"] > 0)
            n_done += 1
            _write_log(stem, "align", {
                "sample_key": sample["sample_key"], "status": "PASS", **diag,
                "mapping_status": status,
            })
        except Exception as exc:  # noqa: BLE001
            failures.append({"sample_key": sample["sample_key"],
                             "error": f"{type(exc).__name__}: {exc}"})
            _write_log(stem, "align", {
                "sample_key": sample["sample_key"], "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}"})
            print(f"  !! {sample['sample_key']}  {type(exc).__name__}: {exc}")

    summary = {
        "stage": "align", "n_selected": len(rows), "n_written": n_done,
        "n_skipped_existing": n_skip, "n_failed": len(failures),
        "sum_aligner_slots": total_slots, "sum_official_words": total_words,
        "sum_mapping_fail": total_mapping_fail,
        "sum_zero_duration_word_count": total_zero_words,
        "n_samples_with_zero_duration_words": n_samples_zero_words,
        "runtime_sec": round(time.time() - started, 1),
        "failures": failures,
    }
    C.write_json(REPORTS_DIR / "stage_align.json", summary)
    return summary


# --------------------------------------------------------------------------
# media
# --------------------------------------------------------------------------


def media_one(args: argparse.Namespace) -> Dict[str, Any]:
    """子进程入口：跑一条样本的媒体阶段。

    刻意**不加载 RoBERTa**——文本特征已在 `.cache/text` 里，
    子进程只需 numpy + PyAV + MediaPipe + openSMILE。
    """
    import q1v2_build as B

    sample = None
    for row in load_registry():
        if row["sample_key"] == args.sample_key:
            sample = row
            break
    if sample is None:
        raise C.HardStop(f"样本键不在登记表里：{args.sample_key}")

    stem = C.safe_stem(sample["sample_key"])
    text_cache_path = CACHE_ROOT / "text" / f"{stem}.npz"
    align_cache_path = CACHE_ROOT / "align" / f"{stem}.json"
    if not text_cache_path.is_file():
        raise C.HardStop(f"缺文本缓存 {text_cache_path.name}，请先跑 --stage=text")
    if not align_cache_path.is_file():
        raise C.HardStop(f"缺对齐缓存 {align_cache_path.name}，请先跑 --stage=align")

    with np.load(text_cache_path, allow_pickle=False) as handle:
        text_cache = {k: handle[k] for k in handle.files}
    cached = json.loads(align_cache_path.read_text(encoding="utf-8"))
    align_cache = {
        "alignment_source": cached["alignment_source"],
        "alignment_params": cached["alignment_params"],
        "run_sha256": cached["run_sha256"],
        "mapping_status": cached["mapping_status"],
        "diagnostics": cached["diagnostics"],
        "intervals": {
            k: np.asarray(v) for k, v in cached["intervals"].items()
        },
    }

    config_sha = C.config_digest()
    try:
        import psutil
        sample["process_rss_bytes"] = int(
            psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        sample["process_rss_bytes"] = ""

    row = B.media_stage(sample=sample, text_cache=text_cache,
                        align_cache=align_cache, feature_dir=FEATURES_DIR,
                        config_sha=config_sha,
                        allow_overwrite=bool(args.overwrite))
    _write_log(stem, "media", {"sample_key": sample["sample_key"],
                               "status": "PASS", "row": row})
    return row


def stage_media(args: argparse.Namespace) -> Dict[str, Any]:
    """S3c：逐样本媒体阶段。默认起子进程（原生栈隔离）。"""
    rows = select_samples(args)
    isolate = getattr(args, "isolate", None)
    if isolate is None:
        isolate = True
    timeout = int(getattr(args, "timeout", 900) or 900)
    retries = int(getattr(args, "retries", 2) or 0)

    n_done = n_skip = 0
    failures: List[Dict[str, Any]] = []
    started = time.time()
    total_frames = total_faces = 0
    n_zero_face = 0

    for index, sample in enumerate(rows, 1):
        stem = C.safe_stem(sample["sample_key"])
        out = FEATURES_DIR / f"{stem}.npz"
        if out.is_file() and not args.overwrite:
            n_skip += 1
            existing = _read_log(stem, "media")
            if existing and existing.get("status") == "PASS":
                row = existing["row"]
                total_frames += int(row.get("video_frame_count") or 0)
                total_faces += int(row.get("face_detected_frame_count") or 0)
                n_zero_face += int(not row.get("face_detected_frame_count"))
            continue

        print(f"  [{index}/{len(rows)}] {sample['sample_key']}")
        if not isolate:
            try:
                row = media_one(argparse.Namespace(
                    sample_key=sample["sample_key"], overwrite=args.overwrite))
                total_frames += int(row["video_frame_count"])
                total_faces += int(row["face_detected_frame_count"])
                n_zero_face += int(not row["face_detected_frame_count"])
                n_done += 1
            except Exception as exc:  # noqa: BLE001
                failures.append({"sample_key": sample["sample_key"],
                                 "error": f"{type(exc).__name__}: {exc}"})
                _write_log(stem, "media", {
                    "sample_key": sample["sample_key"], "status": "HARD_STOP",
                    "error": f"{type(exc).__name__}: {exc}"})
            continue

        last_error = ""
        for attempt in range(retries + 1):
            command = [_child_python(), str(Path(__file__).resolve()),
                       "--stage=media", f"--sample-key={sample['sample_key']}"]
            if args.overwrite:
                command.append("--overwrite")
            try:
                proc = subprocess.run(command, capture_output=True, text=True,
                                      timeout=timeout, env=_child_env(),
                                      cwd=str(C.PROJECT_ROOT))
                if proc.returncode == 0:
                    last_error = ""
                    break
                last_error = (proc.stderr or proc.stdout or "").strip()[-800:]
            except subprocess.TimeoutExpired:
                last_error = f"TimeoutExpired after {timeout}s"
            print(f"      重试 {attempt + 1}/{retries}：{last_error[:200]}")

        if last_error:
            failures.append({"sample_key": sample["sample_key"],
                             "error": last_error})
            _write_log(stem, "media", {
                "sample_key": sample["sample_key"], "status": "HARD_STOP",
                "error": last_error})
            continue
        n_done += 1
        existing = _read_log(stem, "media")
        if existing and existing.get("status") == "PASS":
            row = existing["row"]
            total_frames += int(row.get("video_frame_count") or 0)
            total_faces += int(row.get("face_detected_frame_count") or 0)
            n_zero_face += int(not row.get("face_detected_frame_count"))

    summary = {
        "stage": "media", "n_selected": len(rows), "n_written": n_done,
        "n_skipped_existing": n_skip, "n_failed": len(failures),
        "isolated_subprocess": bool(isolate),
        "sum_video_frame_count": total_frames,
        "sum_face_detected_frame_count": total_faces,
        "n_samples_without_face": n_zero_face,
        "runtime_sec": round(time.time() - started, 1),
        "failures": failures,
    }
    C.write_json(REPORTS_DIR / "stage_media.json", summary)
    return summary


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------


def stage_tables(args: argparse.Namespace) -> Dict[str, Any]:
    """S4：汇总逐样本日志 → manifest / results_100 / 两张审计表。"""
    import q1v2_build as B

    rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    for sample in load_registry():
        stem = C.safe_stem(sample["sample_key"])
        log = _read_log(stem, "media")
        if not log or log.get("status") != "PASS":
            missing.append(sample["sample_key"])
            continue
        rows.append(log["row"])

    if missing:
        print(f"[tables] !! {len(missing)} 条样本尚无成功的媒体产物，"
              f"表只写已完成的 {len(rows)} 行")

    result = B.rebuild_output_tables(OUTPUT_ROOT, rows)

    # 路由审计表与异常台账
    mode_counts: Dict[str, int] = {}
    for row in rows:
        mode_counts[row["alignment_mode"]] = mode_counts.get(row["alignment_mode"], 0) + 1
    C.write_csv(AUDIT_DIR / "v2_route_100.csv", rows, (
        "sample_key", "alignment_mode", "alignment_granularity",
        "text_av_time_mapping_status", "text_audio_correspondence",
        "content_assertion", "word_time_basis", "alignment_source",
        "n_official_words", "n_structurally_valid_words",
        "audio_speech_valid", "video_frame_count", "face_valid_ratio",
    ))

    anomalies = []
    for row in rows:
        notes = []
        if row["alignment_mode"] != C.MODE_UNCERTAIN_REVIEW:
            notes.append(row["alignment_mode"])
        if int(row["alignment_mapping_fail_count"] or 0) > 0:
            notes.append("mapping_fail")
        if not row["face_detected_frame_count"]:
            notes.append("no_face_detected")
        if float(row["word_union_coverage_ratio"] or 0) < 0.5:
            notes.append("word_coverage_below_0.5")
        if notes:
            anomalies.append({**{k: row[k] for k in (
                "sample_key", "alignment_mode", "n_official_words",
                "n_structurally_valid_words", "video_frame_count",
                "face_detected_frame_count", "face_valid_ratio",
                "word_union_coverage_ratio", "original_effective_duration_sec")},
                "notes": "|".join(notes),
                "counts_as_deletion": 0,
                "action": "仅标注，不删除"})
    C.write_csv(AUDIT_DIR / "anomaly_ledger.csv", anomalies, (
        "sample_key", "alignment_mode", "n_official_words",
        "n_structurally_valid_words", "video_frame_count",
        "face_detected_frame_count", "face_valid_ratio",
        "word_union_coverage_ratio", "original_effective_duration_sec",
        "notes", "counts_as_deletion", "action",
    ))

    summary = {
        "stage": "tables", **result,
        "n_missing_media": len(missing), "missing": missing,
        "alignment_mode_counts": mode_counts,
        "content_assertion_positive": sum(
            int(r["content_assertion"] or 0) for r in rows),
        "word_time_basis_own": sum(
            1 for r in rows if r["word_time_basis"] == C.WORD_TIME_BASIS_OWN),
        "audio_speech_valid_counts": {
            str(v): sum(1 for r in rows if int(r["audio_speech_valid"]) == v)
            for v in (1, 0, -1)},
        "n_anomalies_logged": len(anomalies),
    }
    C.write_json(REPORTS_DIR / "stage_tables.json", summary)
    return summary


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def stage_verify(args: argparse.Namespace) -> Dict[str, Any]:
    import q1v2_verify as V
    return V.verify_all(quick=bool(getattr(args, "quick", False)),
                        correspondence_csv=getattr(args, "correspondence_csv", None))


def _run_all(args: argparse.Namespace) -> None:
    for stage in STAGES:
        print(f"\n{'=' * 72}\n[{stage}]")
        args.stage = stage
        if stage == "media":
            stage_media(args)
        elif stage == "text":
            stage_text(args)
        elif stage == "align":
            stage_align(args)
        elif stage == "prepare":
            stage_prepare(args)
        elif stage == "tables":
            stage_tables(args)
        elif stage == "verify":
            stage_verify(args)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="问题一 v2 全流程（原生时间分辨率，不做 50 槽等分）")
    parser.add_argument("--stage", default="all",
                        choices=list(STAGES) + ["all"])
    parser.add_argument("--sample-key", default=None,
                        help="只跑指定样本（多条用逗号分隔），用于调试与重跑")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true",
                        help="默认跳过已存在的产物以支持续跑")
    parser.add_argument("--allow-download", action="store_true",
                        help="允许预取 MediaPipe 权重（仅首次需要）")
    isolate = parser.add_mutually_exclusive_group()
    isolate.add_argument("--isolate", dest="isolate", action="store_true",
                         default=None, help="媒体阶段强制子进程（默认）")
    isolate.add_argument("--no-isolate", dest="isolate", action="store_false",
                         help="媒体阶段改为批内直跑")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--quick", action="store_true",
                        help="核验阶段只跑轻量项")
    parser.add_argument("--correspondence-csv", default=None,
                        help="S6 开发期回归：外部人工证据 CSV（不进交付物）")
    args = parser.parse_args(argv)

    if args.stage == "all":
        _run_all(args)
        return 0

    handlers = {
        "prepare": stage_prepare, "text": stage_text, "align": stage_align,
        "media": stage_media, "tables": stage_tables, "verify": stage_verify,
    }
    # 子进程入口走单样本路径
    if args.stage == "media" and args.sample_key and args.sample_key in {
            r["sample_key"] for r in load_registry()} and args.limit is None:
        row = media_one(args)
        print(f"OK {row['sample_key']}  {row['npz_path']}  "
              f"{row['npz_bytes']} B  {row['runtime_sec']} s")
        return 0

    summary = handlers[args.stage](args)
    print(f"\n[{args.stage}] " + json.dumps(
        _jsonable({k: v for k, v in summary.items() if k != "failures"}),
        ensure_ascii=False))
    if summary.get("failures"):
        print(f"  失败 {len(summary['failures'])} 条")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
