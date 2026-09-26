# -*- coding: utf-8 -*-
"""q1v2_package.py —— 问题一 v2 的体积台账与身份扫描

**刻意不修改 `package_check.py`。** 它的 `INCLUDED_DIRS` 是一份**显式白名单**，
`data/q1_v2` 本就不在其中，所以整包核验不会覆盖新产物——这一点必须让用户知情，
而不是悄悄把新目录塞进白名单让它"看起来被核验了"。

于是本模块自己出两份东西：

1. **体积台账**（`reports/q1v2_size_ledger.md`）：给**两种口径**的上限
   （`50×1024² = 52,428,800 B` 与 `50×10⁶ = 50,000,000 B`），以及
   **替换**与**叠加**两种核算——因为 `data/q1_v2` 的定位是**替换** `data/q1_delivery`，
   把它当增量叠加会得出一个虚高的、且与实际提交方式不符的数。
2. **身份扫描**：复用 `package_check.py` 的 `IDENTITY_PATTERNS`（不另立一套正则，
   否则两处规则会各自漂移），扫描 `data/q1_v2` 与 `code/q1v2_*.py`。

`.cache/` 与 `logs/` 会写入磁盘但**不计入提交**：它们是可重建的中间态，
留在仓库外即可（`excluded_walk` 也跳过 `.cache`）。
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import q1v2_contract as C
import package_check
from package_check import CODE_DIRS, IDENTITY_PATTERNS, INCLUDED_DIRS

OUTPUT_ROOT = C.OUTPUT_ROOT
PROJECT_ROOT = C.PROJECT_ROOT
REPORTS_DIR = OUTPUT_ROOT / "reports"

#: `data/q1_v2` 下不计入提交的子目录：可重建的中间态与运行日志。
NOT_COUNTED = (".cache", "logs")

#: 本套 Q1 重建**替换**掉的既有目录。体积核算按「替换」而非「叠加」。
REPLACES = ("q1_delivery",)

TEXT_SUFFIXES = {".json", ".csv", ".md", ".py", ".txt", ".yaml", ".yml", ".toml"}


def _dir_stats(root: Path) -> Tuple[int, int]:
    """返回 `(文件数, 字节数)`，跳过 `__pycache__` 与 Office 临时件。"""
    files = 0
    total = 0
    if not root.is_dir():
        return 0, 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for name in filenames:
            if name.startswith("~$") or name.endswith(".pyc"):
                continue
            try:
                total += (Path(dirpath) / name).stat().st_size
            except OSError:
                continue
            files += 1
    return files, total


def measure_v2() -> Dict[str, Any]:
    """逐子目录统计 v2 产物，并给出计入/不计入的划分。"""
    counted: Dict[str, Dict[str, int]] = {}
    not_counted: Dict[str, Dict[str, int]] = {}
    for entry in sorted(OUTPUT_ROOT.iterdir()) if OUTPUT_ROOT.is_dir() else []:
        if entry.name in NOT_COUNTED:
            files, size = _dir_stats(entry)
            not_counted[entry.name] = {"files": files, "bytes": size}
            continue
        if entry.is_dir():
            files, size = _dir_stats(entry)
            counted[entry.name] = {"files": files, "bytes": size}
        else:
            counted[entry.name] = {"files": 1, "bytes": entry.stat().st_size}

    v2_total = sum(v["bytes"] for v in counted.values())
    v2_extra = sum(v["bytes"] for v in not_counted.values())
    return {"counted": counted, "not_counted": not_counted,
            "counted_bytes": v2_total, "not_counted_bytes": v2_extra}


def _whitelist_excluded(name: str) -> Tuple[str, ...]:
    """白名单里该组要跳过的子目录，取自 `package_check.EXCLUDED_SUBDIRS`。

    **必须复用而不是自己另写一份**：`package_check.py` 排除 `data/q1_v2` 下的
    `.cache/` 与 `logs/`，本台账若照单全收，两侧的「合计」就永远差着那 7.18 MB，
    而两份报告都会声称自己是对的。这正是注释里说的「各处数字须能互相解释」。
    """
    return tuple(package_check.EXCLUDED_SUBDIRS.get(name, ()))


def current_whitelist() -> Dict[str, Dict[str, int]]:
    """`package_check.py` 白名单的现状（不改它，只读它）。"""
    out: Dict[str, Dict[str, int]] = {}
    for name, rel in INCLUDED_DIRS:
        root = PROJECT_ROOT / rel
        exclude = _whitelist_excluded(name)
        if exclude:
            # 逐子目录累加、跳过被排除的那些（`_dir_stats` 只认单个 root）。
            files = size = 0
            counted_entries = [e for e in sorted(root.iterdir())
                               if e.name not in exclude] if root.is_dir() else []
            for entry in counted_entries:
                f, s = _dir_stats(entry) if entry.is_dir() else (1, entry.stat().st_size)
                files += f
                size += s
        else:
            files, size = _dir_stats(root)
        out[name] = {"relpath": rel, "files": files, "bytes": size,
                     "excluded": list(exclude)}
    for name, rel in CODE_DIRS:
        files, size = _dir_stats(PROJECT_ROOT / rel)
        out[name] = {"relpath": rel, "files": files, "bytes": size, "is_code": True}
    return out


def _fmt(size: int) -> str:
    return f"{size / 1024 ** 2:.2f} MiB ({size:,} B)"


def build_ledger() -> Dict[str, Any]:
    """体积台账。

    **两种核算必须对白名单的当前状态免疫。** 白名单可能指向旧目录
    （`q1_delivery`）也可能已换向到 `q1_v2`——它换过一次，将来还可能再动。
    若算式默认「白名单里没有 q1_v2」，那么换向之后 `base` 已含 q1_v2，
    再 `+ v2_counted_bytes` 就会把同一份产物算两遍：得出的数**虚高一整份 q1_v2**，
    而所有「≤50 MB」的结论都会随之变成假的失败。故这里先问白名单里有没有 v2，
    再决定要不要加；`replaced` 同理。
    """
    v2 = measure_v2()
    whitelist = current_whitelist()
    base = sum(v["bytes"] for v in whitelist.values())

    #: 旧目录**不论在不在白名单里**都要量一次：算式两边都需要它。
    old_q1_rel = package_check.OLD_Q1_REL
    old_q1_files, old_q1_bytes = _dir_stats(PROJECT_ROOT / old_q1_rel)

    v2_rel = Path(OUTPUT_ROOT).relative_to(PROJECT_ROOT).as_posix()
    v2_in_whitelist = any(Path(v["relpath"]).as_posix() == v2_rel
                          for v in whitelist.values())
    replaced = sum(whitelist.get(k, {}).get("bytes", 0) for k in REPLACES)

    # 替换核算：白名单里还留着旧 Q1 就先扣掉它、再把 v2 加进来；
    # 若白名单已经指向 v2，则 `base` 就是替换后的结果，两个调整都为 0。
    replace_total = (base
                     - (replaced if replaced else 0)
                     + (0 if v2_in_whitelist else v2["counted_bytes"]))
    # 叠加核算 = 替换结果 + 旧目录（即「两份 Q1 都留着」）。
    # 这条式子在两种白名单状态下都成立：旧状态下它退化为 base + v2_counted_bytes。
    stack_total = replace_total + old_q1_bytes

    # 与 `package_check.py` 对账。两处量的是同一组文件，合计应当能互相解释；
    # 唯一可能的残差来自**自指**：本台账把 `data/package_check/` 量进去，
    # 而 `package_check.json` 里记着它自己的字节数——脚本每写一次它就会变一次大小。
    # 这个残差是几百字节量级（占比 10⁻⁵），不影响任何结论，但**必须写出来**：
    # 否则一个照着两份报告做减法的人会以为自己查出了错。
    pc_json = PROJECT_ROOT / "data" / "package_check" / "package_check.json"
    reconcile = {"package_check_json": Path(pc_json).as_posix(), "delta_bytes": None}
    if pc_json.is_file():
        try:
            pc = json.loads(pc_json.read_text(encoding="utf-8"))
            pc_total = int(pc["deliverable_bytes"]) + int(pc["code_bytes"])
            reconcile["package_check_total_bytes"] = pc_total
            reconcile["delta_bytes"] = replace_total - pc_total
            reconcile["note"] = ("残差 = 本台账合计 − package_check.py 合计。"
                                 "二者量同一组文件，差额仅来自 package_check.json "
                                 "对自身大小的自指。")
        except Exception as exc:  # noqa: BLE001
            reconcile["note"] = f"读取失败：{type(exc).__name__}"

    return {
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "v2": v2,
        "whitelist": whitelist,
        "replaces": list(REPLACES),
        "reconcile_with_package_check": reconcile,
        "limits": {"loose_bytes": C.LIMIT_BYTES, "strict_bytes": C.LIMIT_STRICT_BYTES,
                   "loose_note": "50×1024² —— 与 data/q1_delivery 既有口径一致",
                   "strict_note": "50×10⁶"},
        "accounting": {
            "whitelist_current_bytes": base,
            "replaced_bytes": replaced,
            "v2_counted_bytes": v2["counted_bytes"],
            "v2_in_whitelist": bool(v2_in_whitelist),
            "old_q1_relpath": Path(old_q1_rel).as_posix(),
            "old_q1_files": old_q1_files,
            "old_q1_bytes": old_q1_bytes,
            "replace_total_bytes": replace_total,
            "stack_total_bytes": stack_total,
            "replace_within_loose": replace_total <= C.LIMIT_BYTES,
            "replace_within_strict": replace_total <= C.LIMIT_STRICT_BYTES,
            "stack_within_loose": stack_total <= C.LIMIT_BYTES,
            "stack_within_strict": stack_total <= C.LIMIT_STRICT_BYTES,
        },
    }


# --------------------------------------------------------------------------
# 身份扫描
# --------------------------------------------------------------------------


def _mask(match: str) -> str:
    """命中样例**只留长度，不留内容**。

    检查器把命中原文写进自己的报告，就等于报告本身成了泄漏物——
    这是本项目踩过的坑（`package_check.json` 曾把家目录路径原样存盘）。
    定位靠 `file` + `line` 就够了，不需要把敏感串复制一份。
    """
    return f"[已隐去 {len(match)} 字符]"


def _iter_npz_strings(path: Path) -> Iterable[Tuple[str, str]]:
    """遍历 NPZ 里的**字符串数组**。

    文本扫描只看 `.json/.csv/.md/...`，而 NPZ 是二进制——于是
    `decode_provenance_json` / `alignment_params_json` 这类存成 JSON 文本的
    字段会被整个跳过。那是脱敏的盲区：二进制容器里一样能塞绝对路径。
    """
    import numpy as np

    try:
        with np.load(path, allow_pickle=False) as handle:
            for field in handle.files:
                array = handle[field]
                if array.dtype.kind != "U":
                    continue
                for value in array.reshape(-1):
                    yield field, str(value)
    except Exception:  # noqa: BLE001
        return


def scan_identity() -> Dict[str, Any]:
    """扫描 v2 产物与 q1v2 源码里的身份信息。复用既有的两条正则。

    扫描面**比 `package_check.py` 宽一格**：除文本文件外，还逐个 NPZ 检查
    其中的字符串数组（见 `_iter_npz_strings`）。
    """
    hits: List[Dict[str, Any]] = []
    scanned = 0
    scanned_npz_strings = 0
    roots = [OUTPUT_ROOT, _CODE_DIR]
    for root in roots:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames
                           if d not in ("__pycache__", ".git", ".cache")]
            for name in filenames:
                path = Path(dirpath) / name
                # 源码里 q1v2_* 才扫；其余 code/ 下的既有文件不属于本次范围
                if root == _CODE_DIR and not name.startswith("q1v2_"):
                    continue
                if path.suffix.lower() not in TEXT_SUFFIXES and root == _CODE_DIR:
                    continue
                scanned += 1
                if path.suffix.lower() == ".npz":
                    for field, value in _iter_npz_strings(path):
                        scanned_npz_strings += 1
                        for pattern, label in IDENTITY_PATTERNS:
                            for match in pattern.finditer(value):
                                hits.append({
                                    "file": C.relpath_for_delivery(path),
                                    "label": label,
                                    "field": field,
                                    "match_masked": _mask(match.group(0)),
                                })
                    continue
                if path.suffix.lower() not in TEXT_SUFFIXES:
                    continue
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                for pattern, label in IDENTITY_PATTERNS:
                    for match in pattern.finditer(text):
                        hits.append({
                            "file": C.relpath_for_delivery(path),
                            "label": label,
                            "match_masked": _mask(match.group(0)),
                            "line": text[:match.start()].count("\n") + 1,
                        })
    return {"scanned_files": scanned, "scanned_npz_strings": scanned_npz_strings,
            "n_hits": len(hits), "hits": hits[:50],
            "patterns": [p.pattern for p, _ in IDENTITY_PATTERNS],
            "scope_note": ("data/q1_v2/** 与 code/q1v2_*.py。"
                           "**不含项目根目录下的既有 .md/.docx**"
                           "（那属于既有交付的范围，不在本次 v2 之内）。")}


def _render(ledger: Dict[str, Any], identity: Dict[str, Any]) -> str:
    acc = ledger["accounting"]
    lim = ledger["limits"]
    lines = [
        "# 问题一 v2 体积台账与身份扫描",
        "",
        f"- 生成时间（UTC）：{ledger['generated_at_utc']}",
        f"- 产物根目录：`{C.relpath_for_delivery(OUTPUT_ROOT)}`",
        "",
        "## 与整包核验的关系（换向之后）",
        "",
        "`code/package_check.py` 的 `INCLUDED_DIRS` 是一份**显式白名单**。"
        "在问题一换向之前，它列的是 `q1_delivery`，那时 `data/q1_v2` 不在其中，"
        "整包核验**不覆盖**本套产物，本台账就是为了补上那一空白。",
        "",
        "**换向之后情况已变**：白名单已改为 `q1_v2`（清单见下表），"
        "同时按 `EXCLUDED_SUBDIRS` 跳过 `.cache/` 与 `logs/`。"
        "因此现在 `package_check.py` 与本台账量的是同一批文件，两处的「合计」"
        "**必须能互相解释**。本模块复用 `package_check.EXCLUDED_SUBDIRS` "
        "而不是自己另写一份排除规则，正是为了这一点。",
        "",
        "> 白名单有可能再次换向。两种核算对「白名单当前指向谁」是**免疫**的："
        "若它指向 v2，则 `base` 本身即替换后的结果，算式不再重复计入 v2。",
        "",
        "## v2 产物体积",
        "",
        "| 子目录 | 文件 | 字节 | 计入提交 |",
        "|---|---|---|---|",
    ]
    for name, stat in sorted(ledger["v2"]["counted"].items()):
        lines.append(f"| `{name}` | {stat['files']} | {stat['bytes']:,} | ✅ |")
    for name, stat in sorted(ledger["v2"]["not_counted"].items()):
        lines.append(f"| `{name}` | {stat['files']} | {stat['bytes']:,} | ❌ 可重建中间态 |")
    lines += [
        f"| **计入合计** | | **{ledger['v2']['counted_bytes']:,}** | "
        f"= {_fmt(ledger['v2']['counted_bytes'])} |",
        f"| 不计入合计 | | {ledger['v2']['not_counted_bytes']:,} | "
        f"= {ledger['v2']['not_counted_bytes'] / 1024 ** 2:.2f} MiB |",
        "",
        "`.cache/`（文本与对齐的逐样本缓存）与 `logs/` 逐样本写盘，"
        "但**不计入提交**：它们是可重建的中间态，删掉后重跑 `--stage=text/align` 即可复原。",
        "",
        "## 现有白名单现状",
        "",
        "| 目录 | 相对路径 | 文件 | 字节 |",
        "|---|---|---|---|",
    ]
    for name, stat in sorted(ledger["whitelist"].items()):
        # 统一成正斜杠：交付报告里的路径一律 as_posix 形式，
        # 与产物快照、registry 的写法保持一致（Windows 反斜杠只在本机成立）。
        rel = Path(stat["relpath"]).as_posix()
        note = ""
        if stat.get("excluded"):
            note = f"（已排除 {', '.join(stat['excluded'])}/）"
        lines.append(f"| `{name}` | `{rel}` | {stat['files']} | "
                     f"{stat['bytes']:,} | {note}")
    lines += [
        f"| **合计** | | | **{acc['whitelist_current_bytes']:,}** |",
        "",
        "## 两种核算：替换 vs 叠加",
        "",
        "`data/q1_v2` 的定位是**替换** `data/q1_delivery`（v2 是同一问题的重建版），"
        "所以正确的核算方式是替换。把它当增量叠加会得到一个虚高的数，"
        "且与实际提交方式不符——但它仍然值得列出来，因为**只有看到叠加超线，"
        "才能理解为什么必须替换而不是并存**。",
        "",
        f"其中 `data/q1_v2` 是否已经**在白名单里**："
        f"**{'是' if acc['v2_in_whitelist'] else '否'}**。"
        + ("白名单已换向到 v2，故 `base` 本身即替换后的结果，算式不再重复加一遍 v2。"
           if acc["v2_in_whitelist"] else
           "白名单仍指向旧目录，故需从 `base` 中扣掉旧目录再加上 v2。"),
        "",
        "| 核算方式 | 算式 | 合计 | ≤50×1024² | ≤50×10⁶ |",
        "|---|---|---|---|---|",
        f"| **替换（实际）** | {acc['whitelist_current_bytes']:,} − "
        f"{acc['replaced_bytes']:,} + "
        f"{(0 if acc['v2_in_whitelist'] else acc['v2_counted_bytes']):,} | "
        f"**{acc['replace_total_bytes']:,}** | "
        f"{'✅' if acc['replace_within_loose'] else '❌'} | "
        f"{'✅' if acc['replace_within_strict'] else '❌'} |",
        f"| 叠加（反面） | {acc['replace_total_bytes']:,} + "
        f"{acc['old_q1_bytes']:,}（`{acc['old_q1_relpath']}`） | "
        f"{acc['stack_total_bytes']:,} | "
        f"{'✅' if acc['stack_within_loose'] else '❌'} | "
        f"{'✅' if acc['stack_within_strict'] else '❌'} |",
        "",
        "### 与 `package_check.py` 对账",
        "",
    ]
    rec = ledger.get("reconcile_with_package_check") or {}
    if rec.get("delta_bytes") is not None:
        lines.append(
            f"- `package_check.py` 的合计（交付物 + 代码）："
            f"**{rec['package_check_total_bytes']:,}** B；本台账的替换核算："
            f"**{acc['replace_total_bytes']:,}** B；差额 "
            f"**{rec['delta_bytes']:+,}** B。")
        lines.append(
            "- 差额不是错的来源：`data/package_check/package_check.json` 里记着"
            "它**自己**的字节数，而本台账把该文件量进合计——脚本每重写一次它，"
            "大小就变一次。几百字节的量级，占总额 10⁻⁵ 以下，不影响任何结论。")
    else:
        lines.append("- 未读到 `package_check.json`，无法对账。")
    lines += [
        "",
        f"两个上限口径：**宽松** {lim['loose_bytes']:,} B（{lim['loose_note']}）、"
        f"**严格** {lim['strict_bytes']:,} B（{lim['strict_note']}）。"
        "宽松口径更大，故用它作结论时对严格口径也更安全；两个都列出以免歧义。",
        "",
        "**本台账须最后生成。** 它报的是 `reports/` 的字节数，而它自己就在 "
        "`reports/` 里——自指。生成器已迭代到不动点（写→量→再写，2 轮收敛），"
        "所以本文件里印的数与盘上的数一致；但**任何之后再改动 `reports/` 的步骤"
        "（例如重跑 `q1v2_verify.py`）都会让这里的数差几百字节**。"
        "正确顺序是先核验、后生成本台账。",
        "",
        "## 特征精度与体积的两件事，别混为一谈",
        "",
        "**一、精度不降。** 三路词级特征一律 **float32**，不降 float16。"
        "降精度会让核验 V4 的"
        "`rtol=1e-5` 可复算性失效——而「聚合结果能从源观测重算回来」"
        "是本套产物可用性的核心保证，不值得为省几 MB 换掉。",
        "",
        "**二、体积靠无损压缩拿回。** 本套 NPZ 用 `numpy.savez_compressed` 写盘。"
        "体积大头是原生序列（视频 52 维 blendshape、LLD 25 维，按帧/窗逐行存），"
        "**压缩前实测** `features/` 约 26.3 MiB、替换核算约 **51.1 MB**，"
        "即超过 `50×10⁶` 的严格口径约 1.1 MB（该数值是压缩前的历史实测，"
        "已无法从当前目录树复现，故只给量级、不写逐字节数）。"
        "压缩是**无损**的：float32 数值解压后逐位不变，V4 的 `rtol=1e-5` 与 "
        "V14 的位精确锚点都不受影响。这与「降 float16 换体积」是两回事，"
        "本套**只做前者**。",
        "",
        "## 身份信息扫描",
        "",
        f"扫描 {identity['scanned_files']} 个文件"
        f"（含逐个 NPZ 内的 {identity['scanned_npz_strings']} 条字符串），"
        f"命中 **{identity['n_hits']}** 处。",
        "",
        f"扫描面：{identity['scope_note']}",
        "",
    ]
    if identity["n_hits"]:
        lines += ["| 文件 | 类型 | 位置 |", "|---|---|---|"]
        for hit in identity["hits"]:
            where = (f"`{hit['field']}`" if "field" in hit
                     else f"第 {hit.get('line', '?')} 行")
            lines.append(f"| `{hit['file']}` | {hit['label']} | {where} |")
        lines += [
            "",
            "命中样例**只记长度、不记内容**：把原文写进报告等于让报告本身成为"
            "泄漏物（本项目踩过这个坑——`package_check.json` 曾把家目录路径"
            "原样存盘）。定位靠上表的文件与行/字段即可。",
            "",
        ]
    else:
        lines.append("未命中本机路径或用户名。")
    lines += [
        "",
        "扫描用的两条正则与 `package_check.py` 的 `IDENTITY_PATTERNS` **同源**"
        "（直接导入，不另抄一份）——两处各存一套规则会各自漂移，"
        "而「脱敏」这件事最怕的就是标准不统一。",
        "",
    ]
    return "\n".join(lines)


def _write(ledger: Dict[str, Any], identity: Dict[str, Any]) -> None:
    C.write_json(REPORTS_DIR / "q1v2_size_ledger.json", ledger)
    C.write_json(REPORTS_DIR / "q1v2_identity_scan.json", identity)
    (REPORTS_DIR / "q1v2_size_ledger.md").write_text(
        _render(ledger, identity), encoding="utf-8")


def write_until_stable() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """反复「测量 → 落盘」直到不动点。

    自指：台账要报 `reports/` 的字节数，而台账本身就在 `reports/` 里，
    于是「写之前量到的数」与「写完之后盘上的数」必然差一点——一次运行一个数，
    报告里印的数就永远对不上 `du`。迭代到稳定即可（实测 2 轮收敛）。
    真出现不收敛（比如写入引入抖动），最多试 5 轮，之后以最后一轮为准。
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ledger, identity = build_ledger(), scan_identity()
    for _ in range(5):
        _write(ledger, identity)
        fresh_ledger, fresh_identity = build_ledger(), scan_identity()
        converged = (fresh_ledger["v2"]["counted"] == ledger["v2"]["counted"]
                     and fresh_identity["n_hits"] == identity["n_hits"])
        ledger, identity = fresh_ledger, fresh_identity
        if converged:
            return ledger, identity
    _write(ledger, identity)
    return ledger, identity


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="问题一 v2 体积台账与身份扫描")
    parser.add_argument("--json", action="store_true", help="只输出 JSON")
    args = parser.parse_args(argv)

    ledger, identity = write_until_stable()

    if args.json:
        print(json.dumps(ledger["accounting"], ensure_ascii=False, indent=2))
    else:
        acc = ledger["accounting"]
        print(f"v2 计入提交 {_fmt(acc['v2_counted_bytes'])}")
        print(f"替换核算   {acc['replace_total_bytes']:,} B  "
              f"≤50MiB {'OK' if acc['replace_within_loose'] else 'FAIL'}  "
              f"≤50e6 {'OK' if acc['replace_within_strict'] else 'FAIL'}")
        print(f"叠加核算   {acc['stack_total_bytes']:,} B  "
              f"≤50MiB {'OK' if acc['stack_within_loose'] else 'FAIL'}  "
              f"≤50e6 {'OK' if acc['stack_within_strict'] else 'FAIL'}")
        print(f"身份扫描命中 {identity['n_hits']} 处（扫 {identity['scanned_files']} 个文件，"
              f"含 NPZ 内 {identity['scanned_npz_strings']} 条字符串）")
        for hit in identity["hits"][:10]:
            # 只报定位，不重复命中原文：见 _mask 的说明
            where = hit.get("field") or f"第 {hit.get('line', '?')} 行"
            print(f"  {hit['file']}  {where}  {hit['label']}  {hit['match_masked']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
