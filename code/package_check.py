# -*- coding: utf-8 -*-
"""
package_check.py —— 提交体积核算与赛题红线扫描

赛题「四、结果与提交说明·（二）附件提交要求」：全部附件合计不超过 50 MB，且不得含身份信息。
「二、数据说明」附件1的注 R1：不替换、不增加、不删除、不修改任何样本或标签。
「二、数据说明」附件2/3/4的注 R2：训练/验证/专项测试使用同一特征版本与同一输入接口。
「五、补充说明·1.数据与工具使用规范」R3：只能使用 CMU-MOSEI 系列数据。

本脚本不产生新模型，只做**核账**：把将要随论文提交的产物逐个量体积、
扫描敏感内容，并输出一份可引用的核算表。它刻意不统计原始数据
（附件1-4 由赛题方提供，不计入我方提交体积）与代码缓存。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import config  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

LIMIT_BYTES = 50 * 1024 * 1024       # 按 50×1024² 解释，与 q1_delivery 的口径一致

# 计入提交体积的目录（相对项目根）。原始附件、__pycache__、日志不计入。
INCLUDED_DIRS = (
    ("q1_delivery", os.path.join("data", "q1_delivery")),
    ("q2", os.path.join("data", "q2")),
    ("q3", os.path.join("data", "q3")),
    # audit 步的产物：缺失结构审计。它是问题二「缺失从何而来」的唯一凭据，
    # 论文 5.2.1 与附录引用它，故必须计入体积——漏计会让「≤50 MB」的结论失去依据。
    ("q2q3_audit", os.path.join("data", "q2q3")),
    # 打包核验自身的产物（体积台账、核验回执）
    ("package_check", os.path.join("data", "package_check")),
)
# 代码与文档也要随论文提交，但多数赛制把代码单独列；这里单列出来供选择
CODE_DIRS = (("code", "code"),)

# 身份信息扫描：路径泄漏、本机用户名、以及原视频文件被误打包
IDENTITY_PATTERNS = (
    (re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s\"',;)]+", re.I), "本机 Windows 用户目录路径"),
    (re.compile(r"/home/[A-Za-z0-9_.-]+"), "本机 Linux 用户目录路径"),
    # 注：曾有一条专门匹配「第三方方案作者机器目录」的规则（把对方用户名硬编码在正则里）。
    # 已删除——上面第一条通用规则 `[A-Za-z]:[\\/]+Users[\\/]+<名字>` 本来就覆盖它，
    # 专用规则既冗余、又把第三方用户名留在了本仓库源码里，与它自己要防的事情相抵触。
)

# 脱敏替换规则：保留**可迁移的等价信息**（版本号、文件名校验值），只抹掉用户名。
# 直接删掉整条路径会让「环境如何复现」失去依据，所以是替换而非删除。
_REDACT_RULES = (
    (re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s\"',;)]+[\\/]+AppData[\\/]+Local", re.I),
     "%LOCALAPPDATA%"),
    (re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s\"',;)]+", re.I), "~"),
    (re.compile(r"/home/[A-Za-z0-9_.-]+"), "~"),
)


def _redact(text: str) -> Tuple[str, int]:
    n = 0
    for pat, rep in _REDACT_RULES:
        text, k = pat.subn(rep, text)
        n += k
    return text, n


def sanitize_delivery(roots: Sequence[str]) -> Dict:
    """对交付物做就地脱敏，并把「改了什么、改了几处」写进同一个文件留痕。

    只处理文本类文件；不改动任何样本、标签或数值特征。脱敏结果不隐藏：
    每个被改写的文件都会多出一个 `sanitization` 记录块。
    """
    changed: List[Dict] = []
    for rel in roots:
        p = os.path.join(config.PROJECT_ROOT, rel)
        if not os.path.isdir(p):
            continue
        for f in walk_files(p):
            ext = os.path.splitext(f)[1].lower()
            if ext not in TEXT_EXT:
                continue
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    txt = fh.read()
            except Exception:
                continue
            new, n = _redact(txt)
            if n == 0:
                continue
            if ext == ".json":
                try:
                    obj = json.loads(new)
                    if isinstance(obj, dict):
                        obj["sanitization"] = {
                            "applied": True,
                            "replacements": n,
                            "rules": ["%LOCALAPPDATA%", "~"],
                            "note": ("本机绝对路径已替换为可迁移的等价写法；"
                                     "版本与校验值信息保留，仅移除用户名。"
                                     "数值特征、样本与标签未被改动。"),
                        }
                        new = json.dumps(obj, ensure_ascii=False, indent=2)
                except Exception:
                    pass
            with open(f, "w", encoding="utf-8", newline="") as fh:
                fh.write(new)
            changed.append({"file": os.path.relpath(f, config.PROJECT_ROOT),
                            "replacements": n})
    return {"files_changed": len(changed), "details": changed}

# 扫描的文本类扩展名
TEXT_EXT = {".py", ".md", ".json", ".csv", ".txt", ".log", ".yaml", ".yml", ".cfg", ".ini"}
BINARY_MEDIA_EXT = {".mp4", ".avi", ".mov", ".mkv", ".wav", ".mp3", ".flac", ".m4a"}


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n:.2f} GB"


def walk_files(root: str) -> List[str]:
    out: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ("__pycache__", ".git", ".claude")]
        for fn in filenames:
            if fn.startswith("~$") or fn.endswith(".pyc"):
                continue
            out.append(os.path.join(dirpath, fn))
    return out


def measure(groups: Sequence[Tuple[str, str]]) -> Tuple[List[Dict], int]:
    rows: List[Dict] = []
    total = 0
    for name, rel in groups:
        p = os.path.join(config.PROJECT_ROOT, rel)
        if not os.path.isdir(p):
            rows.append({"group": name, "path": rel, "exists": 0,
                         "n_files": 0, "bytes": 0})
            continue
        files = walk_files(p)
        b = sum(os.path.getsize(f) for f in files)
        total += b
        rows.append({"group": name, "path": rel, "exists": 1,
                     "n_files": len(files), "bytes": b})
    return rows, total


def scan_identity(roots: Sequence[str]) -> List[Dict]:
    """扫描文本类文件里的路径泄漏，以及是否误把原始视频/音频打进提交物。"""
    hits: List[Dict] = []
    for rel in roots:
        p = os.path.join(config.PROJECT_ROOT, rel)
        if not os.path.isdir(p):
            continue
        for f in walk_files(p):
            ext = os.path.splitext(f)[1].lower()
            if ext in BINARY_MEDIA_EXT:
                hits.append({"file": os.path.relpath(f, config.PROJECT_ROOT),
                             "issue": "二进制媒体文件（原始音视频不得进入提交物）"})
                continue
            if ext not in TEXT_EXT:
                continue
            try:
                with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                    txt = fh.read(2_000_000)
            except Exception:
                continue
            for pat, desc in IDENTITY_PATTERNS:
                m = pat.search(txt)
                if m:
                    hits.append({"file": os.path.relpath(f, config.PROJECT_ROOT),
                                 "issue": desc, "sample": m.group(0)[:80]})
    return hits


def check_interface_consistency() -> Dict:
    """R2 核验：训练/验证/附件3/附件4 是否真的走同一输入接口。

    判据是可执行的事实，不是声明：四处的字段集合必须一致地落在
    `text_bert + audio + vision` 上，且维度相同。
    """
    out: Dict[str, object] = {"expected_fields": ["text_bert", "audio", "vision"]}
    try:
        import q2q3_common as Q
        a2 = Q.load_attachment2()["train"]
        a3 = Q.load_attachment3()
        a4 = Q.load_attachment4()
        out["attachment2_train"] = {
            "text_bert": list(a2.text_bert.shape[1:]),
            "audio": list(a2.audio.shape[1:]),
            "vision": list(a2.vision.shape[1:]),
            "has_768_text": a2.text_feat is not None,
        }
        out["attachment3"] = {
            "text_bert": list(a3[0].text_bert.shape),
            "audio": list(a3[0].audio.shape),
            "vision": list(a3[0].vision.shape),
            "has_768_text": False,
        }
        out["attachment4"] = {
            "text_bert": list(a4[0].text_bert.shape),
            "audio": list(a4[0].audio.shape),
            "vision": list(a4[0].vision.shape),
            "has_768_text": True,
        }
        shapes_ok = (
            list(a2.text_bert.shape[1:]) == list(a3[0].text_bert.shape) == list(a4[0].text_bert.shape)
            and list(a2.audio.shape[1:]) == list(a3[0].audio.shape) == list(a4[0].audio.shape)
            and list(a2.vision.shape[1:]) == list(a3[0].vision.shape) == list(a4[0].vision.shape)
        )
        out["interface_consistent"] = bool(shapes_ok)
        out["note"] = (
            "问题二的模型输入统一为 text_bert(3,50) + audio(50,74) + vision(50,35)，"
            "训练（附件2）、推理（附件3）、问题三（附件4）三处形状完全一致；"
            "附件3 不提供 768 维 text，故 768 维文本分支**不进入主模型**，"
            "仅作为附件2/4 上的消融对照。"
        )
    except Exception as e:  # noqa: BLE001
        out["interface_consistent"] = None
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="提交体积核算与红线扫描")
    ap.add_argument("--sanitize", action="store_true",
                    help="对交付物就地脱敏（移除本机用户名等身份信息）后再核算")
    args = ap.parse_args(argv)

    print("=" * 72)
    print("提交体积核算与红线扫描")
    print("=" * 72)

    if args.sanitize:
        print("\n[脱敏] 就地改写交付物中的本机绝对路径")
        # 脱敏范围必须与下面的身份扫描范围**一致**（交付物目录 + 代码目录）。
        # 曾只覆盖交付物目录，于是 code/logs/*.log 里的本机路径被扫描检出、
        # 却永远轮不到脱敏——「已加 --sanitize 但仍报身份信息=需处理」的假阳性。
        san = sanitize_delivery([r[1] for r in INCLUDED_DIRS] + [r[1] for r in CODE_DIRS])
        print(f"  改写 {san['files_changed']} 个文件")
        for d in san["details"]:
            print(f"    {d['file']}  （{d['replacements']} 处）")
        if not san["files_changed"]:
            print("  无需改写")

    rows, total = measure(INCLUDED_DIRS)
    code_rows, code_total = measure(CODE_DIRS)

    print("\n[交付物体积]")
    print(f"{'组':<14}{'文件数':>8}{'体积':>14}  路径")
    for r in rows:
        print(f"{r['group']:<14}{r['n_files']:>8}{human(r['bytes']):>14}  {r['path']}")
    print(f"{'合计':<14}{sum(r['n_files'] for r in rows):>8}{human(total):>14}")
    print(f"\n[代码体积]  {sum(r['n_files'] for r in code_rows)} 文件  {human(code_total)}")
    print(f"[预算]      上限 {human(LIMIT_BYTES)}；"
          f"交付物占 {total/LIMIT_BYTES:.1%}；剩余 {human(LIMIT_BYTES-total)}")

    print("\n[身份信息扫描]")
    hits = scan_identity([r[1] for r in INCLUDED_DIRS] + [r[1] for r in CODE_DIRS])
    if not hits:
        print("  未发现身份信息或原始媒体文件")
    else:
        for h in hits[:40]:
            print(f"  ⚠ {h['file']}: {h['issue']}"
                  + (f"  （样例：{h['sample']}）" if h.get("sample") else ""))
        print(f"  合计 {len(hits)} 处")

    print("\n[R2 输入接口一致性核验]")
    iface = check_interface_consistency()
    print(json.dumps(iface, ensure_ascii=False, indent=2))

    out_dir = os.path.join(config.PROJECT_ROOT, "data", "package_check")
    os.makedirs(out_dir, exist_ok=True)
    report = {
        "limit_bytes": LIMIT_BYTES,
        "deliverable_bytes": total,
        "deliverable_ratio": round(total / LIMIT_BYTES, 6),
        "remaining_bytes": LIMIT_BYTES - total,
        "groups": rows,
        "code_bytes": code_total,
        "identity_hits": hits,
        "interface_check": iface,
        "verdict": {
            "size_ok": total <= LIMIT_BYTES,
            "identity_ok": len(hits) == 0,
            "interface_ok": bool(iface.get("interface_consistent")),
        },
    }
    with open(os.path.join(out_dir, "package_check.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "size_budget.csv"), "w", encoding="utf-8-sig",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group", "path", "exists", "n_files", "bytes"])
        w.writeheader()
        w.writerows(rows)

    v = report["verdict"]
    print(f"\n[结论] 体积={'通过' if v['size_ok'] else '超限'}  "
          f"身份信息={'通过' if v['identity_ok'] else '需处理'}  "
          f"接口一致性={'通过' if v['interface_ok'] else '不一致'}")
    print(f"核算表：{os.path.join(out_dir, 'package_check.json')}")
    return 0 if all(v.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
