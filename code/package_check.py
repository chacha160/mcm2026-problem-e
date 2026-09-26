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
# 「50 MB」的另一种读法。两者的松紧关系容易记反：52.43 MB > 50.00 MB，
# 所以 1024² 那一读法**更宽松**。执行宽松读法，但把严格读法的占用一并印出来。
LIMIT_STRICT_BYTES = 50 * 10 ** 6

# 计入提交体积的目录（相对项目根）。原始附件、__pycache__、日志不计入。
#
# 问题一换向（v1 → v2）：问题一已重建为词级多粒度架构，交付物是 `data/q1_v2/`；
# 旧的 `data/q1_delivery/`（50 等分槽架构）**留在盘上但不计提交**。不换向的话，
# 下面「≤50 MB」的结论扫的是已被弃用的目录，真正的交付物从未被核过。
# 旧口径的体积在预算段一并印出（`OLD_Q1_REL`），让换向带来的差异可见、可复算。
INCLUDED_DIRS = (
    ("q1_v2", os.path.join("data", "q1_v2")),
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

#: 换向前的旧问题一交付目录。只用于在预算段印出旧口径做对照，**不计入提交**。
OLD_Q1_REL = os.path.join("data", "q1_delivery")

#: 仓库**根目录**下、**不属于交付物**的文档。它们不计体积（不是产物），
#: 但必须被身份扫描覆盖，理由有两条：
#:
#:   1. 打包时最容易一起压进去的就是根目录文件——`data/` 与 `code/` 反而有子目录
#:      这道心理边界；
#:   2. 这些文件里记的多是**工作方案、外部资料索引**一类的内容，恰恰最容易出现
#:      机器路径与他人用户名。
#:
#: 曾有一份根级方案对比文档，既含本机用户名、又含外部参考资料的作者机器路径，
#: 而它落在扫描范围之外——「身份扫描零命中」这句话对它根本不成立。列入此处后，
#: 扫描覆盖它，但**仍不计入提交体积**（它不是产物，只是不该泄漏）。
NON_DELIVERABLE_ROOT_DOCS = (
    "方案对比报告_A0Pilot参考方案_vs_本方案.md",
    "Q1建模流程与逻辑框架总结.md",
)

#: 计入体积时要跳过的子目录（按组名）。v2 的这两处是**可重建的中间态**：
#: `.cache/` 是分模态原始观测的逐样本缓存，`logs/` 是执行日志——两者都能由
#: `run_q1v2_all.py` 重跑再生，计进提交体积会虚增约 7 MB 且掩盖真实占用。
#: 这与 `reports/q1v2_size_ledger.json` 的口径一致（各处数字须能互相解释）。
EXCLUDED_SUBDIRS = {
    "q1_v2": (".cache", "logs"),
}

# 身份信息扫描：路径泄漏、本机用户名、以及原视频文件被误打包
IDENTITY_PATTERNS = (
    (re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s\"',;)]+", re.I), "本机 Windows 用户目录路径"),
    (re.compile(r"/home/[A-Za-z0-9_.-]+"), "本机 Linux 用户目录路径"),
    # 注：曾有一条把某个外部资料作者的机器目录名硬编码进来的专用规则。已删除——
    # 上面第一条通用规则 `[A-Za-z]:[\\/]+Users[\\/]+<名字>` 本来就覆盖它，
    # 专用规则既冗余，又把那个名字留在了本仓库源码里，与它自己要防的事情相抵触。
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
    """按 1024 进制折行。**单位如实写 MiB/KiB/GiB**——本函数据 1024 连除，
    印成「MB」会让同一份输出里出现 1024 进制的数配 1000 进制的标签。"""
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n:.2f} GiB"


def walk_files(root: str, exclude: Sequence[str] = ()) -> List[str]:
    """遍历 `root` 下的文件。`exclude` 是目录名黑名单，按名字匹配任一层。

    排除的是**可重建中间态与工具目录**，不是样本或标签：被排除的目录里
    不含任何特征、掩码或标注，重跑即可再生。
    """
    skip = {"__pycache__", ".git", ".claude"} | set(exclude)
    out: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip]
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
        exclude = EXCLUDED_SUBDIRS.get(name, ())
        files = walk_files(p, exclude)
        b = sum(os.path.getsize(f) for f in files)
        total += b
        rows.append({"group": name, "path": rel, "exists": 1,
                     "n_files": len(files), "bytes": b,
                     "excluded": list(exclude)})
    return rows, total


#: 核算表自己所在的那一组。它的体积里含 `package_check.json` 与 `size_budget.csv`
#: 两个文件，而这两个文件写的正是这一组的体积：先量后写，写出来的数字一变、字数跟着
#: 变，量出来的值也就跟着变，于是报告可能与实物差一个字节。
#: 实测踩到过：改一份交付文档后这一组由 3205 B 变 3204 B，报告仍记 3205。
_SELF_GROUP = "package_check"


def write_accounting(out_dir: str, report: Dict, rows: List[Dict]) -> Dict:
    """写两份核算表，写到报告与实物自洽为止。

    每一轮按 `rows` 重算合计与派生量、写文件，再重量本组目录；量出来的值与报告里记的
    一致即收敛。数字位数变化会改变文件长度，故不能只写一次。几轮仍不自洽说明存在
    循环依赖之外的错，报错退出，不放出一份与实物不符的核算表。
    """
    for _ in range(8):
        total = sum(r["bytes"] for r in rows)
        report.update({
            "deliverable_bytes": total,
            "deliverable_ratio": round(total / LIMIT_BYTES, 6),
            "deliverable_ratio_vs_strict": round(total / LIMIT_STRICT_BYTES, 6),
            "remaining_bytes": LIMIT_BYTES - total,
            "remaining_bytes_strict": LIMIT_STRICT_BYTES - total,
            "groups": rows,
            # 严格读法也一并判：两种读法都过，「不会超」这句话才不依赖口径选择。
            "verdict": {
                "size_ok": total <= LIMIT_STRICT_BYTES,
                "identity_ok": not report["identity_hits"],
                "interface_ok": bool(report["interface_check"].get("interface_consistent")),
            },
        })
        with open(os.path.join(out_dir, "package_check.json"), "w",
                  encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        with open(os.path.join(out_dir, "size_budget.csv"), "w", encoding="utf-8-sig",
                  newline="") as f:
            w = csv.DictWriter(f, fieldnames=["group", "path", "exists", "n_files",
                                              "bytes", "excluded"],
                               extrasaction="ignore")
            w.writeheader()
            # `excluded` 要进表：体积核算里「哪些子目录被排除」是可复算这件事的一部分——
            # 只给合计而不给排除名单，读者无法判断这个合计覆盖了什么。
            for r in rows:
                w.writerow({**r, "excluded": ",".join(r.get("excluded") or [])})

        row = next(r for r in rows if r["group"] == _SELF_GROUP)
        fresh, _ = measure([(_SELF_GROUP, row["path"])])
        if (fresh[0]["n_files"], fresh[0]["bytes"]) == (row["n_files"], row["bytes"]):
            return report
        row.update(fresh[0])
    raise SystemExit(f"[失败] {_SELF_GROUP} 一组写到不动点仍不自洽：报告记 "
                     f"{row['bytes']} B，实测 {fresh[0]['bytes']} B")


def scan_identity(roots: Sequence[str]) -> List[Dict]:
    """扫描文本类文件里的路径泄漏，以及是否误把原始视频/音频打进提交物。

    扫两类位置：`roots` 里的交付/代码目录，以及根目录下**非交付物**的文档
    （见 `NON_DELIVERABLE_ROOT_DOCS`）——后者不在 `roots` 里，但打包时最容易被
    顺手压进去。
    """
    hits: List[Dict] = []
    root_docs = [os.path.join(config.PROJECT_ROOT, n) for n in NON_DELIVERABLE_ROOT_DOCS]
    for f in root_docs:
        if not os.path.isfile(f):
            continue
        with open(f, "r", encoding="utf-8", errors="ignore") as fh:
            txt = fh.read(2_000_000)
        for pat, desc in IDENTITY_PATTERNS:
            m = pat.search(txt)
            if m:
                hits.append({"file": os.path.relpath(f, config.PROJECT_ROOT),
                             "issue": desc + "（根级非交付文档，勿随包提交）",
                             "sample": m.group(0)[:80]})
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
        note = ""
        if r.get("excluded"):
            note = f"  （已排除 {', '.join(r['excluded'])}/：可重建中间态）"
        print(f"{r['group']:<14}{r['n_files']:>8}{human(r['bytes']):>14}  {r['path']}{note}")
    print(f"{'合计':<14}{sum(r['n_files'] for r in rows):>8}{human(total):>14}")

    # 换向前口径：只印不计。让「白名单换向」这一改动的量级可见，旧结论仍可复算。
    old_rows, old_total = measure((("q1_delivery(旧)", OLD_Q1_REL),))
    if old_rows[0]["exists"]:
        print(f"\n[换向前口径，仅供参考，不计提交]  {old_rows[0]['n_files']} 文件  "
              f"{human(old_total)}  {OLD_Q1_REL}")
    print(f"\n[代码体积]  {sum(r['n_files'] for r in code_rows)} 文件  {human(code_total)}")
    print(f"[预算]      上限 宽松 50×1024² = {human(LIMIT_BYTES)} = "
          f"{LIMIT_BYTES/1e6:.2f} MB(decimal)（执行口径）")
    print(f"            上限 严格 50×10⁶  = {human(LIMIT_STRICT_BYTES)} = "
          f"{LIMIT_STRICT_BYTES/1e6:.2f} MB(decimal)")
    print(f"            交付物 {human(total)} = {total/1e6:.2f} MB(decimal)；"
          f"占宽松读法 {total/LIMIT_BYTES:.1%}，占严格读法 {total/LIMIT_STRICT_BYTES:.1%}")
    print(f"            剩余 宽松 {human(LIMIT_BYTES-total)}，"
          f"严格 {human(LIMIT_STRICT_BYTES-total)}")

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
        "limit_bytes_strict": LIMIT_STRICT_BYTES,
        "limit_note": "「50 MB」两种读法：宽松 50×1024²=52,428,800 B（执行口径），"
                      "严格 50×10⁶=50,000,000 B（评审可能采用）。前者更宽松。",
        "deliverable_bytes": total,
        "deliverable_ratio": round(total / LIMIT_BYTES, 6),
        "deliverable_ratio_vs_strict": round(total / LIMIT_STRICT_BYTES, 6),
        "remaining_bytes": LIMIT_BYTES - total,
        "remaining_bytes_strict": LIMIT_STRICT_BYTES - total,
        "groups": rows,
        "code_bytes": code_total,
        "identity_hits": hits,
        # 扫描范围只记**数量**，不记根级非交付文档的文件名——其中一个文件名本身
        # 就写着外部参考资料的代号，把它写进这份交付的 JSON，等于用「防止泄漏」
        # 的功能把那个代号带进了产物。命中项另论：命中时必须给得出文件名，
        # 否则报告无法据以修复（见 hits 的 file 字段）。
        "identity_scan_scope": {
            "delivery_dirs": [r[1] for r in INCLUDED_DIRS],
            "code_dirs": [r[1] for r in CODE_DIRS],
            "n_non_deliverable_root_docs_scanned": len(NON_DELIVERABLE_ROOT_DOCS),
            "note": "根级非交付文档也纳入扫描（不计体积），否则「零命中」这句话"
                    "对它们不成立；文件名不写入本报告。",
        },
        "interface_check": iface,
    }
    # 合计与结论由 write_accounting 一轮轮重算——本组体积里含正在写的这两个文件。
    report = write_accounting(out_dir, report, rows)

    v = report["verdict"]
    print(f"\n[结论] 体积={'通过' if v['size_ok'] else '超限'}  "
          f"身份信息={'通过' if v['identity_ok'] else '需处理'}  "
          f"接口一致性={'通过' if v['interface_ok'] else '不一致'}")
    print(f"核算表：{os.path.join(out_dir, 'package_check.json')}")
    return 0 if all(v.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
