# -*- coding: utf-8 -*-
"""
_build_q1_body.py —— 问题一 Q2 体例正文的一键构建入口

把三段串起来：

    _gen_paper_tables_q1.py     → _tables_q1/*.md    （表格从 data/q1_v2/ 现取）
    _inject_paper_tables.py     → 问题一论文正文_展开.md
    _make_q1_body_docx.py       → 问题一论文正文.docx  （体例对齐 Q2_论文正文.docx）

与 `_build_q1_paper.py`（华为杯模板体例、带封面摘要目录的整篇论文）并列，
两者共用同一批表格片段，区别只在体例与标题层级。

本脚本负责三件容易出错的事：

1. **顺序**：表格 → 注入 → docx。错序是静默的（占位符会原样进 Word），
   故展开稿里一旦残留占位符就直接退出。
2. **旁观者不可动**：动手前后各取一次指纹，不相等即报错退出。旁观者包括
   三问稿、旧问题一体例稿，以及用户提供的权威稿 `Q2_论文正文.docx`。
3. **体例对齐是可核验的**：出稿后把新 docx 的样式度量与 `Q2_论文正文.docx`
   逐项比对（正文中西文字体/字号/行距、标题字号、页面与页边距、题注字号），
   不一致就报错——「格式参考 Q2」这句话必须落在可复算的数字上。

用法：
    python _build_q1_body.py                # 全链
    python _build_q1_body.py --stage=tables # 只重出表格
    python _build_q1_body.py --stage=md     # 出到展开稿为止
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

SRC = os.path.join(ROOT, "问题一论文正文.md")
EXPANDED = os.path.join(ROOT, "问题一论文正文_展开.md")
DOCX = os.path.join(ROOT, "问题一论文正文.docx")
TABLES = os.path.join(ROOT, "_tables_q1")
#: 体例基准。只读，任何情况下都不得写它。
REF_DOCX = os.path.join(ROOT, "Q2_论文正文.docx")

PROTECTED = (
    os.path.join(ROOT, "论文.docx"),
    os.path.join(ROOT, "问题一论文.docx"),
    os.path.join(ROOT, "问题一论文.md"),
    os.path.join(ROOT, "问题一论文_展开.md"),
    REF_DOCX,
    os.path.join(ROOT, "问题二论文.docx"),
)
PROTECTED_DIRS = (os.path.join(ROOT, "data", "q1_v2"),
                  os.path.join(ROOT, "data", "q1_delivery"))

sys.path.insert(0, ROOT)
from _build_q1_paper import _dir_digest, _file_sha, _identity_scan  # noqa: E402


def _snapshot() -> dict:
    snap = {p: (_file_sha(p) if os.path.isfile(p) else "<absent>")
            for p in PROTECTED}
    snap.update({d: _dir_digest(d) for d in PROTECTED_DIRS})
    return snap


def _run(cmd: list, label: str) -> None:
    print(f"\n{'=' * 72}\n[{label}] {' '.join(cmd)}\n{'=' * 72}")
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"[失败] {label} 退出码 {r.returncode}")


# ---------------------------------------------------------------- 体例比对

def _metrics(path: str) -> dict:
    """抽出可比的体例度量。取的是样式与节属性，不是某个样本段的偶发直接格式。"""
    from docx import Document
    d = Document(path)
    st = d.styles
    sec = d.sections[0]
    n = st["Normal"]
    out = {
        "正文西文字体": n.font.name,
        "正文中文字体": n.element.rPr.rFonts.get(
            "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}eastAsia"),
        "正文字号": n.font.size.pt if n.font.size else None,
        "正文行距": n.paragraph_format.line_spacing,
        "正文段后(pt)": (n.paragraph_format.space_after.pt
                     if n.paragraph_format.space_after else 0),
        "首行缩进(cm)": round(n.paragraph_format.first_line_indent.cm, 3)
                    if n.paragraph_format.first_line_indent else 0,
        "Title字号": st["Title"].font.size.pt if st["Title"].font.size else None,
        "H1字号": st["Heading 1"].font.size.pt if st["Heading 1"].font.size else None,
        "H2字号": st["Heading 2"].font.size.pt if st["Heading 2"].font.size else None,
        "页宽(cm)": round(sec.page_width.cm, 2),
        "页高(cm)": round(sec.page_height.cm, 2),
        "左边距(cm)": round(sec.left_margin.cm, 2),
        "右边距(cm)": round(sec.right_margin.cm, 2),
        "上边距(cm)": round(sec.top_margin.cm, 2),
        "下边距(cm)": round(sec.bottom_margin.cm, 2),
    }
    cap = next((p for p in d.paragraphs
                if p.text.strip().startswith(("图 ", "表 ")) and p.runs
                and p.runs[0].font.size), None)
    out["题注字号"] = cap.runs[0].font.size.pt if cap is not None else None
    return out


def _format_parity(out_path: str) -> bool:
    """与 Q2_论文正文.docx 逐项比对体例。返回是否全部一致。"""
    if not os.path.isfile(REF_DOCX):
        print(f"  ⚠ 体例基准缺失（{os.path.basename(REF_DOCX)}），跳过比对")
        return True
    ref, new = _metrics(REF_DOCX), _metrics(out_path)
    bad = []
    print(f"\n[体例比对] {'项':<16}{'Q2_论文正文':<18}{'本稿':<18}结论")
    for k in ref:
        same = ref[k] == new[k]
        if not same:
            bad.append(k)
        print(f"           {k:<16}{str(ref[k]):<18}{str(new[k]):<18}"
              f"{'一致' if same else '不一致'}")
    if bad:
        print(f"\n[失败] 下列体例项与基准不一致：{bad}")
        return False
    print("           全部一致")
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="问题一 Q2 体例正文构建链")
    ap.add_argument("--stage", default="all",
                    choices=("tables", "md", "docx", "all"))
    args = ap.parse_args(argv)

    assert os.path.basename(DOCX) == "问题一论文正文.docx", DOCX
    assert os.path.abspath(DOCX) != os.path.abspath(REF_DOCX), "输出名等于体例基准"

    before = _snapshot()
    print("[基线] 旁观者指纹已记录：")
    for k, v in before.items():
        print(f"       {os.path.relpath(k, ROOT):<34}{v[:16]}…")

    _run([PY, os.path.join(ROOT, "_gen_paper_tables_q1.py")], "1/3 生成表格片段")
    if args.stage == "tables":
        return _finish(before)

    _run([PY, os.path.join(ROOT, "_inject_paper_tables.py"),
          "--src", SRC, "--out", EXPANDED, "--tables", TABLES], "2/3 注入表格")

    with open(EXPANDED, encoding="utf-8") as f:
        expanded = f.read()
    leftover = re.findall(r"\{[A-Z_]+:[A-Za-z0-9_]+\}|\{STRUCT_ABLATION\}", expanded)
    if leftover:
        raise SystemExit(f"[失败] 展开稿仍有未替换占位符：{sorted(set(leftover))}")
    if args.stage == "md":
        return _finish(before)

    _run([PY, os.path.join(ROOT, "_make_q1_body_docx.py"),
          "--md", EXPANDED, "--out", DOCX], "3/3 生成 docx")

    ok = _finish(before, scanned=(SRC, EXPANDED, DOCX))
    if ok != 0:
        return ok
    return 0 if _format_parity(DOCX) else 1


def _finish(before: dict, scanned: tuple = ()) -> int:
    after = _snapshot()
    bad = [k for k in before if before[k] != after[k]]
    print(f"\n[旁观者核验] 既有交付物是否被改动")
    for k in before:
        mark = "✗ 已改动" if before[k] != after[k] else "✓ 未改动"
        print(f"           {os.path.relpath(k, ROOT):<34}{mark}")
    if bad:
        print("\n[失败] 下列既有交付物被改动，本次构建不可接受：")
        for k in bad:
            print(f"       {k}")
        return 1

    if scanned:
        print(f"\n[身份扫描] 新建的根级产物（package_check.py 只扫 data/ 与 code/，须自扫）")
        hits = _identity_scan(list(scanned))
        if not hits:
            print("           零命中")
        else:
            for fn, desc, s in hits:
                print(f"           ⚠ {fn}: {desc}  （样例：{s}）")
            return 1

    print("\n[完成] 问题一 Q2 体例正文构建链走通")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
