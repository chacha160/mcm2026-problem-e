# -*- coding: utf-8 -*-
"""
_make_report_docx.py —— 把一份普通 markdown 报告转成 Word（复用论文链的排版件）

与 `_make_paper_docx.py` 的区别：**不加封面、不加摘要页、不加目录**。
论文链的 `build()` 把这三样硬编码在其中，只适用于竞赛论文；本文档是工作方案，
直接套会把「摘 要」「目录」页强塞进来。故此处只复用它的解析与排版原语。

用法：
    python _make_report_docx.py <in.md> <out.docx>
"""
from __future__ import annotations

import os
import sys

import _make_paper_docx as P
from docx import Document
from docx.oxml.ns import qn


def build(md_path: str, out_path: str) -> int:
    with open(md_path, "r", encoding="utf-8") as f:
        md = f.read()

    # parse_md 只在遇到 "# 摘" 时才切分；本文档没有摘要，故原样返回全部行。
    lines, abstract = P.parse_md(md)
    if abstract:
        print("警告：本脚本不处理摘要页，摘要块将被忽略。")

    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = P.SONG
    st.font.size = P.PT_XIAOSI
    st.element.rPr.rFonts.set(qn("w:eastAsia"), P.SONG)

    sec = doc.sections[0]
    P.setup_page(sec)
    P.clear_footer(sec)
    P.add_page_number_footer(sec, restart_at=1)

    P.emit(doc, lines, base_dir=os.path.dirname(os.path.abspath(md_path)))
    doc.save(out_path)
    print(f"已生成 {out_path}")
    return 0


def main(argv=None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    if len(a) != 2:
        print(__doc__)
        return 2
    if not os.path.isfile(a[0]):
        print(f"找不到输入：{a[0]}")
        return 2
    return build(a[0], a[1])


if __name__ == "__main__":
    raise SystemExit(main())
