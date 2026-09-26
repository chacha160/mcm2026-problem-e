# -*- coding: utf-8 -*-
"""
_make_q1_body_docx.py —— 把 `问题一论文正文.md` 编译成与 `Q2_论文正文.docx`
同体例的 Word 正文

体例取自 `Q2_论文正文.docx`（用户 2026-09-26 提供的权威稿）。该文件的排版特征
是逐项量出来的，不是猜的，清单如下：

  · 页面 A4，左右边距 2.30 cm、上下 2.20 cm，无页眉页脚，单节
  · 正文 Normal：西文 Times New Roman、中文宋体、11 pt、行距 1.2 倍、
    段后 3 pt、首行缩进 2 字符（pt(22)）
  · 标题走 Word 内建样式：Title 黑体 16 pt 加粗；Heading 1 黑体 13 pt 加粗；
    Heading 2 黑体 11.5 pt 加粗；段前 10 pt、段后 6 pt、行距 1.1、与下段同页
  · 编号公式 = **1 行 2 列的无框线表格**：左格 8561 twip 放 OMML（10.5 pt），
    右格 737 twip 右对齐放编号「(1-1)」（10 pt）。Q2 正文里没有 oMathPara，
    11 个编号公式全部是这种表格
  · 数据表 = 三线表：表头行上下各一条实线（上 10/下 5），末行下实线 10，
    其余边线一律 nil；表头行加粗并重复（tblHeader）；单元格 10 pt，
    列数 ≥ 7 时降为 9 pt；首行单元格居左、其余按内容长短决定居中或居左
  · 图表题注：Normal 样式、10 pt、居中、无首行缩进；表题在表前且与表同页，
    图题在图后
  · 插图宽度 15.9 cm（版心 16.4 cm）

与 `_make_paper_docx.py`（华为杯模板体例：小四宋体、四号黑体居中标题、封面、
摘要页、目录、页码）是**两套体例**，故另起一个生成器，只复用它的 LaTeX→OMML
转换、行内富文本与插图表逻辑——那部分与体例无关，且已经被本仓库的中文公式
写法（\\mathcal、\\mathrm、^_、\\left...\\right）验证过。

用法：
    python _make_q1_body_docx.py                     # 默认进出
    python _make_q1_body_docx.py --md X.md --out Y.docx

约束：本脚本只写 out 指定的文件。源稿 `问题一论文正文.md` 与既有 `问题一论文.*`、
`Q2_论文正文.docx` 都不在它的写入范围内。
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import List, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

# 与华为杯体例共用、且与体例无关的部分
from _make_paper_docx import (  # noqa: E402
    HEI, MONO, SONG, add_rich, latex_to_omml, set_font,
)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

# ---------------------------------------------------------------- 体例常量

PT_BODY = Pt(11)        # 正文
PT_CAP = Pt(10)         # 图表题注
PT_TBL = Pt(10)         # 表格正文
PT_TBL_WIDE = Pt(9)     # 宽表（列数 ≥ 7）
PT_CODE = Pt(9)         # 代码块
PT_EQ = 21              # 公式字号（半磅）：21 = 10.5 pt
PT_EQ_NO = 20           # 公式编号字号：20 = 10 pt

TEXT_WIDTH_TWIPS = 9298  # 版心宽 = 21 cm − 2×2.3 cm，与 Q2 的 tblGrid 合计一致
EQ_W_LEFT, EQ_W_RIGHT = 8561, 737  # Q2 编号公式表的列宽

BODY_LINE = 1.2
BODY_AFTER = Pt(3)
FIRST_INDENT = Pt(22)   # 2 字符 @ 11 pt
TBL_CELL_SPACE = 60     # twip，= 3 pt，与 Q2 单元格一致

# 题注用加粗标出：`**表 1-1 …**` / `**图 1-2 …**` / `**表 A-1 …**`。
# 不用「行首是 表/图 + 编号」来判：正文里有「表 1-8 由 NPZ 重新计算得到」这类句子，
# 那样判会把整句正文排成居中的题注。加粗是排版惯例允许保留的标记，此处兼作语法。
CAP_RE = re.compile(r"^\*\*\s*(表|图)\s*([A-Z]?\d+-\d+)\s*(.*?)\s*\*\*\s*$")
EQ_TAG_RE = re.compile(r"\\tag\s*\{\s*([^}]*?)\s*\}\s*$")
PLACEHOLDER_RE = re.compile(r"\{[A-Z_]+:[A-Za-z0-9_]+\}|\{STRUCT_ABLATION\}")


# ---------------------------------------------------------------- 样式

def setup_styles(doc) -> None:
    """把 doc 的 Normal / Title / Heading 1-2 调成 Q2 正文的样式。"""
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    st.font.size = PT_BODY
    st.font.bold = False
    rpr = st.element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.insert(0, rf)
    rf.set(qn("w:ascii"), "Times New Roman")
    rf.set(qn("w:hAnsi"), "Times New Roman")
    rf.set(qn("w:eastAsia"), SONG)
    pf = st.paragraph_format
    pf.line_spacing = BODY_LINE
    pf.space_before = Pt(0)
    pf.space_after = BODY_AFTER
    pf.first_line_indent = FIRST_INDENT

    specs = {
        "Title":     Pt(16),
        "Heading 1": Pt(13),
        "Heading 2": Pt(11.5),
        "Heading 3": Pt(11),
    }
    for name, size in specs.items():
        bold = True
        s = doc.styles[name]
        s.font.name = "Times New Roman"
        s.font.size = size
        s.font.bold = bold
        s.font.color.rgb = RGBColor(0, 0, 0)
        rpr = s.element.get_or_add_rPr()
        rf = rpr.find(qn("w:rFonts"))
        if rf is None:
            rf = OxmlElement("w:rFonts")
            rpr.insert(0, rf)
        rf.set(qn("w:ascii"), "Times New Roman")
        rf.set(qn("w:hAnsi"), "Times New Roman")
        rf.set(qn("w:eastAsia"), HEI)
        # 内建标题样式默认是蓝色 Calibri Light：底色由主题色给出，只写 w:val 不够，
        # 必须把 themeColor 一并摘掉，否则 Word 仍按主题色渲染
        el = rpr.find(qn("w:color"))
        if el is None:
            el = OxmlElement("w:color")
            rpr.append(el)
        for attr in ("w:themeColor", "w:themeTint", "w:themeShade"):
            if el.get(qn(attr)) is not None:
                del el.attrib[qn(attr)]
        el.set(qn("w:val"), "000000")
        pf = s.paragraph_format
        pf.space_before = Pt(10)
        pf.space_after = Pt(6)
        pf.line_spacing = 1.1
        pf.keep_with_next = True
        pf.first_line_indent = Pt(0)


def setup_page(section) -> None:
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.left_margin = Cm(2.3)
    section.right_margin = Cm(2.3)
    section.top_margin = Cm(2.2)
    section.bottom_margin = Cm(2.2)
    section.header_distance = Cm(1.27)
    section.footer_distance = Cm(1.10)
    # Q2 正文无页眉页脚，本节同样不放任何内容
    for part in (section.header, section.footer):
        part.is_linked_to_previous = False
        for p in list(part.paragraphs):
            p._p.getparent().remove(p._p)
        part.add_paragraph()


# ---------------------------------------------------------------- 段落

def _styled(doc, align=WD_ALIGN_PARAGRAPH.LEFT, before=0, after=BODY_AFTER,
            line=BODY_LINE, indent_first=FIRST_INDENT, style=None):
    p = doc.add_paragraph(style=style)
    pf = p.paragraph_format
    p.alignment = align
    pf.space_before = before
    pf.space_after = after
    pf.line_spacing = line
    pf.first_line_indent = indent_first
    return p


def add_body(doc, text: str) -> None:
    p = _styled(doc)
    add_rich(p, text, size=PT_BODY, base_font=SONG)


def add_caption(doc, text: str, keep_next: bool) -> None:
    """图表题注：10 pt、居中、无首行缩进。表题与表同页，故 keep_with_next。"""
    p = _styled(doc, align=WD_ALIGN_PARAGRAPH.CENTER, before=Pt(6),
                after=Pt(4), line=1.0, indent_first=Pt(0))
    p.paragraph_format.keep_with_next = keep_next
    add_rich(p, text, size=PT_CAP, base_font=SONG)


def add_note(doc, text: str) -> None:
    """表注：10 pt、左对齐、无首行缩进，跟在表后。"""
    p = _styled(doc, align=WD_ALIGN_PARAGRAPH.LEFT, before=Pt(2),
                after=Pt(6), line=1.0, indent_first=Pt(0))
    add_rich(p, text, size=PT_CAP, base_font=SONG)


def add_code(doc, lines: List[str]) -> None:
    for ln in lines:
        p = _styled(doc, before=Pt(0), after=Pt(0), line=1.0,
                    indent_first=Pt(0))
        p.paragraph_format.left_indent = Cm(0.5)
        r = p.add_run(ln if ln.strip() else " ")
        set_font(r, MONO, PT_CODE)
    _styled(doc, before=Pt(0), after=Pt(4), indent_first=Pt(0))


def add_picture(doc, path: str, base_dir: str, width_cm: float = 15.9) -> None:
    full = path if os.path.isabs(path) else os.path.join(base_dir, path)
    if not os.path.isfile(full):
        raise FileNotFoundError(f"插图不存在：{full}（.md 中写的是 {path}）")
    p = _styled(doc, align=WD_ALIGN_PARAGRAPH.CENTER, before=Pt(6),
                after=Pt(2), line=1.0, indent_first=Pt(0))
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(full, width=Cm(width_cm))


# ---------------------------------------------------------------- 表格公共件

def _tbl_pr(tbl, center: bool = True) -> None:
    tblPr = tbl._tbl.tblPr
    for tag, attrs in (
        ("w:tblW", {"w:type": "auto", "w:w": "0"}),
        ("w:jc", {"w:val": "center"}),
        ("w:tblLayout", {"w:type": "fixed"}),
        ("w:tblLook", {"w:firstColumn": "1", "w:firstRow": "1",
                       "w:lastColumn": "0", "w:lastRow": "0",
                       "w:noHBand": "0", "w:noVBand": "1", "w:val": "04A0"}),
    ):
        if tag == "w:jc" and not center:
            continue
        el = OxmlElement(tag)
        for k, v in attrs.items():
            el.set(qn(k), v)
        tblPr.append(el)


def _set_grid(tbl, widths: List[int]) -> None:
    grid = tbl._tbl.find(qn("w:tblGrid"))
    for gc in list(grid):
        grid.remove(gc)
    for w in widths:
        gc = OxmlElement("w:gridCol")
        gc.set(qn("w:w"), str(int(w)))
        grid.append(gc)


def _row_props(row, header: bool = False) -> None:
    trPr = row._tr.get_or_add_trPr()
    trPr.append(OxmlElement("w:cantSplit"))
    if header:
        trPr.append(OxmlElement("w:tblHeader"))


def _cell_borders(cell, spec: dict) -> None:
    """spec: {'top': ('nil'|'single', sz), ...}；未列出的边一律 nil。"""
    tcPr = cell._tc.get_or_add_tcPr()
    old = tcPr.find(qn("w:tcBorders"))
    if old is not None:
        tcPr.remove(old)
    b = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        val, sz = spec.get(edge, ("nil", 6))
        el = OxmlElement("w:" + edge)
        el.set(qn("w:val"), val)
        el.set(qn("w:sz"), str(sz))
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "000000")
        b.append(el)
    tcPr.append(b)


def _cell(cell, width_tw: int, text: str, size: Pt, align, borders: dict,
          bold: bool = False) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    # tcPr 的子元素在 OOXML 里是有序的：tcW → tcBorders → vAlign。
    # 顺序错了 Word 会认为文档损坏，故这里逐段插入而不是一股脑 append。
    tcW = tcPr.find(qn("w:tcW"))
    if tcW is None:            # python-docx 建格时已带一个 tcW，不能重复插
        tcW = OxmlElement("w:tcW")
        tcPr.insert(0, tcW)
    tcW.set(qn("w:type"), "dxa")
    tcW.set(qn("w:w"), str(int(width_tw)))
    _cell_borders(cell, borders)
    vAlign = OxmlElement("w:vAlign")
    vAlign.set(qn("w:val"), "center")
    tcPr.append(vAlign)
    p = cell.paragraphs[0]
    p.alignment = align
    pf = p.paragraph_format
    pf.space_before = Pt(TBL_CELL_SPACE / 20.0)
    pf.space_after = Pt(TBL_CELL_SPACE / 20.0)
    pf.line_spacing = 1.0
    pf.first_line_indent = Pt(0)
    add_rich(p, text, size=size)
    if bold:
        for r in p.runs:
            r.font.bold = True


def _disp_w(s: str) -> int:
    return sum(2 if ord(c) > 0x2E80 else 1 for c in s)


def _col_widths(rows: List[List[str]], ncol: int) -> List[int]:
    """按各列最长内容定列宽，合计锁在版心宽度；列内容长的给宽。"""
    weights = []
    for j in range(ncol):
        w = max((_disp_w(r[j]) for r in rows if j < len(r)), default=4)
        weights.append(max(4.0, min(w, 40.0)))
    total = sum(weights)
    widths = [max(600, int(TEXT_WIDTH_TWIPS * w / total)) for w in weights]
    # 归一化回版心宽（取整后会有残差，落到最宽的一列上）
    widths[widths.index(max(widths))] += TEXT_WIDTH_TWIPS - sum(widths)
    return widths


def add_data_table(doc, rows: List[List[str]]) -> None:
    ncol = max(len(r) for r in rows)
    size = PT_TBL_WIDE if ncol >= 7 else PT_TBL
    widths = _col_widths(rows, ncol)

    tbl = doc.add_table(rows=len(rows), cols=ncol)
    _tbl_pr(tbl)
    _set_grid(tbl, widths)

    # 居中/居左：短列居中，首列与长文本列居左（Q2 的数据表就是这个混合口径）
    aligns = []
    for j in range(ncol):
        longest = max((_disp_w(r[j]) for r in rows if j < len(r)), default=0)
        left = (j == 0) or longest > 12
        aligns.append(WD_ALIGN_PARAGRAPH.LEFT if left else WD_ALIGN_PARAGRAPH.CENTER)

    for i, row in enumerate(rows):
        _row_props(tbl.rows[i], header=(i == 0))
        for j in range(ncol):
            if i == 0:
                spec = {"top": ("single", 10), "bottom": ("single", 5)}
            elif i == len(rows) - 1:
                spec = {"bottom": ("single", 10)}
            else:
                spec = {}
            _cell(tbl.cell(i, j), widths[j], row[j] if j < len(row) else "",
                  size, aligns[j], spec, bold=(i == 0))
    _styled(doc, before=Pt(0), after=Pt(4), indent_first=Pt(0))


def add_equation(doc, latex: str, tag: str) -> None:
    """编号公式：1×2 无框线表格，左格公式居中、右格编号右对齐。"""
    body = latex_to_omml(latex)
    # Q2 的公式字号是 10.5 pt（sz=21），而 _mr 不写字号、会继承正文 11 pt
    body = body.replace('w:eastAsia="Cambria Math"/></w:rPr>',
                        'w:eastAsia="Cambria Math"/><w:sz w:val="%d"/></w:rPr>' % PT_EQ)

    tbl = doc.add_table(rows=1, cols=2)
    _tbl_pr(tbl, center=False)
    _set_grid(tbl, [EQ_W_LEFT, EQ_W_RIGHT])
    _row_props(tbl.rows[0])

    left = tbl.cell(0, 0)
    _cell(left, EQ_W_LEFT, "", PT_EQ, WD_ALIGN_PARAGRAPH.CENTER, {})
    left.paragraphs[0]._p.append(_parse_xml(body))

    right = tbl.cell(0, 1)
    _cell(right, EQ_W_RIGHT, "", PT_EQ_NO, WD_ALIGN_PARAGRAPH.RIGHT, {})
    r = right.paragraphs[0].add_run("(%s)" % tag)
    set_font(r, SONG, Pt(PT_EQ_NO / 2.0))
    _styled(doc, before=Pt(0), after=Pt(2), indent_first=Pt(0))


def _parse_xml(xml: str):
    from docx.oxml import parse_xml
    return parse_xml(xml)


# ---------------------------------------------------------------- 正文解析

def _table_block(lines: List[str], start: int) -> int:
    j = start
    while j < len(lines) and lines[j].strip().startswith("|"):
        j += 1
    return j


def emit(doc, lines: List[str], base_dir: str, title_at: int = -1) -> int:
    i, n = 0, len(lines)
    h1_count = 0
    while i < n:
        s = lines[i].strip()

        if s.startswith("```"):
            i += 1
            buf = []
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            add_code(doc, buf)
            continue

        m = re.match(r"^!\[(.*?)\]\((.+?)\)\s*$", s)
        if m:
            add_picture(doc, m.group(2).strip(), base_dir)
            i += 1
            continue

        if s.startswith("|"):
            j = _table_block(lines, i)
            rows = _rows(lines[i:j])
            if rows:
                add_data_table(doc, rows)
            i = j
            continue

        if s.startswith("$$"):
            buf, rest = [], s[2:]
            if rest.endswith("$$") and len(rest) > 2:
                buf.append(rest[:-2])
                i += 1
            else:
                if rest.strip():
                    buf.append(rest)
                i += 1
                while i < n and "$$" not in lines[i]:
                    buf.append(lines[i])
                    i += 1
                if i < n:
                    buf.append(lines[i].split("$$")[0])
                    i += 1
            src = "\n".join(x for x in buf if x.strip()).strip()
            tag = None
            tm = EQ_TAG_RE.search(src)
            if tm:
                tag = tm.group(1)
                src = src[:tm.start()].strip()
            if tag:
                add_equation(doc, src, tag)
            else:
                p = _styled(doc, align=WD_ALIGN_PARAGRAPH.CENTER, before=Pt(6),
                            after=Pt(6), line=1.0, indent_first=Pt(0))
                p._p.append(_parse_xml(latex_to_omml(src)))
            continue

        m = re.match(r"^(#{1,4})\s+(.*)$", s)
        if m:
            lvl, txt = len(m.group(1)), m.group(2).strip()
            if lvl == 1 and i == title_at:
                p = _styled(doc, style="Title", align=WD_ALIGN_PARAGRAPH.CENTER,
                            before=Pt(0), after=Pt(10), line=1.1,
                            indent_first=Pt(0))
                r = p.add_run(txt)
                set_font(r, HEI, Pt(16), bold=True)
            else:
                if lvl == 1:
                    h1_count += 1
                name = {1: "Heading 1", 2: "Heading 2"}.get(lvl, "Heading 3")
                p = _styled(doc, style=name, before=Pt(10), after=Pt(6),
                            line=1.1, indent_first=Pt(0))
                r = p.add_run(txt)
                set_font(r, HEI,
                         {1: Pt(13), 2: Pt(11.5)}.get(lvl, Pt(11)), bold=True)
            i += 1
            continue

        m = CAP_RE.match(s)
        if m:
            add_caption(doc, "%s %s %s" % (m.group(1), m.group(2), m.group(3)),
                        keep_next=(m.group(1) == "表"))
            i += 1
            continue

        if s.startswith("表注："):
            add_note(doc, s)
            i += 1
            continue

        if not s:
            i += 1
            continue

        add_body(doc, s)
        i += 1
    return h1_count


def _rows(block: List[str]) -> List[List[str]]:
    rows = []
    for k, ln in enumerate(block):
        if k == 1 and re.match(r"^\|[\s:|-]+\|$", ln.strip()):
            continue
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", ln.strip().strip("|"))]
        rows.append(cells)
    return rows


def build(md_path: str, out_path: str) -> int:
    with open(md_path, encoding="utf-8") as f:
        lines = f.read().split("\n")

    title_at = next((k for k, ln in enumerate(lines) if ln.startswith("# ")), -1)

    doc = Document()
    setup_styles(doc)
    setup_page(doc.sections[0])
    md_dir = os.path.dirname(os.path.abspath(md_path))
    h1 = emit(doc, lines, base_dir=md_dir, title_at=title_at)
    doc.save(out_path)

    n_tbl = len(doc.tables)
    n_eq = sum(1 for t in doc.tables
               if len(t.rows) == 1 and len(t.columns) == 2
               and t.rows[0].cells[1].text.strip().startswith("("))
    print(f"已生成：{out_path}")
    print(f"  段落 {len(doc.paragraphs)} / 表格 {n_tbl}（其中编号公式 {n_eq}）"
          f" / 插图 {len(doc.inline_shapes)} / 一级标题 {h1}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="问题一正文 Markdown → Q2 体例 docx")
    ap.add_argument("--md", default=os.path.join(here, "问题一论文正文_展开.md"))
    ap.add_argument("--out", default=os.path.join(here, "问题一论文正文.docx"))
    args = ap.parse_args(argv)

    if not os.path.isfile(args.md):
        raise SystemExit(f"[失败] 源稿不存在：{args.md}\n"
                         "       先跑 _inject_paper_tables.py 展开表格占位符。")
    with open(args.md, encoding="utf-8") as f:
        md = f.read()
    left = PLACEHOLDER_RE.findall(md)
    if left:
        raise SystemExit(f"[失败] 源稿仍有未替换占位符：{sorted(set(left))}\n"
                         "       出稿前必须先注入表格，否则占位符会原样进 Word。")
    if os.path.abspath(args.out) in (
            os.path.abspath(os.path.join(here, "论文.docx")),
            os.path.abspath(os.path.join(here, "问题一论文.docx")),
            os.path.abspath(os.path.join(here, "问题二论文.docx"))):
        raise SystemExit(f"[失败] 输出名指向既有交付物，拒绝覆盖：{args.out}")
    return build(args.md, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
