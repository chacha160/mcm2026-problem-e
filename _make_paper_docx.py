# -*- coding: utf-8 -*-
"""
_make_paper_docx.py —— 《论文格式规范》的排版件与 Markdown→Word 转换器

本模块现在是**共用排版件**：字体常量（`HEI` / `SONG` / `MONO`）、标题与表格样式、
公式转换 `latex_to_omml`、行内混排 `add_rich`、`set_font` 都从这里取。两个入口在用它：

* `_make_q1_body_docx.py` —— 问题一正文（`问题一论文正文.md` → `.docx`），
  当前定稿走的就是这条链，**要出正文请走它**（那条链自带写保护与图件替换）。
* `_make_report_docx.py` —— 任意报告 markdown → Word。

输入必须是**注入后**的稿子（展开稿），直接喂源稿会把 `{TABLE:...}` 占位符原样
写进 Word，故默认值指向展开稿而非源稿。

规范来源：`_格式规范.txt`（从官方 docx 提取）与 `_模板.txt`（从官方 .doc 提取）。

落实的硬性要求：
  · 论文题目 三号黑体 居中
  · 一级标题 四号黑体 居中
  · 其他汉字 小四号宋体，单倍行距
  · 摘要页起编页码，页脚中部，阿拉伯数字从 1 连续
  · 不能有页眉
  · 封面为独立一页且不计页码
  · 公式用 OMML（Word 原生公式对象），不是图片、不是纯文本

用法：
    python _make_paper_docx.py                    # 展开稿 → Word
    python _make_paper_docx.py --md X.md --out Y.docx

> **注意：本脚本的默认输入输出是「通用稿」这一对，不带写保护**。要出当前定稿
> `问题一论文正文.docx`，走 `_build_q1_body.py`（那条链会在动手前后核对既有
> 交付物、并替换图件），别直接用本脚本的 `--out` 指过去。
> 封面上的华为标志图（本脚本只写文字占位「【此处插入华为标志图片】」）与 Word 里
> 生成的真目录（本脚本只写占位域，需在 Word 中按 F9 重建），两者本脚本都产不出。
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import List, Optional, Sequence, Tuple

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn, nsmap
from docx.shared import Cm, Inches, Pt, RGBColor

# ---------------------------------------------------------------- 字号 / 字体

PT_SANHAO = Pt(16)      # 三号
PT_SIHAO = Pt(14)       # 四号
PT_XIAOSI = Pt(12)      # 小四
PT_WUHAO = Pt(10.5)     # 五号（表格内）
PT_XIAOWU = Pt(9)       # 小五（代码块）

SONG = "宋体"
HEI = "黑体"
KAI = "楷体"
MONO = "Consolas"
MATH = "Cambria Math"

M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass


# ---------------------------------------------------------------- 基础工具

def set_font(run, name: str, size: Pt, bold: bool = False, italic: bool = False) -> None:
    """设置中西文字体（Word 需要单独指定 eastAsia，否则中文会回落到默认字体）。"""
    run.font.name = name
    run.font.size = size
    run.bold = bold
    run.italic = italic
    rpr = run._element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.insert(0, rf)
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rf.set(qn(attr), name)


def styled_par(doc, align=WD_ALIGN_PARAGRAPH.LEFT, space_before=0, space_after=0,
               line=1.0, indent_first=None, left_indent=None):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    p.alignment = align
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    pf.line_spacing = line
    if indent_first is not None:
        pf.first_line_indent = indent_first
    if left_indent is not None:
        pf.left_indent = left_indent
    return p


# ---------------------------------------------------------------- LaTeX -> OMML

_DS = str.maketrans({
    "A": "\U0001D538", "B": "\U0001D539", "C": "ℂ", "D": "\U0001D53B",
    "E": "\U0001D53C", "F": "\U0001D53D", "G": "\U0001D53E", "H": "ℍ",
    "I": "\U0001D540", "J": "\U0001D541", "K": "\U0001D542", "L": "\U0001D543",
    "M": "\U0001D544", "N": "ℕ", "O": "\U0001D546", "P": "ℙ",
    "Q": "ℚ", "R": "ℝ", "S": "\U0001D54A", "T": "\U0001D54B",
    "U": "\U0001D54C", "V": "\U0001D54D", "W": "\U0001D54E", "X": "\U0001D54F",
    "Y": "\U0001D550", "Z": "ℤ",
    "1": "\U0001D7D9",
})

_SYMS = {
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ",
    r"\delta": "δ", r"\Delta": "Δ", r"\eta": "η",
    r"\lambda": "λ", r"\mu": "μ", r"\pi": "π",
    r"\rho": "ρ", r"\sigma": "σ", r"\tau": "τ",
    r"\theta": "θ", r"\phi": "φ", r"\omega": "ω",
    r"\Omega": "Ω", r"\Sigma": "Σ", r"\Lambda": "Λ",
    r"\in": "∈", r"\notin": "∉", r"\wedge": "∧",
    r"\vee": "∨", r"\neg": "¬", r"\top": "ᵀ",
    r"\approx": "≈", r"\times": "×", r"\leq": "≤",
    r"\geq": "≥", r"\neq": "≠", r"\cdot": "·",
    r"\dots": "…", r"\ldots": "…", r"\lVert": "‖",
    r"\rVert": "‖", r"\|": "‖", r"\to": "→",
    r"\rightarrow": "→", r"\Rightarrow": "⇒",
    r"\mathbb{R}": "ℝ", r"\mathcal{L}": "\U0001D4DB",
    r"\mathcal{W}": "\U0001D4B2", r"\mathcal{K}": "\U0001D4A6",
    r"\quad": " ", r"\qquad": "  ", r"\,": " ", r"\;": " ",
    r"\ ": " ",
}

_LARGE = {r"\sum": "∑", r"\prod": "∏", r"\int": "∫"}
_UPRIGHT = {r"\log", r"\exp", r"\max", r"\min", r"\arg", r"\max", r"\ln",
            r"\sin", r"\cos", r"\tan", r"\softmax", r"\mathrm", r"\text",
            r"\operatorname", r"\sgn", r"\mathbb"}


def _mr(text: str, sty: Optional[str] = None, space: bool = True) -> str:
    """一个 OMML 数学 run。<m:sty> 可取 'b'(粗) / 'p'(正体)。"""
    pr = ""
    if sty:
        pr = f'<m:rPr><m:sty m:val="{sty}"/></m:rPr>'
    sp = ' xml:space="preserve"' if space else ""
    return (f'<m:r>{pr}<w:rPr><w:rFonts w:ascii="{MATH}" w:hAnsi="{MATH}" '
            f'w:eastAsia="{MATH}"/></w:rPr><m:t{sp}>{_esc(text)}</m:t></m:r>')


def _esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


class _P:
    """极简 LaTeX 子集解析器：只覆盖本文用到的命令。"""

    def __init__(self, src: str):
        self.s = src
        self.i = 0

    # -- 基本读取 -------------------------------------------------
    def peek(self) -> str:
        return self.s[self.i] if self.i < len(self.s) else ""

    def eof(self) -> bool:
        return self.i >= len(self.s)

    def read_group_raw(self) -> str:
        """读取一个 {...} 组（或单个 token）的原始文本。"""
        while not self.eof() and self.peek() == " ":
            self.i += 1
        if self.peek() == "{":
            depth, self.i, out = 1, self.i + 1, []
            while not self.eof() and depth:
                c = self.s[self.i]
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        self.i += 1
                        break
                out.append(c)
                self.i += 1
            return "".join(out)
        if self.peek() == "\\":
            j = self.i + 1
            while j < len(self.s) and (self.s[j].isalpha()):
                j += 1
            tok = self.s[self.i:j] if j > self.i + 1 else self.s[self.i:self.i + 2]
            self.i = j
            return tok
        c = self.peek()
        self.i += 1
        return c

    # -- 解析 -----------------------------------------------------
    def parse(self, stops: str = "") -> str:
        out: List[str] = []
        while not self.eof():
            c = self.peek()
            if stops and c in stops:
                break
            if c in "^_":
                self.i += 1
                base = out.pop() if out else _mr("")
                sub = sup = None
                if c == "_":
                    sub = self.parse_atom()
                else:
                    sup = self.parse_atom()
                # 允许 a_b^c 形式
                # 注意：`peek()` 在串尾返回 ""，而 Python 里 "" in "^_" 为 True，
                # 所以必须先判非空，否则循环永不终止。
                while self.peek() and self.peek() in "^_":
                    c2 = self.peek()
                    self.i += 1
                    if c2 == "_":
                        sub = self.parse_atom()
                    else:
                        sup = self.parse_atom()
                if sub is not None and sup is not None:
                    out.append(f"<m:sSubSup><m:e>{base}</m:e><m:sub>{sub}</m:sub>"
                               f"<m:sup>{sup}</m:sup></m:sSubSup>")
                elif sub is not None:
                    out.append(f"<m:sSub><m:e>{base}</m:e><m:sub>{sub}</m:sub></m:sSub>")
                else:
                    out.append(f"<m:sSup><m:e>{base}</m:e><m:sup>{sup}</m:sup></m:sSup>")
                continue
            out.append(self.parse_atom())
        return "".join(out)

    def parse_atom(self) -> str:
        while not self.eof() and self.peek() == " ":
            self.i += 1
        if self.eof():
            return ""
        c = self.peek()

        if c == "{":
            raw = self.read_group_raw()
            return _P(raw).parse()
        if c == "\\":
            return self._command()
        self.i += 1
        return _mr(c, space=False)

    def _command(self) -> str:
        j = self.i + 1
        while j < len(self.s) and self.s[j].isalpha():
            j += 1
        if j == self.i + 1:                       # 单字符命令，如 \{ \, \|
            tok = self.s[self.i:self.i + 2]
            self.i += 2
            if tok in _SYMS:
                return _mr(_SYMS[tok])
            return _mr(tok[1], space=False)
        tok = self.s[self.i:j]
        self.i = j

        if tok in (r"\left", r"\right"):
            if tok == r"\right":
                return ""
            pk = self.peek()
            beg = pk if (pk and pk in "([{") else ""
            if beg:
                self.i += 1
            end = ""
            inner_start = self.i
            depth = 0
            while not self.eof():
                if self.peek() == "{":
                    depth += 1
                elif self.peek() == "}":
                    depth -= 1
                elif depth == 0 and self.s.startswith(r"\right", self.i):
                    break
                self.i += 1
            inner = _P(self.s[inner_start:self.i]).parse()
            self.i += len(r"\right")
            pk = self.peek()
            end = pk if (pk and pk in ")]}") else ""
            if end:
                self.i += 1
            beg_str = {"": "(", "(": "(", "[": "[", "{": "{"}.get(beg, beg)
            end_str = {"": ")", ")": ")", "]": "]", "}": "}"}.get(end, end)
            return (f'<m:d><m:dPr><m:begChr m:val="{_esc(beg_str)}"/>'
                    f'<m:endChr m:val="{_esc(end_str)}"/></m:dPr>'
                    f'<m:e>{inner}</m:e></m:d>')

        if tok == r"\frac":
            a = self.parse_atom()
            b = self.parse_atom()
            return f"<m:f><m:num>{a}</m:num><m:den>{b}</m:den></m:f>"

        if tok == r"\sqrt":
            a = self.parse_atom()
            return (f'<m:rad><m:radPr><m:degHide m:val="1"/></m:radPr>'
                    f'<m:deg/><m:e>{a}</m:e></m:rad>')

        if tok == r"\hat":
            a = self.parse_atom()
            return (f'<m:acc><m:accPr><m:chr m:val="̂"/></m:accPr>'
                    f'<m:e>{a}</m:e></m:acc>')

        if tok in _LARGE:
            chr_ = _LARGE[tok]
            sub = sup = ""
            if self.peek() == "_":
                self.i += 1
                sub = self.parse_atom()
            if self.peek() == "^":
                self.i += 1
                sup = self.parse_atom()
            body = self.parse(stops="")
            return (f'<m:nary><m:naryPr><m:chr m:val="{chr_}"/>'
                    f'<m:limLoc m:val="undOvr"/><m:subHide m:val="{0 if sub else 1}"/>'
                    f'<m:supHide m:val="{0 if sup else 1}"/></m:naryPr>'
                    f'<m:sub>{sub}</m:sub><m:sup>{sup}</m:sup>'
                    f'<m:e>{body}</m:e></m:nary>')

        if tok == r"\bigvee":
            return _mr("⋁")

        if tok in (r"\mathbb", r"\mathbf", r"\mathrm", r"\text", r"\mathcal",
                   r"\operatorname"):
            raw = self.read_group_raw()
            if tok == r"\mathbb":
                if raw.strip() == "1":
                    return _mr("\U0001D7D9")
                return _mr(raw.translate(_DS))
            if tok == r"\mathcal":
                return _mr(raw, sty="p")
            if tok == r"\mathbf":
                return _mr(raw, sty="b")
            return _mr(raw, sty="p")

        if tok in _SYMS:
            return _mr(_SYMS[tok])
        if tok in _UPRIGHT:
            return _mr(tok[1:], sty="p")

        # 未知命令：原样输出命令名，避免静默丢失
        return _mr(tok[1:], sty="p")


def latex_to_omml(src: str) -> str:
    """把公式源码转成 `<m:oMath>` 元素。调用方负责放进段落。"""
    body = _P(src.strip()).parse()
    return f'<m:oMath xmlns:m="{M_NS}" xmlns:w="{W_NS}">{body}</m:oMath>'


# 编号公式的写法是 `... \tag{5.3}`。`\tag` 不在 OMML 的记号集里，转换器会把它
# 整个丢掉——结果是正文里「式 (5.3)」的引用指向一个看不见的编号。故在此先把它
# 摘出来，公式本体居中、编号右对齐，用 Word 的右制表位排版。
_TAG_RE = re.compile(r"\\tag\s*\{\s*([^}]*?)\s*\}\s*$")


def add_math_par(doc, src: str, align=WD_ALIGN_PARAGRAPH.CENTER, before=6, after=6):
    src = src.strip()
    m = _TAG_RE.search(src)
    tag = m.group(1) if m else None
    if m:
        src = src[:m.start()].strip()
    p = styled_par(doc, align=align, space_before=before, space_after=after)
    p._p.append(_parse_xml(latex_to_omml(src)))
    if tag:
        # 制表位放在版心右缘（A4 去左右边距约 6.5 英寸），编号贴右
        _add_right_tab(p, Inches(6.3))
        r = p.add_run("\t（" + tag + "）")
        r.font.size = Pt(10.5)
    return p


def _add_right_tab(par, pos):
    """给段落加一个右对齐制表位，用于把公式编号推到行尾。"""
    from docx.oxml.ns import qn as _qn
    pPr = par._p.get_or_add_pPr()
    tabs = pPr.find(_qn("w:tabs"))
    if tabs is None:
        tabs = pPr.makeelement(_qn("w:tabs"), {})
        pPr.append(tabs)
    tab = tabs.makeelement(_qn("w:tab"), {_qn("w:val"): "right", _qn("w:pos"): str(int(pos))})
    tabs.append(tab)


def _parse_xml(xml: str):
    from docx.oxml import parse_xml
    return parse_xml(xml)


def add_inline_math(par, src: str) -> None:
    par._p.append(_parse_xml(latex_to_omml(src)))


# ---------------------------------------------------------------- 富文本行内

_TOKEN_RE = re.compile(
    r"(\*\*.+?\*\*"          # 粗体
    r"|`[^`]+`"              # 行内代码
    r"|\$[^$]+\$"            # 行内公式
    r"|\[[^\]\s]+\]\([^)]+\)"  # 链接
    r")"
)


def add_rich(par, text: str, size=PT_XIAOSI, base_font=SONG,
             bold: bool = False) -> None:
    """把一行 markdown 行内标记写进段落：**粗**、`代码`、$公式$、[文字](链接)。

    粗体**可以嵌套**其它标记（本文里 `**用 \\`attention_mask\\` 之和确定 $L$**`
    这种写法很常见），所以命中 `**...**` 时必须递归处理其内容，否则内层的
    反引号与 $ 会原样出现在 Word 里。
    """
    text = text.replace("\\|", "|")
    for tok in _TOKEN_RE.split(text):
        if not tok:
            continue
        if tok.startswith("**") and tok.endswith("**") and len(tok) > 4:
            add_rich(par, tok[2:-2], size=size, base_font=base_font, bold=True)
        elif tok.startswith("`") and tok.endswith("`") and len(tok) > 2:
            r = par.add_run(tok[1:-1])
            set_font(r, MONO, Pt(size.pt - 1.5), bold=bold)
        elif tok.startswith("$") and tok.endswith("$") and len(tok) > 2:
            add_inline_math(par, tok[1:-1])
        elif tok.startswith("[") and "](" in tok:
            r = par.add_run(tok[1:tok.index("]")])
            set_font(r, base_font, size, bold=bold)
        else:
            r = par.add_run(tok)
            set_font(r, base_font, size, bold=bold)


def strip_md_inline(s: str) -> str:
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"`([^`]+)`", r"\1", s)
    s = re.sub(r"\$([^$]+)\$", r"\1", s)
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
    return s.replace("\\|", "|")


# ---------------------------------------------------------------- 三线表

def _set_cell_border(cell, **kwargs) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    borders = tcPr.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tcPr.append(borders)
    for edge in ("top", "bottom", "left", "right"):
        if edge in kwargs:
            spec = kwargs[edge]
            el = borders.find(qn("w:" + edge))
            if el is None:
                el = OxmlElement("w:" + edge)
                borders.append(el)
            el.set(qn("w:val"), spec.get("val", "single"))
            el.set(qn("w:sz"), str(spec.get("sz", 8)))
            el.set(qn("w:space"), "0")
            el.set(qn("w:color"), spec.get("color", "000000"))
        elif edge in kwargs.get("_clear", ()):  # pragma: no cover
            pass


def add_three_line_table(doc, rows: List[List[str]]) -> None:
    """rows[0] 为表头。三线表：顶线粗、表头下细线、底线粗，无竖线。"""
    ncol = max(len(r) for r in rows)
    t = doc.add_table(rows=len(rows), cols=ncol)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = True
    for i, row in enumerate(rows):
        for j in range(ncol):
            cell = t.cell(i, j)
            txt = row[j] if j < len(row) else ""
            cell.text = ""
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if (i == 0 or j > 0) \
                else WD_ALIGN_PARAGRAPH.LEFT
            pf = p.paragraph_format
            pf.space_before = Pt(1.5)
            pf.space_after = Pt(1.5)
            pf.line_spacing = 1.0
            add_rich(p, txt, size=PT_WUHAO)
            # 清掉所有框线
            _set_cell_border(cell, top={"val": "none"}, bottom={"val": "none"},
                             left={"val": "none"}, right={"val": "none"})
            if i == 0:
                _set_cell_border(cell, top={"sz": 12})
                _set_cell_border(cell, bottom={"sz": 6})
            if i == len(rows) - 1:
                _set_cell_border(cell, bottom={"sz": 12})
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


# ---------------------------------------------------------------- 页码

def add_page_number_footer(section, restart_at: Optional[int] = None) -> None:
    footer = section.footer
    footer.is_linked_to_previous = False
    p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    p.text = ""
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for child in list(p._p):
        if child.tag != qn("w:pPr"):
            p._p.remove(child)
    r = p.add_run()
    set_font(r, SONG, PT_WUHAO)
    fld = _parse_xml(
        f'<w:fldSimple xmlns:w="{W_NS}" w:instr=" PAGE  \\* MERGEFORMAT ">'
        f'<w:r><w:rPr><w:rFonts w:ascii="{SONG}" w:hAnsi="{SONG}" '
        f'w:eastAsia="{SONG}"/><w:sz w:val="21"/></w:rPr>'
        f'<w:t>1</w:t></w:r></w:fldSimple>')
    p._p.append(fld)
    if restart_at is not None:
        sectPr = section._sectPr
        old = sectPr.find(qn("w:pgNumType"))
        if old is not None:
            sectPr.remove(old)
        sectPr.append(_parse_xml(
            f'<w:pgNumType xmlns:w="{W_NS}" w:start="{restart_at}"/>'))


def clear_footer(section) -> None:
    section.footer.is_linked_to_previous = False
    f = section.footer
    for p in list(f.paragraphs):
        p._p.getparent().remove(p._p)
    f.add_paragraph()


# ---------------------------------------------------------------- 文档骨架

def setup_page(section) -> None:
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.54)
    section.bottom_margin = Cm(2.54)
    section.left_margin = Cm(3.17)
    section.right_margin = Cm(3.17)
    section.header_distance = Cm(1.5)
    section.footer_distance = Cm(1.75)
    # 无页眉
    section.header.is_linked_to_previous = False
    for p in list(section.header.paragraphs):
        p._p.getparent().remove(p._p)
    section.header.add_paragraph()


def add_page_break(doc) -> None:
    p = doc.add_paragraph()
    p.add_run().add_break(WD_BREAK.PAGE)


def build_cover(doc) -> None:
    for _ in range(2):
        styled_par(doc)
    p = styled_par(doc, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=6)
    r = p.add_run("【此处插入华为标志图片】")
    set_font(r, SONG, PT_XIAOSI)
    for _ in range(2):
        styled_par(doc)

    for txt, size in (("中国研究生创新实践系列大赛", PT_SIHAO),
                      ("“华为杯”第二十三届中国研究生", PT_SANHAO),
                      ("数学建模竞赛", PT_SANHAO)):
        p = styled_par(doc, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=6)
        r = p.add_run(txt)
        set_font(r, HEI, size)

    for _ in range(4):
        styled_par(doc)

    fields = [("学    校", ""), ("参赛队号", ""),
              ("队员姓名", "1.            2.            3.            ")]
    for label, val in fields:
        p = styled_par(doc, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=14)
        r = p.add_run(f"{label}    {val}")
        set_font(r, SONG, PT_SIHAO)
        r.font.underline = True

    styled_par(doc)
    p = styled_par(doc, align=WD_ALIGN_PARAGRAPH.CENTER)
    r = p.add_run("（按赛制要求，电子版论文不得填写任何身份信息，此页请留空）")
    set_font(r, KAI, PT_WUHAO)


def build_toc(doc) -> None:
    p = styled_par(doc, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=12)
    r = p.add_run("目  录")
    set_font(r, HEI, PT_SIHAO)
    p2 = styled_par(doc)
    fld = _parse_xml(
        f'<w:fldSimple xmlns:w="{W_NS}" w:instr=" TOC \\o &quot;1-3&quot; \\h \\z \\u ">'
        f'<w:r><w:t>【请在 Word 中右键此处 → 更新域，生成目录】</w:t></w:r>'
        f'</w:fldSimple>')
    p2._p.append(fld)


# ---------------------------------------------------------------- Markdown 解析

def parse_md(md: str) -> Tuple[List[str], List[str]]:
    """把整篇 markdown 切成 (正文行, 摘要行)。

    规则：
      · `# 摘 要` 到下一个一级标题之间 = 摘要页内容；
      · 该一级标题起、直到文末 = 正文；
      · 摘要之前的一切（论文题目、以及写给作者自己看的排版说明块）
        都不进入正文——题目由 build() 单独取首行一级标题。
    """
    lines = md.split("\n")
    i_abs = next((i for i, ln in enumerate(lines) if ln.startswith("# 摘")), None)
    if i_abs is None:
        return lines, []

    i_body = next((j for j in range(i_abs + 1, len(lines))
                   if lines[j].startswith("# ")), len(lines))
    abstract = lines[i_abs + 1:i_body]
    body = lines[i_body:]
    return body, abstract


def _split_table_block(lines: List[str], start: int) -> int:
    j = start
    while j < len(lines) and lines[j].strip().startswith("|"):
        j += 1
    return j


def _table_rows(block: List[str]) -> List[List[str]]:
    rows = []
    for k, ln in enumerate(block):
        if k == 1 and re.match(r"^\|[\s:|-]+\|$", ln.strip()):
            continue
        # 按**未被转义的**竖线切分：单元格里的 LaTeX 常用 \| 表示范数/条件竖线
        # （如 $\mathrm{KL}(\mathbf{p}\|\mathbf{p}^{(-m)})$）。若照 naive 的 split("|")
        # 切，该单元格会被劈成两半，$...$ 配对随之断裂，最终在 Word 里原样漏出 LaTeX。
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", ln.strip().strip("|"))]
        rows.append(cells)
    return rows


def add_picture(doc, path: str, base_dir: str, width_cm: float = 13.0) -> None:
    """插入居中图片。path 为 .md 中的相对路径；宽高比过大的图自动缩到页内。"""
    full = path if os.path.isabs(path) else os.path.join(base_dir, path)
    if not os.path.isfile(full):
        raise FileNotFoundError(f"插图不存在：{full}（.md 中写的是 {path}）")
    p = styled_par(doc, align=WD_ALIGN_PARAGRAPH.CENTER,
                   space_before=6, space_after=2)
    p.add_run().add_picture(full, width=Cm(width_cm))


def emit(doc, lines: List[str], heading_levels: bool = True,
         base_dir: str = ".") -> None:
    i = 0
    n = len(lines)
    while i < n:
        ln = lines[i]
        s = ln.strip()

        # 插图：![说明](相对路径)  说明仅作 alt，正式图题用下一行的 **图 N ...**
        m = re.match(r"^!\[(.*?)\]\((.+?)\)\s*$", s)
        if m:
            add_picture(doc, m.group(2).strip(), base_dir)
            i += 1
            continue

        # 代码块
        if s.startswith("```"):
            i += 1
            buf = []
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            for b in buf:
                p = styled_par(doc, space_before=0, space_after=0, line=1.0)
                p.paragraph_format.left_indent = Cm(0.5)
                r = p.add_run(b if b.strip() else " ")
                set_font(r, MONO, PT_XIAOWU)
            styled_par(doc, space_after=4)
            continue

        # 分隔线
        if s == "---":
            i += 1
            continue

        # 表格
        if s.startswith("|"):
            j = _split_table_block(lines, i)
            rows = _table_rows(lines[i:j])
            if rows:
                add_three_line_table(doc, rows)
            i = j
            continue

        # 显示公式块
        if s.startswith("$$"):
            buf = []
            rest = s[2:]
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
            add_math_par(doc, "\n".join(x for x in buf if x.strip()))
            continue

        # 标题
        m = re.match(r"^(#{1,4})\s+(.*)$", s)
        if m:
            lvl, txt = len(m.group(1)), m.group(2)
            if lvl == 1:
                p = styled_par(doc, align=WD_ALIGN_PARAGRAPH.CENTER,
                               space_before=12, space_after=8)
                add_rich(p, txt, size=PT_SIHAO, base_font=HEI)
                for r in p.runs:
                    set_font(r, HEI, PT_SIHAO)
                for el in p._p.findall(qn("w:r")):
                    pass
            elif lvl == 2:
                # 二级标题左对齐：规范只规定「一级标题四号黑体居中」，
                # 其余汉字一律小四宋体，故二三级用 小四宋体加粗、左对齐。
                p = styled_par(doc, space_before=10, space_after=6)
                add_rich(p, txt, size=PT_XIAOSI, base_font=SONG)
                for r in p.runs:
                    set_font(r, SONG, PT_XIAOSI, bold=True)
            else:
                p = styled_par(doc, space_before=8, space_after=4)
                add_rich(p, txt, size=PT_XIAOSI, base_font=SONG)
                for r in p.runs:
                    set_font(r, SONG, PT_XIAOSI, bold=True)
            i += 1
            continue

        # 引用块（须如实说明）
        if s.startswith(">"):
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip().lstrip(">").strip())
                i += 1
            p = styled_par(doc, space_before=4, space_after=4,
                           left_indent=Cm(0.75), line=1.0)
            add_rich(p, " ".join(x for x in buf if x), size=PT_XIAOSI,
                     base_font=KAI)
            continue

        # 列表
        m = re.match(r"^([-*]|\d+\.)\s+(.*)$", s)
        if m:
            p = styled_par(doc, space_before=2, space_after=2,
                           left_indent=Cm(0.75), indent_first=Cm(-0.4))
            marker = "• " if m.group(1) in ("-", "*") else (m.group(1) + " ")
            add_rich(p, marker + m.group(2), size=PT_XIAOSI)
            i += 1
            continue

        # 空行
        if not s:
            i += 1
            continue

        # 表/图题（整行加粗）与普通段落
        is_caption = s.startswith("**") and re.match(r"^\*\*(表|图)\s*\d+", s)
        p = styled_par(
            doc,
            align=WD_ALIGN_PARAGRAPH.CENTER if is_caption else WD_ALIGN_PARAGRAPH.LEFT,
            space_before=4, space_after=4, line=1.0,
            indent_first=None if is_caption else Cm(0.85),
        )
        add_rich(p, s, size=PT_XIAOSI)
        i += 1


# ---------------------------------------------------------------- main

def build(md_path: str, out_path: str) -> int:
    with open(md_path, "r", encoding="utf-8") as f:
        md = f.read()

    body, abstract = parse_md(md)
    doc = Document()

    # 正文默认样式
    st = doc.styles["Normal"]
    st.font.name = SONG
    st.font.size = PT_XIAOSI
    st.element.rPr.rFonts.set(qn("w:eastAsia"), SONG)

    setup_page(doc.sections[0])
    clear_footer(doc.sections[0])
    build_cover(doc)

    # —— 摘要页：新节，页码从 1 开始
    doc.add_section(WD_SECTION.NEW_PAGE)
    setup_page(doc.sections[-1])
    add_page_number_footer(doc.sections[-1], restart_at=1)

    # 题目（三号黑体居中）
    title = None
    for ln in md.split("\n"):
        if ln.startswith("# ") and "摘" not in ln[:6]:
            title = ln[2:].strip()
            break
    p = styled_par(doc, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=10)
    r = p.add_run(title or "论文题目")
    set_font(r, HEI, PT_SANHAO)

    p = styled_par(doc, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=8)
    r = p.add_run("摘    要")
    set_font(r, HEI, PT_SIHAO)

    md_dir = os.path.dirname(os.path.abspath(md_path))
    emit(doc, abstract, base_dir=md_dir)

    add_page_break(doc)
    build_toc(doc)
    add_page_break(doc)
    emit(doc, body, base_dir=md_dir)

    doc.save(out_path)
    print(f"已生成：{out_path}")
    print(f"  摘要段数：{len([x for x in abstract if x.strip()])}")
    print(f"  正文章节：{sum(1 for x in body if re.match(r'^# ', x))} 个一级标题")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="论文 Markdown -> 符合格式规范的 docx")
    # 输入必须是**展开稿**，不是源稿：源稿里含 {TABLE:...} 占位符，直接编译会在
    # Word 里原样漏出（下面还有一道显式拦截）。
    #
    # 两个路径参数都设成必填，**不留默认值**：曾有过默认值，先是默认到一份问题一
    # 旧稿上（无参数运行会静默产出一份看着正常、内容却是废的论文），后来又默认到
    # 早已删除的三问稿上。默认值在这里只会制造"跑出来了、但是错的"这种最难查的
    # 情况；不给默认，跑错就停在 argparse 上，一步都走不下去。
    ap.add_argument("--md", required=True, help="展开稿（注入过表格的 markdown）")
    ap.add_argument("--out", required=True, help="输出的 docx 路径")
    a = ap.parse_args(argv)
    if not os.path.isfile(a.md):
        print(f"找不到输入：{a.md}")
        print("提示：先跑 `python _inject_paper_tables.py` 生成展开稿；"
              "出当前定稿请走 `python _build_q1_body.py`。")
        return 2
    src = open(a.md, encoding="utf-8").read()
    if re.search(r"\{[A-Z_]+:[A-Za-z0-9_]+\}|\{STRUCT_ABLATION\}", src):
        print(f"输入仍是未展开的源稿（含 {{TABLE:...}} 占位符）：{a.md}")
        print("提示：先跑 `python _inject_paper_tables.py`。")
        return 2
    return build(a.md, a.out)


if __name__ == "__main__":
    raise SystemExit(main())
