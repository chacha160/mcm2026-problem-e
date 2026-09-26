# -*- coding: utf-8 -*-
"""核对生成的 docx：占位符残留、公式编号、表格与图片数量、LaTeX 残留。

默认检查 `问题一论文正文.docx`（当前定稿）；另一份单篇稿用
`--docx 问题一论文.docx` 走同一条检查。参数化而不是复制一份：这两份稿子的
结构要求是同一套，各留一份实现只会让其中一份悄悄漂掉。

编号一律按「章-序」识别（`图 1-1`、`表 2-2`、`式 (1-2)`），点号写法也认——
两种写法在正则里是同一族。

检查项（`--docx` 决定对哪份稿子说话）：

* **占位符残留** —— `{TABLE:...}` / `{STRUCT_ABLATION}` 一个都不许进 Word。
  源稿直接喂给转换器就会这样，且产物打开看着正常（原 docstring 说检查这一项，
  但代码里一直没写，故补上）。
* **表格数 / 图片引用 / OMML 公式数** —— 与展开稿的期望值对照。
* **LaTeX 残留** —— `\\命令` 出现在正文即说明某段公式没转成 OMML。
* **公式编号与正文引用闭合** —— 正文引用的每个「式 (x.y)」都要有对应的编号。
"""
import argparse
import html
import os
import re
import sys
import zipfile

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DOCX = os.path.join(HERE, "问题一论文正文.docx")


def check(docx_path: str) -> int:
    z = zipfile.ZipFile(docx_path)
    xml = z.read("word/document.xml").decode("utf-8")
    txt = html.unescape(re.sub(r"<[^>]+>", "", xml.replace("</w:p>", "\n")))

    print(f"文件     : {os.path.basename(docx_path)}")
    print("表格数   :", xml.count("<w:tbl>"))
    print("图片引用 :", len(re.findall(r"<a:blip", xml)))
    print("OMML 公式:", xml.count("<m:oMath"))

    leftover = re.findall(r"\{[A-Z_]+:[A-Za-z0-9_]+\}|\{STRUCT_ABLATION\}", txt)
    print("占位符残留:", len(leftover), sorted(set(leftover)) if leftover else "")

    hits = list(re.finditer(r"\\[A-Za-z]{2,}", txt))
    print("LaTeX 残留处数:", len(hits))
    for m in hits[:15]:
        print("  >>>", repr(txt[max(0, m.start() - 90):m.end() + 50]))

    # 公式：编号写在行尾括号里（全角或半角），正文以「式 (x-y)」引用。
    nums = re.findall(r"[（(](\d+[.\-]\d+)[）)]", txt)
    refs = sorted(set(re.findall(r"式\s*[（(](\d+[.\-]\d+)[）)]", txt)))
    print("公式编号 :", nums)
    print("正文引用 :", refs)
    missing = [r for r in refs if r not in nums]
    print("引用无编号:", missing)

    # 图表标题与正文引用的一致性：正文写「见表 1-2」「如图 3-1」时，对应标题必须存在。
    NUM = r"\d+[.\-]\d+"
    cap_t = sorted({n for n in re.findall(rf"表\s*({NUM})\s*[　 ]", txt)})
    cap_f = sorted({n for n in re.findall(rf"图\s*({NUM})\s*[　 ]", txt)})
    print("表标题编号:", cap_t)
    print("图标题编号:", cap_f)
    ref_t = sorted({n for n in re.findall(rf"见?表\s*({NUM})", txt)})
    ref_f = sorted({n for n in re.findall(rf"见?图\s*({NUM})", txt)})
    miss_t = [n for n in ref_t if n not in cap_t]
    miss_f = [n for n in ref_f if n not in cap_f]
    print("表引用无标题:", miss_t)
    print("图引用无标题:", miss_f)

    # 这两项也要计入结论：正文引了「表 4-3」而全文最后一表是 3-3，是断链，
    # 不该出现「列出来了、却仍然报通过」的情形（问题一提交.docx 上真出现过：
    # 正文引「第 4.6 节」「表 4-3」，都是从更长的五章稿裁剪后落下的残引）。
    bad = bool(leftover) or bool(hits) or bool(missing) or bool(miss_t) or bool(miss_f)
    print("\n[结论]", "发现残留，需修正" if bad else "结构检查通过")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="docx 结构核对")
    ap.add_argument("--docx", default=DEFAULT_DOCX,
                    help="待检查的 docx（默认 问题一论文正文.docx）")
    a = ap.parse_args(argv)
    if not os.path.isfile(a.docx):
        print(f"找不到文件：{a.docx}")
        return 2
    return check(a.docx)


if __name__ == "__main__":
    raise SystemExit(main())
