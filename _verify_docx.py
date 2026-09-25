# -*- coding: utf-8 -*-
"""核对生成的 docx：占位符残留、公式编号、表格与图片数量、LaTeX 残留。"""
import io, re, sys, zipfile, html
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
DOCX = r'D:\23届建模\论文.docx'
z = zipfile.ZipFile(DOCX)
xml = z.read('word/document.xml').decode('utf-8')
txt = html.unescape(re.sub(r'<[^>]+>', '', xml.replace('</w:p>', '\n')))

print('表格数   :', xml.count('<w:tbl>'))
print('图片引用 :', len(re.findall(r'<a:blip', xml)))
print('OMML 公式:', xml.count('<m:oMath'))
hits = list(re.finditer(r'\\[A-Za-z]{2,}', txt))
print('LaTeX 残留处数:', len(hits))
for m in hits[:15]:
    print('  >>>', repr(txt[max(0, m.start() - 90):m.end() + 50]))
nums = re.findall(r'（(\d+\.\d+)）', txt)
print('公式编号 :', [n for n in nums if re.match(r'^5\.\d+$|^6\.\d+$', n)])
refs = sorted(set(re.findall(r'式 \((\d+\.\d+)\)', txt)))
print('正文引用 :', refs)
print('引用无编号:', [r for r in refs if r not in nums])
