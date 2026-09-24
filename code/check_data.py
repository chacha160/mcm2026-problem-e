"""
check_data.py
数据诊断脚本：检查路径、文件、文件夹结构是否正确
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

print("=" * 60)
print("数据路径诊断报告")
print("=" * 60)

# 1. 检查附件1目录
att_dir = config.ATTACHMENT1_DIR
print(f"\n[1] ATTACHMENT1_DIR 设置路径: {att_dir}")
if os.path.exists(att_dir):
    print(f"    状态: ✅ 存在")
    items = os.listdir(att_dir)
    print(f"    内容: {items[:20]}{'...' if len(items) > 20 else ''}")
else:
    print(f"    状态: ❌ 不存在！请修改 config.py 中的 ATTACHMENT1_DIR")

# 2. 检查标注表 label-100.xlsx 及其 label 工作表的 text 列
label_path = os.path.join(att_dir, config.LABEL_XLSX_NAME)
print(f"\n[2] 标注文件路径: {label_path}")
print(f"    读取工作表: {config.LABEL_SHEET_NAME}（text 列 = 英文转写文本）")
if os.path.exists(label_path):
    print(f"    状态: ✅ 存在")
    try:
        import pandas as pd
        from utils import load_label_table
        # 列出该文件的所有工作表，便于确认 label 工作表是否存在
        print(f"    工作表列表: {pd.ExcelFile(label_path).sheet_names}")
        df = load_label_table(label_path, sheet_name=config.LABEL_SHEET_NAME)
        print(f"    样本数量: {len(df)}")
        print(f"    列名: {list(df.columns)}")
        print(f"    text 列非空样本数: {int((df['text'].str.len() > 0).sum())}")
        print(f"    前3行:")
        print(df.head(3).to_string(index=False))
    except Exception as e:
        print(f"    读取失败: {e}")
else:
    print(f"    状态: ❌ 不存在！")
    # 尝试查找目录下所有xlsx
    if os.path.exists(att_dir):
        xlsx_files = [f for f in os.listdir(att_dir) if f.endswith('.xlsx')]
        if xlsx_files:
            print(f"    提示: 该目录下发现其他xlsx文件: {xlsx_files}")
            print(f"    建议: 将 {config.LABEL_XLSX_NAME} 放入该目录，或修改 config.py 中的文件名")

# 3. 检查视频文件夹结构
print(f"\n[3] 视频文件夹结构检查:")
if os.path.exists(att_dir):
    subdirs = [d for d in os.listdir(att_dir) if os.path.isdir(os.path.join(att_dir, d))]
    print(f"    发现子文件夹数: {len(subdirs)}")
    if subdirs:
        print(f"    前5个子文件夹: {subdirs[:5]}")
        # 检查第一个子文件夹内是否有mp4
        first_dir = os.path.join(att_dir, subdirs[0])
        files = os.listdir(first_dir)
        mp4_files = [f for f in files if f.endswith('.mp4')]
        print(f"    例如 '{subdirs[0]}' 内文件: {files[:10]}")
        print(f"    其中MP4数量: {len(mp4_files)}")

# 4. 检查当前流水线的输出路径。
#    config.OUTPUT_DIR / TEMP_AUDIO_DIR 这两条遗留常量**不再检查**：它们属于已删除的
#    上一版流水线，报「将自动创建」是错的——现在没有任何代码会去创建它们。
_PCODE = os.path.dirname(os.path.abspath(__file__))
print(f"\n[4] 当前流水线输出路径检查:")
for _label, _sub in (("未对齐特征", "unaligned_features"),
                     ("对齐结果", "aligned"),
                     ("问题一交付物", "q1_delivery"),
                     ("问题二模型", "model")):
    _p = os.path.normpath(os.path.join(_PCODE, "..", "data", _sub))
    print(f"    {_label}: {_p} -> {'✅ 存在' if os.path.exists(_p) else '（尚未生成）'}")

print("\n" + "=" * 60)
print("诊断结束")
print("=" * 60)
