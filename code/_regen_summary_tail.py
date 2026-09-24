# -*- coding: utf-8 -*-
"""
_regen_summary_tail.py —— 一次性恢复脚本：重写对齐管线的两个**派生** CSV 尾部产物。

背景
----
`align_multimodal.run()` 的尾部顺序是
    save_aggregate → save_slot_time_map → save_summary_csv
其中 `slot_time_map.csv` 被外部程序（WPS）占用，`PermissionError` 使 `save_summary_csv`
没有机会执行，于是 `align_summary.csv` 停留在旧版本（含 18 行 official_id 缺陷）。

本脚本**不重新对齐**，而是从已经写好的单样本对齐产物里把 `run()` 需要的 `results`
结构原样重建（`duration` / `slot_edges` / 每模态的 `src_duration` / `num_units` /
`feature_dim` / `time_basis` / `valid` / `align` 统计全部取自文件），再调用同一个
`save_summary_csv()`。因此产出与正常跑完 `run()` 完全一致，可逐字节比对验证。

注意：这不是常规流程的一部分，只是绕开文件占用的恢复手段。正常复现仍应执行
    python align_multimodal.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import align_multimodal as A  # noqa: E402
import unaligned_common as U  # noqa: E402
from utils import LOGGER  # noqa: E402


def rebuild_results(aligned_dir: str):
    """从单样本对齐文件重建 run() 内使用的 results 结构（只读）。"""
    results = []
    for s in U.list_samples():
        sid = str(s["sample_id"])
        with np.load(os.path.join(aligned_dir, f"{sid}.npz"), allow_pickle=True) as z:
            meta = json.loads(str(np.asarray(z["meta"]).item()))
            r = {
                "sample_id": sid,
                "duration": float(np.asarray(z["duration"], np.float64)),
                "slot_edges": np.asarray(z["slot_edges"], np.float64),
                "modalities": {},
            }
            for m in A.MODALITIES:
                r["modalities"][m] = {
                    "src_duration": meta[m]["src_duration"],
                    "num_units": meta[m]["num_units"],
                    "feature_dim": meta[m]["feature_dim"],
                    "time_basis": meta[m]["time_basis"],
                    "valid": np.asarray(z[f"{m}_valid"]),
                    # ⚠️ `slot_unit_index` **必须拼回**：落盘时它被有意从 `align` 里提出去、
                    # 放到模态一级（见 align_multimodal.save_sample_aligned 的
                    # 「align = {k: v for k, v in md["align_meta"].items() if k != "slot_unit_index"}」），
                    # 而 run() 内存里的 align_meta 是**含** slot_unit_index 的完整 meta。
                    # 只取 meta[m]["align"] 会得到一个缺键的字典，save_slot_time_map()
                    # 读 `md["align_meta"]["slot_unit_index"][k]` 直接 KeyError。
                    # 本项目真踩过：本题一次跑就在这一行崩了。
                    "align_meta": dict(meta[m]["align"],
                                       slot_unit_index=meta[m]["slot_unit_index"]),
                }
        results.append(r)
    return results


def main() -> int:
    aligned_dir = os.path.join(os.path.dirname(_CODE_DIR), "data", "aligned")
    results = rebuild_results(aligned_dir)
    LOGGER.info("[恢复] 从 %d 个单样本对齐文件重建 results", len(results))

    done, blocked = [], []
    for name, fn in (("slot_time_map.csv", lambda: A.save_slot_time_map(
                          aligned_dir, results, int(np.asarray(
                              results[0]["modalities"]["text"]["valid"]).size))),
                     ("align_summary.csv", lambda: A.save_summary_csv(aligned_dir, results))):
        try:
            LOGGER.info("[恢复] 写出 %s", fn())
            done.append(name)
        except PermissionError as exc:
            LOGGER.warning("[恢复] %s 被外部程序占用，跳过（请关闭后重跑本脚本）：%s", name, exc)
            blocked.append(name)
    LOGGER.info("[恢复] 完成 %s；被占用 %s", done, blocked)
    return 0 if not blocked else 2


if __name__ == "__main__":
    sys.exit(main())
