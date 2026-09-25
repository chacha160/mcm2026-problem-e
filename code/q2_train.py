# -*- coding: utf-8 -*-
"""
q2_train.py —— 问题二：缺失模态场景下的鲁棒情感预测

赛题口径（逐条对应，不自行加码也不遗漏）
----------------------------------------
赛题问题2与问题3共用的「说明」段「用附件2 的训练集训练、验证集验证，对附件3 做推理」：
    · 训练 = 附件2 `train`（3395 条）
    · 验证 = 附件2 `valid`（728 条），**只用于早停与超参选择**
    · 附件2 `test`（727 条）**不参与任何选择**，仅作最终报告
    · 推理 = 附件3 的 30 条，导出预测 CSV
赛题问题2「模态缺失下仍能稳定预测情感极性与强度」：
    主结果表 + 退化曲线（按附件3 实际缺失槽数分档）
赛题问题2「分析缺失模态类型、缺失位置、缺失时长三因素影响」：
    `run_missingness_experiment()` 做 4 类型 × 4 位置 × 3 时长 = 48 组受控实验
红线 R1（不增删改样本与标签）：
    受控缺失实验只在**内存中的副本**上做，绝不落盘、绝不改写附件2/3 的原文件；
    实验产生的「人造缺失」在报告中一律标注为 synthetic，与附件3 的真实注入缺失分开。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import config  # noqa: E402
import q2q3_common as Q  # noqa: E402
import q2_model as M  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

OUT_DEFAULT = os.path.join(config.PROJECT_ROOT, "data", "q2")
SEED = 42


# ==================== 训练 ====================

def _forward_loss(model, batch_t, criterion, y_pol=None, y_int=None,
                  modality_dropout: bool = False):
    out = model(batch_t, modality_dropout=modality_dropout)
    if y_pol is None:
        return out, None
    loss = criterion(out, y_pol, y_int)
    return out, loss


@torch.no_grad()
def evaluate(model, batch_np: Dict[str, np.ndarray],
             y_pol: Optional[np.ndarray] = None,
             y_int: Optional[np.ndarray] = None,
             device: torch.device = torch.device("cpu"),
             batch_size: int = 128) -> Dict[str, np.ndarray]:
    """返回逐样本预测（极性类别、强度），可选带上标签一并算指标。"""
    model.eval()
    n = len(batch_np["text_ids"])
    pred_pol = np.zeros(n, dtype=np.int64)
    pred_int = np.zeros(n, dtype=np.float64)
    prob = np.zeros((n, 3), dtype=np.float64)
    for idx in M.batch_indices(n, batch_size):
        sub = {k: v[idx] for k, v in batch_np.items()}
        out = model(M.to_torch(sub, device))
        p = torch.softmax(out["polarity_logits"], dim=-1).cpu().numpy()
        prob[idx] = p
        pred_pol[idx] = p.argmax(axis=1)
        pred_int[idx] = out["intensity"].cpu().numpy()
    res = {"polarity_idx": pred_pol, "intensity": pred_int, "polarity_prob": prob}
    if y_pol is not None:
        pm = M.polarity_metrics(pred_pol, y_pol)
        im = M.intensity_metrics(pred_int, y_int)
        res.update(pm)
        res.update(im)
        res["n"] = n
    return res


def train_model(train_np: Dict[str, np.ndarray], y_pol_tr: np.ndarray, y_int_tr: np.ndarray,
                valid_np: Dict[str, np.ndarray], y_pol_va: np.ndarray, y_int_va: np.ndarray,
                epochs: int, batch_size: int, lr: float, patience: int,
                device: torch.device, verbose: bool = True,
                seed: int = SEED,
                text_emb_table: Optional[np.ndarray] = None,
                structure: str = "full"
                ) -> Tuple[M.MaskedTriModalModel, Dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = M.ModelConfig()
    model = M.MaskedTriModalModel(cfg, text_emb_table=text_emb_table,
                                  structure=structure).to(device)
    if verbose:
        print(f"  文本支路初始化：{'官方768维蒸馏表（PCA→64）' if text_emb_table is not None else '随机'}"
              f"，参数量 {sum(p.numel() for p in model.parameters()):,}")
    cw = M.make_class_weights(y_pol_tr).to(device)
    criterion = M.MultiTaskLoss(cw).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    rng = np.random.default_rng(seed)
    n = len(y_pol_tr)
    best = {"score": -1e9, "epoch": -1, "state": None, "valid": None}
    hist: List[Dict] = []

    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        for idx in M.batch_indices(n, batch_size, shuffle=True, rng=rng):
            sub = {k: v[idx] for k, v in train_np.items()}
            bt = M.to_torch(sub, device)
            yp = torch.as_tensor(y_pol_tr[idx], device=device)
            yi = torch.as_tensor(y_int_tr[idx], dtype=torch.float32, device=device)
            opt.zero_grad()
            _out, loss = _forward_loss(model, bt, criterion, yp, yi, modality_dropout=True)
            loss["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss["total"].detach()))
        sched.step()

        vm = evaluate(model, valid_np, y_pol_va, y_int_va, device)
        # 选模型的主指标：宏 F1 与 1-MAE/3 的调和平均。
        # 只用宏 F1 会漏掉强度，只用 MAE 会被「恒输出 2」骗过，故两者并列要求。
        sel = 0.5 * vm["macro_f1"] + 0.5 * (1.0 - vm["mae"] / cfg.max_intensity)
        hist.append({"epoch": ep, "train_loss": float(np.mean(losses)),
                     "valid_macro_f1": vm["macro_f1"], "valid_mae": vm["mae"],
                     "valid_pearson": vm["pearson"], "valid_accuracy": vm["accuracy"],
                     "valid_neutral_recall": vm["recall_neutral"], "select_score": sel,
                     "sec": round(time.time() - t0, 1)})
        if verbose:
            print(f"  epoch {ep:3d}  loss={np.mean(losses):.4f}  "
                  f"valid acc={vm['accuracy']:.3f} macroF1={vm['macro_f1']:.3f} "
                  f"MAE={vm['mae']:.3f} r={vm['pearson']:.3f}  sel={sel:.4f}")

        if sel > best["score"]:
            best.update({"score": sel, "epoch": ep, "valid": {
                k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                for k, v in vm.items() if k in ("accuracy", "macro_f1", "mae", "pearson",
                                                "recall_neutral", "recall_negative",
                                                "recall_positive")},
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}})
        elif ep - best["epoch"] >= patience:
            if verbose:
                print(f"  早停：第 {ep} 轮，最佳为第 {best['epoch']} 轮")
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model, {"best_epoch": best["epoch"], "best_select_score": best["score"],
                   "best_valid": best["valid"], "history": hist}


# ==================== 受控缺失实验（赛题问题2的三因素要求）====================

MISS_TYPES = {
    "text": (0,),
    "audio": (1,),
    "vision": (2,),
    "audio+vision": (1, 2),
}
MISS_POSITIONS = ("head", "middle", "tail", "random")
MISS_DURATIONS = {"short": 0.2, "medium": 0.4, "long": 0.6}


def inject_missing(batch: Dict[str, np.ndarray], miss_mods: Sequence[int],
                   position: str, ratio: float, seed: int) -> Dict[str, np.ndarray]:
    """在内存副本上注入受控缺失，返回新的批次（原批次不被修改）。

    注入方式刻意与附件3 的生成方式同构：把某个模态在**它自己有观测的连续槽
    区间**上置零，并同步更新 `miss_flag` 与 `obs_mask`，使模型看到的信号与
    附件3 真实缺失时完全一致（否则实验测的不是同一件事）。

    关键点：`obs_mask` 是**逐模态**的，所以「屏蔽文本」只改文本那一层，语音与
    视觉在同一槽位上的观测不受影响。若把掩码做成跨模态并集，四种缺失类型会
    退化成同一件事——这正是本函数先前版本的缺陷。
    """
    rng = np.random.default_rng(seed)
    out = {k: v.copy() for k, v in batch.items()}
    n = out["obs_mask"].shape[0]
    for i in range(n):
        for m in miss_mods:
            idx = np.flatnonzero(out["obs_mask"][i, m])   # 该模态自己的观测槽
            Lm = int(idx.size)
            if Lm <= 1:
                continue
            k = min(max(1, int(round(ratio * Lm))), Lm - 1)
            if position == "head":
                s = 0
            elif position == "tail":
                s = Lm - k
            elif position == "middle":
                s = max(0, (Lm - k) // 2)
            else:  # random
                s = int(rng.integers(0, max(1, Lm - k + 1)))
            sel = idx[s:s + k]
            if m == 0:
                out["text_ids"][i, sel] = Q.PAD_TOKEN_ID
            elif m == 1:
                out["audio"][i, sel, :] = 0.0
            else:
                out["vision"][i, sel, :] = 0.0
            out["obs_mask"][i, m, sel] = False
            out["miss_flag"][i, m, sel] = True
    # 模态整体可用性：按「该模态是否还有任何可用槽」重算
    for m in range(3):
        if m == 0:
            alive = out["text_ids"] != Q.PAD_TOKEN_ID
        elif m == 1:
            alive = np.abs(out["audio"]).sum(-1) != 0
        else:
            alive = np.abs(out["vision"]).sum(-1) != 0
        out["modal_avail"][:, m] = alive.any(axis=1)
    return out


def run_missingness_experiment(model, valid_np: Dict[str, np.ndarray],
                               y_pol: np.ndarray, y_int: np.ndarray,
                               device: torch.device, seed: int = SEED) -> List[Dict]:
    """48 组受控实验：类型 × 位置 × 时长。基线为「不注入缺失」。"""
    rows: List[Dict] = []

    base = evaluate(model, valid_np, y_pol, y_int, device)
    # 基线行也必须带齐「干预落点」三列（值为 0）：`_write_csv` 以首行键为表头，
    # 缺列会让后续行写不进去。
    rows.append({"missing_type": "none", "position": "none", "duration": "none",
                 "ratio": 0.0, "accuracy": base["accuracy"], "macro_f1": base["macro_f1"],
                 "mae": base["mae"], "pearson": base["pearson"],
                 "neutral_recall": base["recall_neutral"],
                 "slots_cut_text": 0, "slots_cut_audio": 0, "slots_cut_vision": 0,
                 "slots_cut_target_only": True})

    for tname, mods in MISS_TYPES.items():
        for pos in MISS_POSITIONS:
            for dname, ratio in MISS_DURATIONS.items():
                inj = inject_missing(valid_np, mods, pos, ratio, seed)
                m = evaluate(model, inj, y_pol, y_int, device)
                # 干预落点核账：被屏蔽的槽必须**只**落在目标模态上。
                # 若「屏蔽文本」把语音/视觉的观测也抹掉了，这两列会露馅。
                # 这正是先前版本把掩码做成跨模态并集时所犯的错误，
                # 故此处把「改了多少」写进结果表，让实验自证。
                cut = ~inj["obs_mask"] & valid_np["obs_mask"]
                rows.append({
                    "missing_type": tname, "position": pos, "duration": dname,
                    "ratio": ratio, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                    "mae": m["mae"], "pearson": m["pearson"],
                    "neutral_recall": m["recall_neutral"],
                    "slots_cut_text": int(cut[:, 0].sum()),
                    "slots_cut_audio": int(cut[:, 1].sum()),
                    "slots_cut_vision": int(cut[:, 2].sum()),
                    "slots_cut_target_only": bool(
                        cut[:, [m for m in range(3) if m not in mods]].sum() == 0),
                })
    return rows


# ==================== 结构消融实验（赛题问题2「消融实验」）====================
#
# 48 组受控缺失实验回答的是「缺失类型/位置/时长各有多大影响」，
# 属于**敏感性分析**；它不回答「本文的缺失感知设计本身值不值得」。
# 后者需要一个把结构信号逐路拆掉的对照，即本节。
#
# 探针条件取「文本缺失 · 中部 · 40%」：文本是唯一显著有害的缺失类型
# （见受控实验规则一），因此最能暴露结构差异；固定成单一条件是刻意的，
# 目的是让五行结果之间**只差结构信号这一路**，不掺入缺失配置的变化。
#
# 五个变体共用同一份训练代码、同一随机种子、同一轮数与同一蒸馏初值，
# 唯一差异由 `q2_model.STRUCTURE_CHOICES` 的分支引入。

STRUCTURE_LABELS = {
    "full": "完整（交付配置）",
    "no_state_emb": "去掉缺失状态嵌入",
    "no_obs_mask": "注意力不屏蔽缺失槽",
    "union_mask": "掩码退化为跨模态并集",
    "no_md": "去掉训练期模态丢弃",
}
STRUCTURE_PROBE = ("text", "middle", "medium")   # 类型 / 位置 / 时长


def run_structure_ablation(train_np, y_pol_tr, y_int_tr, valid_np, y_pol_va, y_int_va,
                           test_np, y_pol_te, y_int_te, table, args, device) -> List[Dict]:
    """逐路拆掉结构信号，在同一受控缺失探针下对比。"""
    tname, pos, dname = STRUCTURE_PROBE
    mods = MISS_TYPES[tname]
    ratio = MISS_DURATIONS[dname]
    probe_va = inject_missing(valid_np, mods, pos, ratio, args.seed)
    probe_te = inject_missing(test_np, mods, pos, ratio, args.seed)

    rows: List[Dict] = []
    for st in M.STRUCTURE_CHOICES:
        print(f"\n[结构消融] {st} —— {STRUCTURE_LABELS[st]}")
        mdl, hist = train_model(train_np, y_pol_tr, y_int_tr,
                                valid_np, y_pol_va, y_int_va,
                                args.epochs, args.batch_size, args.lr, args.patience,
                                device, verbose=False, seed=args.seed,
                                text_emb_table=table, structure=st)
        clean_va = evaluate(mdl, valid_np, y_pol_va, y_int_va, device)
        probe_va_m = evaluate(mdl, probe_va, y_pol_va, y_int_va, device)
        probe_te_m = evaluate(mdl, probe_te, y_pol_te, y_int_te, device)
        rows.append({
            "structure": st, "structure_label": STRUCTURE_LABELS[st],
            "probe": f"{tname}/{pos}/{dname}",
            "best_epoch": hist["best_epoch"],
            # 无缺失条件下的指标：用于确认「拆掉结构信号」不是把干净场景也弄坏了
            "clean_valid_macro_f1": clean_va["macro_f1"],
            "clean_valid_accuracy": clean_va["accuracy"],
            "clean_valid_mae": clean_va["mae"],
            "clean_valid_pearson": clean_va["pearson"],
            # 探针条件下的指标：这里才是结构信号该发挥作用的地方
            "probe_valid_macro_f1": probe_va_m["macro_f1"],
            "probe_valid_accuracy": probe_va_m["accuracy"],
            "probe_valid_mae": probe_va_m["mae"],
            "probe_valid_pearson": probe_va_m["pearson"],
            "probe_test_macro_f1": probe_te_m["macro_f1"],
            "probe_test_accuracy": probe_te_m["accuracy"],
            "probe_test_mae": probe_te_m["mae"],
            "probe_test_pearson": probe_te_m["pearson"],
        })
        print(f"    干净 valid: macroF1={clean_va['macro_f1']:.4f} MAE={clean_va['mae']:.4f}"
              f"  |  探针 valid: macroF1={probe_va_m['macro_f1']:.4f} "
              f"MAE={probe_va_m['mae']:.4f} r={probe_va_m['pearson']:.4f}")
    return rows


def summarize_missingness(rows: List[Dict]) -> Dict[str, Dict[str, float]]:
    """把 48 组实验按三个因素各自聚合，直接产出赛题问题2要的「影响规律」。"""
    def agg(key: str) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        groups: Dict[str, List[Dict]] = {}
        for r in rows:
            if r["missing_type"] == "none":
                continue
            groups.setdefault(r[key], []).append(r)
        for g, rs in groups.items():
            out[g] = {
                "macro_f1": float(np.mean([x["macro_f1"] for x in rs])),
                "accuracy": float(np.mean([x["accuracy"] for x in rs])),
                "mae": float(np.mean([x["mae"] for x in rs])),
                "pearson": float(np.mean([x["pearson"] for x in rs])),
                "n_configs": len(rs),
            }
        return out
    return {"by_missing_type": agg("missing_type"),
            "by_position": agg("position"),
            "by_duration": agg("duration")}


# ==================== 错误归因（赛题 L46 / L52）====================

def error_analysis(pred: Dict[str, np.ndarray], y_pol: np.ndarray, y_int: np.ndarray,
                   batch_np: Dict[str, np.ndarray],
                   sample_names: Optional[Sequence[str]] = None) -> Dict:
    """把「错了多少」拆成「错在哪」，这是 L46/L52 要的「错误归因结论」。

    四个切面，每个都给出可直接写进论文的句子级结论：

    1. **混在哪个类之间**：3×3 混淆矩阵（行归一化召回）。三分类里
       Neutral 通常是被牺牲的那一类，必须用数字点明而非含糊带过。
    2. **错误是否集中在强度而非极性**：分别统计「极性判对/判错」两组的强度 MAE。
       若极性判错的样本强度误差显著更大，说明两类任务共享同一批难样本。
    3. **错误与缺失规模的关系**：按样本缺失槽数分档（0 / 1–2 / 3–5 / >5），
       逐档给出四项指标。这是「缺失鲁棒」是否成立的直接检验。
    4. **错误与模态可用性的关系**：按「哪些模态可用」分组（如 TAV / TA / TV / T …），
       给出每组样本数与准确率。这能暴露「某个模态缺席时是不是就崩了」。
    """
    pi = np.asarray(pred["polarity_idx"], dtype=np.int64)
    pi_hat = np.asarray(y_pol, dtype=np.int64)
    inten = np.asarray(pred["intensity"], dtype=np.float64)
    y_true_int = np.asarray(y_int, dtype=np.float64)
    n = len(pi)
    correct = pi == pi_hat

    # 1) 混淆矩阵
    cm = np.zeros((3, 3), dtype=np.int64)
    for t in range(3):
        for p in range(3):
            cm[t, p] = int(np.sum((pi_hat == t) & (pi == p)))
    recall = [float(cm[t, t] / cm[t].sum()) if cm[t].sum() else float("nan") for t in range(3)]
    precision = [float(cm[t, t] / cm[:, t].sum()) if cm[:, t].sum() else float("nan")
                 for t in range(3)]
    labels = [Q.POLARITY_CN[v] for v in Q.POLARITY_ORDER]
    confusion = {
        "labels": labels,
        "matrix_true_by_pred": cm.tolist(),
        "recall": dict(zip(labels, recall)),
        "precision": dict(zip(labels, precision)),
    }

    # 错误去向：真实类 t 的错分都跑到了哪些类
    misroute: Dict[str, Dict[str, int]] = {}
    for t in range(3):
        row = {labels[p]: int(cm[t, p]) for p in range(3) if p != t}
        dup = {k: v for k, v in row.items() if v > 0}
        if dup:
            misroute[labels[t]] = dup
    neutral_involved = int(np.sum(~correct & ((pi_hat == 1) | (pi == 1))))

    # 2) 极性对/错两组的强度误差
    def _mae_sub(mask) -> Optional[float]:
        return float(np.mean(np.abs(inten[mask] - y_true_int[mask]))) if mask.any() else None
    intensity_by_polarity = {
        "mae_when_polarity_correct": _mae_sub(correct),
        "mae_when_polarity_wrong": _mae_sub(~correct),
        "n_correct": int(correct.sum()), "n_wrong": int((~correct).sum()),
    }

    # 3) 按缺失规模分档
    # 只数**有效区内的缺失槽**（miss_flag 的定义），不含尾部填充
    n_missing = np.asarray(batch_np["miss_flag"].sum(axis=(1, 2)), dtype=np.int64)

    def _bucket(k: int) -> str:
        return "0" if k == 0 else ("1-2" if k <= 2 else ("3-5" if k <= 5 else ">5"))
    by_missing: Dict[str, Dict] = {}
    for b in ("0", "1-2", "3-5", ">5"):
        m = np.array([_bucket(int(x)) == b for x in n_missing])
        if not m.any():
            continue
        pm = M.polarity_metrics(pi[m], pi_hat[m])
        im = M.intensity_metrics(inten[m], y_true_int[m])
        by_missing[b] = {"n": int(m.sum()), "accuracy": pm["accuracy"],
                         "macro_f1": pm["macro_f1"], "mae": im["mae"],
                         "pearson": im["pearson"]}

    # 4) 按模态可用性分组
    avail = np.asarray(batch_np["modal_avail"], dtype=bool)      # (N,3)
    keys = []
    for i in range(n):
        keys.append("".join(m[0].upper() for m, a in zip(Q.MODALITIES, avail[i]) if a) or "none")
    by_modality: Dict[str, Dict] = {}
    for k in sorted(set(keys)):
        m = np.array([x == k for x in keys])
        pm = M.polarity_metrics(pi[m], pi_hat[m])
        im = M.intensity_metrics(inten[m], y_true_int[m])
        by_modality[k] = {"n": int(m.sum()), "accuracy": pm["accuracy"],
                          "macro_f1": pm["macro_f1"], "mae": im["mae"]}

    # 只在样本数够的分组间比较：n=1 的分组能轻松刷出「最差/最好」，
    # 拿它下结论是把噪声当规律。样本不足时如实说明，不硬凑结论。
    solid = {k: v for k, v in by_modality.items() if v["n"] >= 10}
    worst = max(solid.items(), key=lambda kv: kv[1]["mae"]) if len(solid) >= 2 else None
    best = min(solid.items(), key=lambda kv: kv[1]["mae"]) if len(solid) >= 2 else None
    small = {k: v["n"] for k, v in by_modality.items() if v["n"] < 10}
    conclusions = [
        f"极性总体准确率 {float(correct.mean()):.4f}（{int(correct.sum())}/{n}），"
        f"其中 Neutral 召回 {recall[1]:.4f} 为三类最低。",
        f"{neutral_involved} 个错误至少一侧是 Neutral，占全部错误的 "
        f"{neutral_involved / max(1, int((~correct).sum())):.1%}——"
        f"错误高度集中在「中性 vs 非中性」这条边界上，而不是随机散布。",
    ]
    if intensity_by_polarity["mae_when_polarity_wrong"] is not None and \
            intensity_by_polarity["mae_when_polarity_correct"] is not None:
        conclusions.append(
            f"极性判错组的强度 MAE "
            f"{intensity_by_polarity['mae_when_polarity_wrong']:.4f}，"
            f"判对组 {intensity_by_polarity['mae_when_polarity_correct']:.4f}——"
            + ("错误共享同一批难样本。" if
               intensity_by_polarity["mae_when_polarity_wrong"] >
               intensity_by_polarity["mae_when_polarity_correct"]
               else "两类任务捕捉的难点不同，可分别优化。"))
    if worst and best:
        # 结论的**方向必须由数据决定**，且必须**三个指标口径一致**才敢下方向性判断。
        # 踩过两次坑：
        #   1) 预设「模态越少越差」——实测就有反例；
        #   2) 只看 MAE 定方向，却用「模态更少反而更好」这种**概括性**措辞——
        #      实测 TA（n=15）MAE 0.4590 优于 TAV（n=713）0.5002，
        #      但同一组的准确率只有 0.2667（TAV 是 0.5596），
        #      即两个口径给出**相反**的结论，此时任何方向性断言都是错的。
        def _k(tag: str) -> int:
            return 0 if tag == "none" else len(tag)
        # 每个指标各自的方向：+1 表示「模态少的更差」
        votes = []
        for met, higher_is_worse in (("mae", True), ("accuracy", False), ("macro_f1", False)):
            if met not in worst[1] or met not in best[1]:
                continue
            d = worst[1][met] - best[1][met]
            if abs(d) < (0.02 if met == "mae" else 0.05):     # 差距不显著
                votes.append(0)
            else:
                votes.append((1 if d > 0 else -1) * (1 if higher_is_worse else -1))
        agree = [v for v in votes if v != 0]
        consistent = bool(agree) and all(v == agree[0] for v in agree)
        if len(agree) < 2:
            direction = "两者差距不足 0.02(MAE)/0.05(Acc,F1)，说明模型对模态缺席基本不敏感。"
        elif not consistent:
            direction = ("**三个指标口径不一致**：该组 MAE 更优但分类指标更差"
                         "（或反之），故**不下「模态少更好/更差」的方向性结论**；"
                         "样本量小的组尤其不可据此推断，详见上方各指标数值。")
        elif _k(worst[0]) < _k(best[0]):
            direction = "模态更少的那组确实更差，缺失鲁棒性仍有空间。"
        elif _k(worst[0]) > _k(best[0]):
            direction = ("方向与直觉相反：**模态更少的那组反而更好**"
                         "（三个指标口径一致）——说明被拿掉的那个模态在这批样本上"
                         "贡献的信息少于它带来的噪声，与受控缺失实验里"
                         "「屏蔽语音/视觉后指标略升」的现象一致。")
        else:
            direction = "两者模态数相同，差异来自模态**组合**而非数量。"
        conclusions.append(
            f"模态可用性分组里 MAE 最差的是 {worst[0]}（{worst[1]['mae']:.4f}，"
            f"准确率 {worst[1].get('accuracy', float('nan')):.4f}，n={worst[1]['n']}），"
            f"最好的是 {best[0]}（{best[1]['mae']:.4f}，"
            f"准确率 {best[1].get('accuracy', float('nan')):.4f}，n={best[1]['n']}）——"
            + direction)
    else:
        conclusions.append(
            "模态可用性分组中样本数 ≥10 的组别不足两个，不足以比较"
            "「缺哪个模态更伤」——本批数据的模态组合过于集中，此处不下结论。")
    if small:
        conclusions.append(
            f"以下模态组合样本过少，仅为记录、不作比较依据：{small}")

    out: Dict[str, object] = {
        "n": n, "confusion": confusion, "misroute_by_true_class": misroute,
        "neutral_related_errors": neutral_involved,
        "intensity_by_polarity_correctness": intensity_by_polarity,
        "by_missing_slot_count": by_missing,
        "by_available_modalities": by_modality,
        "conclusions": conclusions,
    }
    if sample_names is not None:
        out["misclassified_samples"] = [
            {"sample": sample_names[i], "true": labels[int(pi_hat[i])],
             "pred": labels[int(pi[i])],
             "true_intensity": round(float(y_true_int[i]), 4),
             "pred_intensity": round(float(inten[i]), 4)}
            for i in np.flatnonzero(~correct)[:40]]
    return out


def write_error_analysis(an: Dict, out_dir: str, tag: str = "valid") -> None:
    """落盘错误归因：JSON 供论文引用，CSV 供复核，另出混淆矩阵图。"""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"error_analysis_{tag}.json"), "w",
              encoding="utf-8") as f:
        json.dump(an, f, ensure_ascii=False, indent=2)

    rows = []
    for b, s in an["by_missing_slot_count"].items():
        rows.append({"slice": "missing_slots", "group": b, **s})
    for b, s in an["by_available_modalities"].items():
        rows.append({"slice": "available_modalities", "group": b, **s})
    cmx = np.asarray(an["confusion"]["matrix_true_by_pred"], dtype=np.int64)
    for ti, lab in enumerate(an["confusion"]["labels"]):
        p = an["confusion"]["precision"][lab]
        r = an["confusion"]["recall"][lab]
        f1 = (2 * p * r / (p + r)) if (p and r and (p + r) > 0) else float("nan")
        rows.append({"slice": "per_class", "group": lab,
                     "n": int(cmx[ti].sum()),          # 该类的真实样本数
                     "accuracy": r,                    # 逐类召回
                     "macro_f1": f1, "mae": float("nan"), "pearson": float("nan")})
    _write_csv(os.path.join(out_dir, f"error_analysis_{tag}.csv"), rows)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cm = np.array(an["confusion"]["matrix_true_by_pred"], dtype=np.float64)
        row = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(4.6, 4.0))
        im = ax.imshow(row, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(3), an["confusion"]["labels"])
        ax.set_yticks(range(3), an["confusion"]["labels"])
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"confusion matrix ({tag}, row-normalized)")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{row[i, j]:.2f}\n({int(cm[i, j])})",
                        ha="center", va="center", fontsize=8,
                        color="white" if row[i, j] > 0.55 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.savefig(os.path.join(out_dir, f"confusion_{tag}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"  （混淆矩阵图未生成：{type(e).__name__}: {e}）")


# ==================== 主流程 ====================

def _load_or_build_text_table(train_split, cache_path: str):
    """构建（或从缓存读取）词元嵌入蒸馏表。

    缓存是**加速**而非正确性来源：表只由附件2 train 决定，故落到 data/q2 下
    与模型一起提交，任何人重跑都能复现同一张表。缓存里同时存统计量，
    便于报告直接引用「覆盖了多少词元」这一事实。
    """
    if os.path.isfile(cache_path):
        z = np.load(cache_path, allow_pickle=False)
        stats = json.loads(str(z["stats"]))
        if stats.get("target_dim") == M.TEXT_EMB_DIM:
            return M.expand_token_text_table(z["ids"], z["vecs"], stats), stats
        # 维度配置变了则缓存失效，重算（表只由附件2 train 决定，重算结果一致）
    ids, vecs, stats = M.fit_token_text_vectors(train_split)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, ids=ids, vecs=vecs,
                        stats=json.dumps(stats, ensure_ascii=False))
    return M.expand_token_text_table(ids, vecs, stats), stats

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="问题二：缺失模态鲁棒情感预测")
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--structure-ablation-only", action="store_true",
                    help=("只跑结构消融：训练五个结构变体并写 structure_ablation.csv，"
                          "不训练也不改写交付模型"))
    ap.add_argument("--structure-ablation", action="store_true",
                    help=("结构消融：在同一受控缺失探针下逐一拆掉结构信号"
                          "（state_emb / obs_mask / 逐模态掩码 / 模态丢弃），"
                          "输出 structure_ablation.csv"))
    ap.add_argument("--skip-experiment", action="store_true",
                    help="跳过 48 组受控缺失实验（快速出主结果用）")
    ap.add_argument("--text-table", choices=("distill", "random", "both"),
                    default="distill",
                    help="文本支路初始化：distill=用官方 768 维蒸馏表（默认）；"
                         "random=随机初始化；both=两种都训一遍做消融对照")
    ap.add_argument("--error-analysis-only", action="store_true",
                    help="不训练：直接加载 `--out` 里已有的 q2_model.pt，"
                         "只重算并覆写错误归因产物（改过归因逻辑后用它更新，"
                         "避免重训一遍导致 q2_model.pt 与问题三产物不同源）")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cpu")
    print("=" * 72)
    print("问题二：缺失模态鲁棒情感预测")
    print("=" * 72)

    # ---- 1. 装载官方数据 ----
    a2 = Q.load_attachment2()
    tr, va, te = a2["train"], a2["valid"], a2["test"]
    print(f"附件2：train={len(tr)}  valid={len(va)}  test={len(te)}")

    y_pol_tr = Q.to_polarity(tr.reg_label)
    y_int_tr = Q.to_intensity(tr.reg_label)
    y_pol_va = Q.to_polarity(va.reg_label)
    y_int_va = Q.to_intensity(va.reg_label)
    y_pol_te = Q.to_polarity(te.reg_label)
    y_int_te = Q.to_intensity(te.reg_label)

    base = M.constant_baselines(y_int_va, y_pol_va)
    print(f"验证集平凡基线：MAE(预测训练均值)={base['mae_predict_train_mean']:.3f}  "
          f"Acc(预测多数类 {base['majority_class']})={base['accuracy_predict_majority']:.3f}")

    train_np = M.stack_split(tr)
    valid_np = M.stack_split(va)
    test_np = M.stack_split(te)

    # ---- 0. 只重算错误归因：加载既有模型，不训练 ----
    # 用途：改了归因/结论逻辑后更新产物，而**不**重训——
    # 重训会让 q2_model.pt 变成另一个实例（参数漂移约 1e-2），
    # 与已经用它生成的问题三产物不再同源。
    if args.error_analysis_only:
        ck_path = os.path.join(args.out, "q2_model.pt")
        if not os.path.isfile(ck_path):
            print(f"[错误] 找不到 {ck_path}；--error-analysis-only 需要既有模型。")
            return 2
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        # 不必传蒸馏表：文本嵌入是模型参数的一部分，load_state_dict 会整体覆盖初值；
        # 这里传 None 也得到逐位相同的模型。
        model = M.MaskedTriModalModel(ck["config"], text_emb_table=None)
        model.load_state_dict(ck["state_dict"])
        model.to(device)
        model.eval()
        print(f"[错误归因重算] 已加载 {ck_path}（不训练、不改动模型文件）")
        for name, npb, yp, yi in (("valid", valid_np, y_pol_va, y_int_va),
                                  ("test", test_np, y_pol_te, y_int_te)):
            m = evaluate(model, npb, yp, yi, device)
            ea = error_analysis(m, yp, yi, npb,
                                sample_names=list(va.ids) if name == "valid" else list(te.ids))
            write_error_analysis(ea, args.out, tag=name)
            print(f"\n[{name} 错误归因]（已覆写 error_analysis_{name}.json/.csv）")
            for c in ea["conclusions"]:
                print(f"  · {c}")
        # q2_metrics.json 里**内嵌了一份** error_analysis 副本。若只改散文件不改它，
        # 同一份结论就会在仓库里出现两个互相矛盾的版本——这种不一致最容易在
        # 评审交叉核对时被抓到。这里一并同步。
        mpath = os.path.join(args.out, "q2_metrics.json")
        if os.path.isfile(mpath):
            with open(mpath, encoding="utf-8") as f:
                mj = json.load(f)
            for name in ("valid", "test"):
                p = os.path.join(args.out, f"error_analysis_{name}.json")
                if os.path.isfile(p):
                    with open(p, encoding="utf-8") as f:
                        mj[f"error_analysis_{name}"] = json.load(f)
            with open(mpath, "w", encoding="utf-8") as f:
                json.dump(mj, f, ensure_ascii=False, indent=2)
            print(f"[同步] q2_metrics.json 内嵌的 error_analysis_* 已更新")
        print(f"\n产物目录：{args.out}")
        return 0

    # ---- 1b. 文本支路初始化：官方 768 维语义蒸馏成 64 维词元嵌入 ----
    tbl_path = os.path.join(args.out, "text_emb_table.npz")
    table, tbl_stats = _load_or_build_text_table(tr, tbl_path)
    print(f"\n[文本蒸馏表] {tbl_stats['n_distinct_ids']} 个词元 id，"
          f"覆盖词元区 {tbl_stats['coverage_of_word_tokens']:.1%} 的词元；"
          f"{tbl_stats['source_dim']}→{tbl_stats['target_dim']} 维")
    print(f"  仅由附件2 train 统计（验证/测试/附件3/附件4 未参与），缓存于 {tbl_path}")

    # ---- 1c. 只跑结构消融：五个变体各自训练，不触碰交付模型 ----
    # 用途：交付模型（q2_model.pt）已经生成问题三的全部产物，重训会得到另一个
    # 实例（参数漂移约 1e-2）而与已生成产物不再同源。结构消融只需要「同一探针下
    # 五行只差结构信号」，与交付模型无关，故单独一条路径，产物只写
    # structure_ablation.csv，其余任何文件都不改写。
    if args.structure_ablation_only:
        out_csv = os.path.join(args.out, "structure_ablation.csv")
        print("\n[结构消融] 探针 = "
              f"{STRUCTURE_PROBE[0]}缺失/{STRUCTURE_PROBE[1]}/{STRUCTURE_PROBE[2]}；"
              f"变体 = {'/'.join(M.STRUCTURE_CHOICES)}")
        print("  五个变体共用同一划分、同一种子、同一轮数与同一蒸馏初值；"
              "只写 structure_ablation.csv，不改动交付模型与既有产物。")
        struct_rows = run_structure_ablation(
            train_np, y_pol_tr, y_int_tr, valid_np, y_pol_va, y_int_va,
            test_np, y_pol_te, y_int_te, table, args, device)
        _write_csv(out_csv, struct_rows)
        ref = next(r for r in struct_rows if r["structure"] == "full")
        print(f"\n  [相对完整配置的差值（探针 valid）]  基准 macroF1="
              f"{ref['probe_valid_macro_f1']:.4f}")
        for r in struct_rows:
            if r["structure"] == "full":
                continue
            print(f"    {r['structure']:14s} ΔmacroF1="
                  f"{r['probe_valid_macro_f1']-ref['probe_valid_macro_f1']:+.4f}  "
                  f"ΔMAE={r['probe_valid_mae']-ref['probe_valid_mae']:+.4f}")
        print(f"\n产物：{out_csv}")
        return 0

    # ---- 2. 训练 ----
    print("\n[训练] 只按 valid 早停；test 全程不参与任何选择")
    table_for_run = table if args.text_table != "random" else None
    model, hist = train_model(train_np, y_pol_tr, y_int_tr,
                              valid_np, y_pol_va, y_int_va,
                              args.epochs, args.batch_size, args.lr, args.patience,
                              device, seed=args.seed, text_emb_table=table_for_run)
    metrics_extra: Dict[str, object] = {"text_emb_table": tbl_stats,
                                        "text_branch_init": args.text_table}
    if args.text_table == "both":
        print("\n[消融] 同一划分、同一随机种子，只换文本支路初始化")
        abl = []
        for name, t in (("distill", table), ("random", None)):
            ab_model, h = train_model(train_np, y_pol_tr, y_int_tr,
                                      valid_np, y_pol_va, y_int_va,
                                      args.epochs, args.batch_size, args.lr, args.patience,
                                      device, verbose=False, seed=args.seed,
                                      text_emb_table=t)
            # 消融表以**模型实际跑出的 valid 预测**为准，而不是训练日志里
            # 记录的最佳轮次指标——两者在加载最佳权重后必须一致，这里直接复核。
            mv = evaluate(ab_model, valid_np, y_pol_va, y_int_va, device)
            mt = evaluate(ab_model, test_np, y_pol_te, y_int_te, device)
            abl.append({"text_branch_init": name, "best_epoch": h["best_epoch"],
                        "valid_macro_f1": mv["macro_f1"], "valid_accuracy": mv["accuracy"],
                        "valid_mae": mv["mae"], "valid_pearson": mv["pearson"],
                        "test_macro_f1": mt["macro_f1"], "test_accuracy": mt["accuracy"],
                        "test_mae": mt["mae"], "test_pearson": mt["pearson"]})
            print(f"    {name:8s} valid: macroF1={mv['macro_f1']:.4f} Acc={mv['accuracy']:.4f} "
                  f"MAE={mv['mae']:.4f} r={mv['pearson']:.3f}   |   "
                  f"test: macroF1={mt['macro_f1']:.4f} Acc={mt['accuracy']:.4f}")
        metrics_extra["text_branch_ablation"] = abl
        _write_csv(os.path.join(args.out, "text_branch_ablation.csv"), abl)

    # ---- 3. 评测 ----
    metrics: Dict[str, object] = {"baseline_valid": base, "train_history": hist,
                                  **metrics_extra}
    for name, npb, yp, yi in (("valid", valid_np, y_pol_va, y_int_va),
                              ("test", test_np, y_pol_te, y_int_te)):
        m = evaluate(model, npb, yp, yi, device)
        metrics[name] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                         for k, v in m.items()
                         if k in ("accuracy", "macro_f1", "mae", "pearson", "n",
                                  "recall_negative", "recall_neutral", "recall_positive")}
        print(f"\n[{name}] Acc={m['accuracy']:.4f}  macroF1={m['macro_f1']:.4f}  "
              f"MAE={m['mae']:.4f}  Pearson={m['pearson']:.4f}  "
              f"(Neutral 召回={m['recall_neutral']:.3f})")
        metrics[name]["baseline"] = M.constant_baselines(yi, yp)

        # ---- 3b. 错误归因（赛题 L46：可视化分析与错误归因结论）----
        ea = error_analysis(m, yp, yi, npb, sample_names=list(va.ids) if name == "valid"
                            else list(te.ids))
        write_error_analysis(ea, args.out, tag=name)
        metrics[f"error_analysis_{name}"] = ea
        print(f"\n[{name} 错误归因]")
        for c in ea["conclusions"]:
            print(f"  · {c}")

    # ---- 4. 受控缺失实验 ----
    if not args.skip_experiment:
        print("\n[受控缺失实验] 4 类型 × 4 位置 × 3 时长 = 48 组（仅内存副本，不落盘）")
        rows = run_missingness_experiment(model, valid_np, y_pol_va, y_int_va, device,
                                          seed=args.seed)
        summ = summarize_missingness(rows)
        metrics["missingness_experiment"] = {"rows": rows, "summary": summ}
        _write_csv(os.path.join(args.out, "missingness_experiment.csv"), rows)
        with open(os.path.join(args.out, "missingness_summary.json"), "w",
                  encoding="utf-8") as f:
            json.dump(summ, f, ensure_ascii=False, indent=2)
        for k, v in summ.items():
            print(f"\n  按{k}：")
            for g, s in v.items():
                print(f"    {g:14s} macroF1={s['macro_f1']:.4f} "
                      f"Acc={s['accuracy']:.4f} MAE={s['mae']:.4f} r={s['pearson']:.3f}")
        off_target = [r for r in rows if r["missing_type"] != "none"
                      and not r["slots_cut_target_only"]]
        print(f"\n  [干预落点核验] 48 组中「屏蔽越界到非目标模态」的组数 = "
              f"{len(off_target)}（应为 0；非 0 说明掩码又退化成跨模态并集）")
        if off_target:
            print("    ⚠ 越界样例：" + json.dumps(off_target[:2], ensure_ascii=False))

    # ---- 4b. 结构消融：结构信号逐路拆解（赛题问题2「消融实验」）----
    if args.structure_ablation:
        print("\n[结构消融] 探针 = "
              f"{STRUCTURE_PROBE[0]}缺失/{STRUCTURE_PROBE[1]}/{STRUCTURE_PROBE[2]}；"
              f"变体 = {'/'.join(M.STRUCTURE_CHOICES)}")
        struct_rows = run_structure_ablation(
            train_np, y_pol_tr, y_int_tr, valid_np, y_pol_va, y_int_va,
            test_np, y_pol_te, y_int_te, table, args, device)
        _write_csv(os.path.join(args.out, "structure_ablation.csv"), struct_rows)
        ref = next(r for r in struct_rows if r["structure"] == "full")
        print(f"\n  [相对完整配置的损失（探针 valid）]  基准 macroF1="
              f"{ref['probe_valid_macro_f1']:.4f}")
        for r in struct_rows:
            if r["structure"] == "full":
                continue
            print(f"    {r['structure']:14s} ΔmacroF1="
                  f"{r['probe_valid_macro_f1']-ref['probe_valid_macro_f1']:+.4f}  "
                  f"ΔMAE={r['probe_valid_mae']-ref['probe_valid_mae']:+.4f}")
        metrics_extra["structure_ablation"] = struct_rows
        metrics["structure_ablation"] = struct_rows


    # ---- 5. 附件3 推理（赛题问题2对附件3 的交付要求）----
    print("\n[附件3 推理] 30 条缺失样本")
    a3 = Q.load_attachment3()
    a3_np = M.stack_attachment3(a3)
    pred3 = evaluate(model, a3_np, device=device)
    rows3 = []
    for i, s in enumerate(a3):
        rows3.append({
            "sample": s.key,
            "valid_len_token": s.info.valid_len,
            "modalities_present": ";".join(
                m for m in Q.MODALITIES if s.info.available[m]),
            "audio_missing_slots": s.info.missing_counts["audio"],
            "vision_missing_slots": s.info.missing_counts["vision"],
            "text_missing_slots": s.info.missing_counts["text"],
            "pred_polarity": Q.POLARITY_CN[int(pred3["polarity_idx"][i]) - 1],
            "pred_polarity_prob_neg": round(float(pred3["polarity_prob"][i, 0]), 6),
            "pred_polarity_prob_neu": round(float(pred3["polarity_prob"][i, 1]), 6),
            "pred_polarity_prob_pos": round(float(pred3["polarity_prob"][i, 2]), 6),
            "pred_intensity": round(float(pred3["intensity"][i]), 6),
            "pred_signed_value": round(
                float(Q.polarity_index_to_value(np.array([pred3["polarity_idx"][i]]))[0]
                      * pred3["intensity"][i]), 6),
        })
    _write_csv(os.path.join(args.out, "attachment3_predictions.csv"), rows3)
    dist = {Q.POLARITY_CN[c - 1]: int((pred3["polarity_idx"] == c).sum()) for c in (0, 1, 2)}
    print(f"  极性分布：{dist}")
    print(f"  强度：mean={pred3['intensity'].mean():.3f}  std={pred3['intensity'].std():.3f}  "
          f"min={pred3['intensity'].min():.3f}  max={pred3['intensity'].max():.3f}")

    # ---- 6. 缺失程度 vs 性能（用附件3 自身缺失规模分档，作为「稳定预测」的直接证据）----
    miss_total = np.array([s.info.missing_counts["audio"] + s.info.missing_counts["vision"]
                           for s in a3], dtype=np.float64)
    conf = pred3["polarity_prob"].max(axis=1)
    metrics["attachment3_robustness"] = {
        "n_samples": len(a3),
        "missing_slots_min": int(miss_total.min()),
        "missing_slots_max": int(miss_total.max()),
        "missing_slots_mean": float(miss_total.mean()),
        "corr_missing_vs_confidence": float(np.corrcoef(miss_total, conf)[0, 1])
        if miss_total.std() > 0 and conf.std() > 0 else float("nan"),
        "polarity_distribution": dist,
    }
    print(f"  缺失规模 vs 预测置信度相关系数："
          f"{metrics['attachment3_robustness']['corr_missing_vs_confidence']:.3f}")

    # ---- 7. 落盘 ----
    torch.save({"state_dict": model.state_dict(),
                "config": M.ModelConfig().__dict__,
                "seed": args.seed,
                "train_history": hist}, os.path.join(args.out, "q2_model.pt"))
    with open(os.path.join(args.out, "q2_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n参数量 {n_params:,}（fp32 ≈ {n_params*4/1024/1024:.1f} MB）")
    print(f"产物目录：{args.out}")
    return 0


def _write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
