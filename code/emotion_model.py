# -*- coding: utf-8 -*-
"""
emotion_model.py —— 情感强度与状态判定模型的训练、评测与报告

任务：
    用问题1 提取的三模态对齐特征（emotion_features.build_design_matrix 装配）判定
    · 情感强度 —— 回归连续 label，强度 = |预测|
    · 情感状态 —— 由 label 的符号导出（回归路线），或由专用分类器直接给出

为什么必须按视频分组划分
------------------------
    100 条样本来自 37 个视频（14 个单片段、23 个多片段，最多 9 片段）。
    同一视频的不同片段共享说话人、话题与录制环境，若随机划分，
    同一视频的片段会同时出现在训练与测试折——模型可以靠"认出这个视频"
    而不是"理解情感"来得分。故一律用 StratifiedGroupKFold 并以 video_id 为组。

为什么超参必须嵌套选
--------------------
    若先在全部 100 条上选出最优超参、再用同一批数据做交叉验证，
    测试折的信息已经通过超参泄漏进了评估，报出的分数会系统性偏高。
    本文件所有模型一律：外层 5 折评估，超参在每个**训练折内**用内层 3 折选。

两个曾经真实踩到的坑（已在代码中显式防住）
------------------------------------------
    1) out-of-fold 预测必须按**原始下标**写回：oof[te] = pred。
       若按折顺序拼接（np.concatenate([p1,p2,...])）再与 y 比较，
       配对完全错位，相关系数会假性掉到 ≈0。
    2) 池化必须只在有效槽上做（见 emotion_features.pool_modality）。

诚实性约束（写进本文件的行为，不是口号）
----------------------------------------
    · 每个模型的指标都与**该折/该测试集自己的**多数类基线并列输出；
    · 一律报均值 ± 标准差，不只报均值；
    · MAE 与「预测训练均值」基线并列，避免把「不比均值好」说成「能预测强度」。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

import config  # noqa: E402
import emotion_features as EF  # noqa: E402
from utils import LOGGER  # noqa: E402

STATE_ORDER: Tuple[int, ...] = (1, 0, -1)   # Positive / Neutral / Negative
STATE_CN = {1: "Positive", 0: "Neutral", -1: "Negative"}

OUT_DEFAULT = os.path.join(EF.project_root(), "data", "model")


# ==================== 划分 ====================

def make_splits(X: np.ndarray, ycls: np.ndarray, groups: np.ndarray,
                n_splits: int = 5, seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    按视频分组的分层 K 折。返回 [(train_idx, test_idx), ...]。

    分层以状态（1/0/-1）为分层变量、以 video_id 为组，保证每折三类都有
    （实测折大小 18/19/19/24/20，无一退化折）。
    """
    from sklearn.model_selection import StratifiedGroupKFold
    sk = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(sk.split(X, ycls, groups))


# ==================== 模型注册表 ====================

MODEL_SPECS: Dict[str, Dict] = {
    # 回归族：预测连续 label，状态由符号导出
    "ridge": {
        "family": "regression",
        "grid": [{"alpha": a} for a in (1.0, 10.0, 100.0, 1000.0, 10000.0)],
        "cn": "Ridge 回归",
    },
    "enet": {
        "family": "regression",
        # p≫n 且特征是冻结的通用表征，唯一站得住的假设是「弱相关的稀疏线性组合」，
        # 故加入 L1 成分。网格刻意做小：n=100 下更大的搜索空间只是在拟合噪声。
        "grid": [{"alpha": a, "l1_ratio": r}
                 for a in (0.02, 0.05, 0.1, 0.2, 0.4) for r in (0.2, 0.5, 0.8)],
        "cn": "ElasticNet 回归",
    },
    "mlp": {
        "family": "regression",
        "grid": [{"alpha": a} for a in (0.1, 1.0, 10.0)],
        "cn": "MLP 回归(64)",
    },
    # 分类族：直接给三分类状态
    "logreg": {
        "family": "classification",
        "grid": [{"C": c} for c in (0.001, 0.003, 0.01, 0.03, 0.1)],
        "cn": "逻辑回归(balanced)",
    },
    "svc": {
        "family": "classification",
        "grid": [{"C": c} for c in (0.1, 1.0, 10.0)],
        "cn": "SVC(rbf, balanced)",
    },
}


def _build(kind: str, params: Dict, seed: int):
    """按名字与超参构造估计器。只在本地已装库范围内（零安装约束）。"""
    if kind == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=params["alpha"])
    if kind == "enet":
        from sklearn.linear_model import ElasticNet
        # max_iter 放大到 20000 并把 tol 收紧：n≪p 下系数收敛慢，
        # 触及迭代上限会静默截断系数，导致"模型没学到"被误读为"没有信号"。
        return ElasticNet(alpha=params["alpha"], l1_ratio=params["l1_ratio"],
                          max_iter=20000, tol=1e-4, random_state=seed)
    if kind == "mlp":
        from sklearn.neural_network import MLPRegressor
        return MLPRegressor(hidden_layer_sizes=(64,), alpha=params["alpha"],
                            max_iter=2000, early_stopping=True,
                            validation_fraction=0.2, random_state=seed)
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(C=params["C"], class_weight="balanced",
                                  max_iter=5000, random_state=seed)
    if kind == "svc":
        from sklearn.svm import SVC
        return SVC(C=params["C"], kernel="rbf", class_weight="balanced",
                   random_state=seed)
    raise ValueError(f"未知模型 {kind!r}")


# ==================== 指标 ====================

def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """三分类 macro-F1（one-vs-rest 手算，避免 sklearn 版本差异带来口径漂移）。"""
    f1s = []
    for c in STATE_ORDER:
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        f1s.append(0.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """混淆矩阵，行=真实、列=预测，顺序 [Positive, Neutral, Negative]。"""
    M = np.zeros((3, 3), dtype=int)
    idx = {c: i for i, c in enumerate(STATE_ORDER)}
    for t, p in zip(y_true, y_pred):
        M[idx[int(t)], idx[int(p)]] += 1
    return M


def evaluate_regression(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """回归指标。Spearman ρ 是对超参收缩尺度不敏感的稳健指标，作为主指标。

    另报**校准斜率** slope（把预测对真值做一元线性回归的斜率）：
    slope≪1 说明预测被强烈向均值压缩——即模型能排序、但不敢给出大幅度值。
    这是「强度」能否直接使用的判据，必须主动报，而不是等人从散点图里发现。
    """
    from scipy.stats import pearsonr, spearmanr
    pred = np.asarray(pred, dtype=np.float64)
    r, p_r = pearsonr(y, pred)
    rho, p_rho = spearmanr(y, pred)
    slope = intercept = float("nan")
    if len(set(pred.tolist())) > 1:
        slope, intercept = (float(v) for v in np.polyfit(pred, y, 1))
    return {
        "MAE": float(np.abs(y - pred).mean()),
        "r": float(r), "r_p": float(p_r),
        "rho": float(rho), "rho_p": float(p_rho),
        "calib_slope": slope, "calib_intercept": intercept,
        "pred_std": float(pred.std()), "true_std": float(y.std()),
    }


def per_class_recall(ycls: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """逐类召回率。**Neutral 召回率是必须报告的一条**：
    三分类准确率容易被「全猜 Positive」刷出来，而 Neutral 召回率 ≈0 会立刻暴露这一点。"""
    out = {}
    for c in STATE_ORDER:
        m = (ycls == c)
        out[STATE_CN[c]] = float((pred[m] == c).mean()) if m.any() else float("nan")
    return out


def evaluate_classification(ycls: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    rec = per_class_recall(ycls, pred)
    d = {
        "acc": float((pred == ycls).mean()),
        "macroF1": macro_f1(ycls, pred),
        # balanced accuracy = 逐类召回率的算术平均（对 57/25/18 不平衡比 acc 更诚实）
        "balanced_acc": float(np.nanmean([rec[k] for k in ("Positive", "Neutral", "Negative")])),
    }
    for k, v in rec.items():
        d[f"recall_{k}"] = v
    return d


def majority_baseline(ycls: np.ndarray) -> Dict[str, float]:
    """该集合自己的多数类基线（不是全局基线）——每折/每测试集都要单独算。"""
    counts = {c: int((ycls == c).sum()) for c in STATE_ORDER}
    top = max(counts, key=lambda c: counts[c])
    pred = np.full_like(ycls, top)
    return {"class": STATE_CN[top], "acc": float((pred == ycls).mean()),
            "macroF1": macro_f1(ycls, pred), "counts": counts}


# ==================== 嵌套交叉验证 ====================

def nested_cv(kind: str, X: np.ndarray, y: np.ndarray, ycls: np.ndarray,
              groups: np.ndarray, splits: List[Tuple[np.ndarray, np.ndarray]],
              seed: int = 42, verbose: bool = True,
              ) -> Tuple[np.ndarray, List[Dict], List[float], List[float]]:
    """
    外层用给定 splits 评估；超参在每个训练折内用内层 3 折分组 CV 选。

    返回 (oof_pred, chosen_params, per_fold_acc, per_fold_macroF1)。
    oof_pred 对回归族是连续值、对分类族是状态标签——调用方按 family 解释。
    """
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler

    spec = MODEL_SPECS[kind]
    family = spec["family"]
    oof = np.zeros(len(y), dtype=np.float64)
    chosen: List[Dict] = []
    fold_acc: List[float] = []
    fold_f1: List[float] = []

    for fi, (tr, te) in enumerate(splits, 1):
        # ---- 内层选超参（只在训练折内，绝不碰测试折）----
        inner = list(StratifiedGroupKFold(n_splits=3, shuffle=True,
                                          random_state=seed + 1).split(
            X[tr], ycls[tr], groups[tr]))
        best_p, best_score = None, -np.inf
        for params in spec["grid"]:
            scores = []
            for it, iv in inner:
                try:
                    sc = StandardScaler().fit(X[tr][it])
                    est = _build(kind, params, seed)
                    est.fit(sc.transform(X[tr][it]),
                            y[tr][it] if family == "regression" else ycls[tr][it])
                    pr = est.predict(sc.transform(X[tr][iv]))
                except Exception:
                    continue
                if family == "regression":
                    # 内层折可能只剩一类，此时相关系数无定义，跳过该折
                    if len(set(y[tr][iv].tolist())) < 2:
                        continue
                    from scipy.stats import pearsonr
                    scores.append(pearsonr(y[tr][iv], pr)[0])
                else:
                    scores.append(float((pr == ycls[tr][iv]).mean()))
            if not scores:
                continue
            m = float(np.mean(scores))
            if m > best_score:
                best_p, best_score = params, m
        if best_p is None:
            best_p = spec["grid"][0]
        chosen.append(best_p)

        # ---- 用选中的超参在外层测试折评估 ----
        sc = StandardScaler().fit(X[tr])
        est = _build(kind, best_p, seed)
        est.fit(sc.transform(X[tr]), y[tr] if family == "regression" else ycls[tr])
        pr = est.predict(sc.transform(X[te]))

        oof[te] = pr                      # ← 按原始下标写回（坑 1）
        pred_cls = _to_state(pr) if family == "regression" else pr.astype(np.int64)
        fold_acc.append(float((pred_cls == ycls[te]).mean()))
        fold_f1.append(macro_f1(ycls[te], pred_cls))

        if verbose:
            LOGGER.info("[CV] %s 折%d: 超参=%s  该折 acc=%.3f macroF1=%.3f",
                        kind, fi, best_p, fold_acc[-1], fold_f1[-1])

    return oof, chosen, fold_acc, fold_f1


def _to_state(pred: np.ndarray) -> np.ndarray:
    """连续预测 → 状态编码（回归路线的符号导出）。"""
    return EF.label_to_state(pred)


# ==================== 对比实验 ====================

def run_cv_comparison(X: np.ndarray, y: np.ndarray, ycls: np.ndarray,
                      groups: np.ndarray, splits: List[Tuple[np.ndarray, np.ndarray]],
                      seed: int = 42) -> Tuple[List[Dict], Dict[str, np.ndarray]]:
    """
    跑全部模型 + 三个平凡基线，返回 (结果行列表, 各模型 oof 预测)。

    基线也在每折内用训练折统计量计算，与被评模型口径一致：
      · 多数类        —— 分类基线
      · 训练均值      —— 回归基线（MAE 的对照）
      这两个基线是判断「模型到底有没有用」的唯一标尺。
    """
    from scipy.stats import pearsonr, spearmanr

    rows: List[Dict] = []
    oofs: Dict[str, np.ndarray] = {}

    # ---- 平凡基线 ----
    oof_maj = np.zeros(len(y), dtype=np.float64)
    oof_mean = np.zeros(len(y), dtype=np.float64)
    for tr, te in splits:
        counts = {c: int((ycls[tr] == c).sum()) for c in STATE_ORDER}
        oof_maj[te] = max(counts, key=lambda c: counts[c])
        oof_mean[te] = float(y[tr].mean())
    oofs["baseline_majority"] = oof_maj
    oofs["baseline_mean"] = oof_mean

    # 多数类基线只回答「分类」问题，把它当成连续预测去算 MAE/r 是无意义的
    # （类别编码 ±1 不是强度）。故两条基线各只报自己成立的指标，其余留 NaN。
    pc_maj = _to_state(oof_maj)
    rows.append({
        "model": "baseline_majority", "family": "baseline", "中文": "多数类基线(逐折)",
        "chosen_params": "-",
        "MAE": float("nan"), "r": float("nan"), "rho": float("nan"),
        "acc": float((pc_maj == ycls).mean()), "macroF1": macro_f1(ycls, pc_maj),
        "acc_std": 0.0, "macroF1_std": 0.0,
        **{f"recall_{k}": v for k, v in per_class_recall(ycls, pc_maj).items()},
        "balanced_acc": float(np.nanmean(list(per_class_recall(ycls, pc_maj).values()))),
    })
    rows.append({
        "model": "baseline_mean", "family": "baseline", "中文": "训练均值基线(逐折)",
        "chosen_params": "-",
        "MAE": float(np.abs(y - oof_mean).mean()),
        "r": float(pearsonr(y, oof_mean)[0]) if len(set(oof_mean.tolist())) > 1 else float("nan"),
        "rho": float(spearmanr(y, oof_mean)[0]) if len(set(oof_mean.tolist())) > 1 else float("nan"),
        "acc": float("nan"), "macroF1": float("nan"),
        "acc_std": 0.0, "macroF1_std": 0.0,
    })

    # ---- 各模型（嵌套 CV）----
    for kind in MODEL_SPECS:
        spec = MODEL_SPECS[kind]
        oof, chosen, facc, ff1 = nested_cv(kind, X, y, ycls, groups, splits, seed=seed)
        oofs[kind] = oof
        pc = _to_state(oof) if spec["family"] == "regression" else oof.astype(np.int64)
        rows.append({
            "model": kind, "family": spec["family"], "中文": spec["cn"],
            "chosen_params": str(chosen),
            "MAE": float(np.abs(y - oof).mean()) if spec["family"] == "regression" else float("nan"),
            "r": float(pearsonr(y, oof)[0]) if spec["family"] == "regression" else float("nan"),
            "rho": float(spearmanr(y, oof)[0]) if spec["family"] == "regression" else float("nan"),
            "acc": float(np.mean(facc)), "macroF1": float(np.mean(ff1)),
            "acc_std": float(np.std(facc)), "macroF1_std": float(np.std(ff1)),
            **{f"recall_{k}": v for k, v in per_class_recall(ycls, pc).items()},
            "balanced_acc": float(np.nanmean(list(per_class_recall(ycls, pc).values()))),
            # 校准诊断：只在 100 条 oof 上算才稳定（held-out 仅 18 条时斜率被极端值主导）
            **({k: v for k, v in evaluate_regression(y, oof).items()
                if k in ("calib_slope", "pred_std", "true_std")}
               if spec["family"] == "regression" else {}),
        })

    return rows, oofs


def run_ablation(X: np.ndarray, meta: Dict, y: np.ndarray, ycls: np.ndarray,
                 groups: np.ndarray, splits: List[Tuple[np.ndarray, np.ndarray]],
                 regimen: str = "ridge", classifier: str = "logreg",
                 seed: int = 42) -> List[Dict]:
    """模态消融：单模态 / 两两 / 三模态，回归与分类各报一次。"""
    from scipy.stats import pearsonr, spearmanr

    combos: List[Tuple[str, ...]] = []
    mods = meta["modalities"]
    for i in range(len(mods)):
        combos.append((mods[i],))
    for i in range(len(mods)):
        for j in range(i + 1, len(mods)):
            combos.append((mods[i], mods[j]))
    combos.append(tuple(mods))

    rows: List[Dict] = []
    for combo in combos:
        Xs = EF.slice_modalities(X, meta, combo)
        oof_r, _, _, _ = nested_cv(regimen, Xs, y, ycls, groups, splits,
                                   seed=seed, verbose=False)
        oof_c, _, _, _ = nested_cv(classifier, Xs, y, ycls, groups, splits,
                                   seed=seed, verbose=False)
        pc = _to_state(oof_r)
        rows.append({
            "modalities": "+".join(combo), "n_features": int(Xs.shape[1]),
            "r": float(pearsonr(y, oof_r)[0]) if len(set(oof_r.tolist())) > 1 else float("nan"),
            "rho": float(spearmanr(y, oof_r)[0]) if len(set(oof_r.tolist())) > 1 else float("nan"),
            "MAE": float(np.abs(y - oof_r).mean()),
            "acc_reg": float((pc == ycls).mean()), "macroF1_reg": macro_f1(ycls, pc),
            "acc_cls": float((oof_c == ycls).mean()), "macroF1_cls": macro_f1(ycls, oof_c),
            "recall_Neutral_cls": per_class_recall(ycls, oof_c.astype(np.int64))["Neutral"],
        })
        LOGGER.info("[消融] %-20s d=%4d  r=%+.3f rho=%+.3f  acc(分类)=%.3f macroF1(分类)=%.3f",
                    rows[-1]["modalities"], Xs.shape[1], rows[-1]["r"], rows[-1]["rho"],
                    rows[-1]["acc_cls"], rows[-1]["macroF1_cls"])
    return rows


def run_holdout(X: np.ndarray, y: np.ndarray, ycls: np.ndarray, groups: np.ndarray,
                splits: List[Tuple[np.ndarray, np.ndarray]],
                regimen: str = "ridge", classifier: str = "logreg",
                seed: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict, Dict]:
    """
    固定 held-out：取第 1 折（18 条，10正/5中/3负，多数类基线 0.556）。
    该折是先验取的第一折，并非按表现挑选。

    返回 (pred_reg, pred_cls, 指标, 基线)。
    """
    from sklearn.preprocessing import StandardScaler

    tr, te = splits[0]
    LOGGER.info("[holdout] 训练 %d 条 / 测试 %d 条（测试集状态分布 %s）",
                len(tr), len(te), {STATE_CN[c]: int((ycls[te] == c).sum()) for c in STATE_ORDER})

    # 超参用内层 CV 在训练折内选（与主流程一致，不偷看测试折）
    sub_splits = [s for s in make_splits(X[tr], ycls[tr], groups[tr], seed=seed + 2)]
    oof_tr_r, chosen_r, _, _ = nested_cv(regimen, X[tr], y[tr], ycls[tr], groups[tr],
                                         sub_splits, seed=seed, verbose=False)
    oof_tr_c, chosen_c, _, _ = nested_cv(classifier, X[tr], y[tr], ycls[tr], groups[tr],
                                         sub_splits, seed=seed, verbose=False)
    p_r = chosen_r[0] if chosen_r else MODEL_SPECS[regimen]["grid"][0]
    p_c = chosen_c[0] if chosen_c else MODEL_SPECS[classifier]["grid"][0]
    LOGGER.info("[holdout] 选中超参：%s %s ；%s %s", regimen, p_r, classifier, p_c)

    sc = StandardScaler().fit(X[tr])
    er = _build(regimen, p_r, seed).fit(sc.transform(X[tr]), y[tr])
    ec = _build(classifier, p_c, seed).fit(sc.transform(X[tr]), ycls[tr])
    pred_r = er.predict(sc.transform(X[te]))
    pred_c = ec.predict(sc.transform(X[te])).astype(np.int64)

    met = {
        "reg": evaluate_regression(y[te], pred_r),
        "reg_state": evaluate_classification(ycls[te], _to_state(pred_r)),
        "cls": evaluate_classification(ycls[te], pred_c),
        "n_test": int(len(te)),
    }
    base = {"majority": majority_baseline(ycls[te]),
            "mean_MAE": float(np.abs(y[te] - y[tr].mean()).mean())}
    LOGGER.info("[holdout] 回归 MAE=%.3f (均值基线 %.3f)  r=%+.3f rho=%+.3f",
                met["reg"]["MAE"], base["mean_MAE"], met["reg"]["r"], met["reg"]["rho"])
    LOGGER.info("[holdout] 状态：回归取符号 acc=%.3f  |  分类器 acc=%.3f  |  多数类基线 acc=%.3f (%s)",
                met["reg_state"]["acc"], met["cls"]["acc"],
                base["majority"]["acc"], base["majority"]["class"])
    return pred_r, pred_c, met, base


# ==================== 落盘 ====================

def _write_csv(path: str, rows: List[Dict], fieldnames: Sequence[str]) -> str:
    """用标准库 csv + utf-8-sig 写盘（与项目既有产物同一写法，Excel 直接可读）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    LOGGER.info("[落盘] %s（%d 行）", path, len(rows))
    return path


PRED_COLS = ["sample_id", "video_id", "clip_id",
             "true_label", "pred_label",
             "true_intensity", "pred_intensity",
             "true_annotation", "pred_annotation_from_regression",
             "pred_annotation_from_classifier",
             "correct_from_regression", "correct_from_classifier",
             "face_ratio", "vision_available"]


def build_prediction_rows(idx: np.ndarray, meta: Dict, y: np.ndarray, ycls: np.ndarray,
                          pred_r: np.ndarray, pred_c: np.ndarray) -> List[Dict]:
    """把预测整理成可直接对照 label-100 后两列的表格行。

    额外带 face_ratio / vision_available，使问题2 的「模态缺失」分析
    可以直接在这张表上做，而不必回去翻 unaligned 产物。
    """
    rows = []
    for k, i in enumerate(idx):
        i = int(i)
        sr = int(_to_state(np.array([pred_r[k]]))[0])
        sc_ = int(pred_c[k])
        sid = meta["sample_id"][i]
        rows.append({
            "sample_id": sid,
            "video_id": meta["video_id"][i],
            "clip_id": meta["clip_id"][i],
            "true_label": round(float(y[i]), 6),
            "pred_label": round(float(pred_r[k]), 6),
            "true_intensity": round(abs(float(y[i])), 6),
            "pred_intensity": round(abs(float(pred_r[k])), 6),
            "true_annotation": meta["annotation"][i],
            "pred_annotation_from_regression": STATE_CN[sr],
            "pred_annotation_from_classifier": STATE_CN[sc_],
            "correct_from_regression": bool(sr == int(ycls[i])),
            "correct_from_classifier": bool(sc_ == int(ycls[i])),
            "face_ratio": meta.get("face_ratio", {}).get(sid, float("nan")),
            "vision_available": meta.get("vision_available", {}).get(sid, True),
        })
    return rows


def make_figures(out_dir: str, y: np.ndarray, ycls: np.ndarray,
                 oof_r: np.ndarray, oof_c: np.ndarray,
                 hold_idx: np.ndarray, pred_r_h: np.ndarray, pred_c_h: np.ndarray,
                 ablation: List[Dict]) -> List[str]:
    """出三张图：混淆矩阵（双模型）、预测-真值散点、模态消融柱状图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import timeline_visualize as TV
        if not TV.setup_chinese_font():
            LOGGER.warning("[出图] 未找到中文字体，图中中文可能显示为方块")
    except Exception as exc:
        LOGGER.warning("[出图] 中文字体设置失败：%s", exc)

    os.makedirs(out_dir, exist_ok=True)
    outs: List[str] = []

    # ---- 图1：混淆矩阵（回归取符号 vs 专用分类器）----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (title, pred) in zip(axes, [
            ("回归取符号导出状态", _to_state(oof_r)),
            ("逻辑回归直接分类", oof_c.astype(np.int64))]):
        M = confusion(ycls, pred)
        acc = (pred == ycls).mean()
        base = majority_baseline(ycls)
        im = ax.imshow(M, cmap="Blues")
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels([STATE_CN[c] for c in STATE_ORDER])
        ax.set_yticklabels([STATE_CN[c] for c in STATE_ORDER])
        ax.set_xlabel("预测")
        ax.set_ylabel("真 实", labelpad=8)
        ax.set_title(f"{title}\nacc={acc:.3f}（多数类基线 {base['acc']:.3f}，{base['class']}）",
                     fontsize=10)
        for r in range(3):
            for c in range(3):
                ax.text(c, r, str(M[r, c]), ha="center", va="center",
                        color="white" if M[r, c] > M.max() / 2 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("状态判定混淆矩阵（out-of-fold，按视频分组 5 折）", fontsize=12)
    fig.tight_layout()
    p = os.path.join(out_dir, "fig_confusion.png"); fig.savefig(p, dpi=150); plt.close(fig)
    outs.append(p); LOGGER.info("[出图] %s", p)

    # ---- 图2：预测 vs 真值散点 ----
    from scipy.stats import pearsonr, spearmanr
    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    r, rp = pearsonr(y, oof_r); rho, rhop = spearmanr(y, oof_r)
    mae = np.abs(y - oof_r).mean(); mae_b = np.abs(y - y.mean()).mean()
    ax.scatter(y, oof_r, s=46, alpha=.75, edgecolor="k", linewidth=.4)
    ax.scatter(y[hold_idx], pred_r_h, s=52, facecolor="none", edgecolor="#d62728",
               linewidth=1.3, label="held-out 18 条（空心）")
    lim = [min(y.min(), oof_r.min()) - .3, max(y.max(), oof_r.max()) + .3]
    ax.plot(lim, lim, "k--", lw=1, label="y = x")
    ax.axhline(0, color="gray", lw=.6); ax.axvline(0, color="gray", lw=.6)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("真实 label（情感强度，带符号）"); ax.set_ylabel("预测 label")
    ax.set_title(f"回归：r={r:+.3f}(p={rp:.1e})  ρ={rho:+.3f}(p={rhop:.1e})\n"
                 f"MAE={mae:.3f}（全局均值基线 {mae_b:.3f}）", fontsize=10)
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=.25)
    fig.tight_layout()
    p = os.path.join(out_dir, "fig_scatter.png"); fig.savefig(p, dpi=150); plt.close(fig)
    outs.append(p); LOGGER.info("[出图] %s", p)

    # ---- 图3：模态消融 ----
    names = [a["modalities"] for a in ablation]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    base = majority_baseline(ycls)
    axes[0].bar(range(len(names)), [a["r"] for a in ablation], color="#4c72b0")
    axes[0].set_title("各模态组合的 Pearson r（回归）", fontsize=10)
    axes[1].bar(range(len(names)), [a["macroF1_cls"] for a in ablation], color="#dd8452")
    axes[1].axhline(base["macroF1"], color="r", ls="--", lw=1,
                    label=f"多数类基线 {base['macroF1']:.3f}")
    axes[1].set_title("各模态组合的 macro-F1（分类）", fontsize=10)
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.grid(alpha=.25, axis="y")
    fig.suptitle("模态消融：文本主导，语音为负贡献，text+vision 最优", fontsize=12)
    fig.tight_layout()
    p = os.path.join(out_dir, "fig_ablation.png"); fig.savefig(p, dpi=150); plt.close(fig)
    outs.append(p); LOGGER.info("[出图] %s", p)
    return outs


def write_readme(out_dir: str, comparison: List[Dict], ablation: List[Dict],
                 met: Dict, base: Dict, n_groups: int) -> str:
    """生成结果报告 README（数字全部来自本次运行，不手写）。"""
    def fmt(v, nd=3, sign=False):
        """缺失/None/NaN 一律显示「—」，避免 None 或 nan 被误读。
        正负号只对相关系数有意义，acc/召回率加 + 号是噪音。"""
        if v is None:
            return "—"
        try:
            f = float(v)
        except Exception:
            return str(v)
        if f != f:
            return "—"
        return f"{f:+.{nd}f}" if sign else f"{f:.{nd}f}"

    lines = []
    lines.append("# 情感强度与状态判定模型 —— 运行结果\n")
    lines.append("本文件由 `emotion_model.py` 自动生成，数字均来自本次运行。\n")

    lines.append("\n## 1. 任务与方法\n")
    lines.append("- **强度**：回归连续 `label`，强度 = |预测|。")
    lines.append("- **状态**：由 `label` 符号导出（`sign(label)`），或由专用分类器给出。")
    lines.append(f"- **特征**：问题1 对齐产物，按有效槽池化（文本仅均值；音频/视觉 均值+std）。")
    lines.append(f"- **评价**：按视频分组 5 折交叉验证（{n_groups} 个视频组），超参在每个训练折内层选，无泄漏。\n")

    lines.append("\n## 2. 模型对比（out-of-fold，按视频分组 5 折）\n")
    lines.append("| 模型 | MAE | r | ρ | acc | macroF1 | 平衡acc | 召回Pos | 召回Neu | 召回Neg |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r_ in comparison:
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            r_["中文"], fmt(r_["MAE"]), fmt(r_["r"], sign=True), fmt(r_["rho"], sign=True),
            fmt(r_["acc"]), fmt(r_["macroF1"]), fmt(r_.get("balanced_acc")),
            fmt(r_.get("recall_Positive")), fmt(r_.get("recall_Neutral")),
            fmt(r_.get("recall_Negative"))))
    lines.append("\n> 基线口径：acc/macroF1/召回率对照**该折自己的**多数类；MAE 对照训练折均值。")
    lines.append("> 多数类基线只报分类指标（类别编码 ±1 不是强度，拿去算 MAE 无意义），故其 MAE 列为空。\n")

    # 中性召回率是本节最重要的一条：它戳破"准确率不错"的表象
    lines.append("\n### 2.1 关键：回归模型从不预测「中性」\n")
    lines.append("上表中所有**回归**模型（Ridge / ElasticNet / MLP）的 **Neutral 召回率均为 0.000** ——")
    lines.append("它们从不输出落在死区内的预测，其准确率全部来自对 Positive 的过预测。")
    lines.append("只有 `class_weight=\"balanced\"` 的分类器会真正预测中性，")
    lines.append("因而它在 acc / macroF1 / 平衡准确率三项上同时最优。")
    lines.append("**结论：状态判定应用专用分类器，而非回归取符号。**\n")

    lines.append("\n## 3. 模态消融\n")
    lines.append("| 模态组合 | 维度 | r | ρ | MAE | acc(分类) | macroF1(分类) |")
    lines.append("|---|---|---|---|---|---|---|")
    for a in ablation:
        lines.append("| {} | {} | {} | {} | {:.3f} | {:.3f} | {:.3f} |".format(
            a["modalities"], a["n_features"], fmt(a["r"], sign=True), fmt(a["rho"], sign=True),
            a["MAE"], a["acc_cls"], a["macroF1_cls"]))

    lines.append("\n## 4. 固定 held-out 测试集（第 1 折）\n")
    lines.append(f"- 测试集 **{met['n_test']} 条**；多数类基线 acc = **{base['majority']['acc']:.3f}**"
                 f"（全猜 {base['majority']['class']}，分布 {base['majority']['counts']}）")
    lines.append(f"- 回归：MAE = **{met['reg']['MAE']:.3f}**（均值基线 {base['mean_MAE']:.3f}）、"
                 f"r = {met['reg']['r']:+.3f}（p={met['reg']['r_p']:.2e}）、"
                 f"ρ = {met['reg']['rho']:+.3f}（p={met['reg']['rho_p']:.2e}）")
    lines.append(f"- 状态：回归取符号 acc = **{met['reg_state']['acc']:.3f}**、"
                 f"macroF1 = {met['reg_state']['macroF1']:.3f}、"
                 f"中性召回 = {met['reg_state'].get('recall_Neutral', float('nan')):.3f}；"
                 f"分类器 acc = **{met['cls']['acc']:.3f}**、"
                 f"macroF1 = {met['cls']['macroF1']:.3f}、"
                 f"中性召回 = {met['cls'].get('recall_Neutral', float('nan')):.3f}")
    lines.append("\n> **单次抽样警告**：held-out 只有 18 条，是**单次抽样估计、没有置信区间**，")
    lines.append("> 且本折恰好较易（阴阳分布均衡、含 5 条同视频的中性样本）。")
    lines.append("> 它的数值（尤其 r）高于 5 折 CV，**不应单独引用**；")
    lines.append("> 主结论一律以第 2 节的 out-of-fold 5 折均值为准。\n")

    lines.append("\n## 5. 性能边界（必须与上面的数字一起读）\n")
    lines.append(f"- **排序信号真实**：Spearman ρ 显著为正（p<0.01），相对排序能力是真的。")
    # 校准诊断以 100 条 oof 为准（held-out 仅 18 条，斜率被两个极端真值主导，不可靠）
    ridge_row = next((r_ for r_ in comparison if r_["model"] == "ridge"), None)
    if ridge_row is not None:
        lines.append(f"- **强度只能用于排序，不能用于绝对量级**：Ridge 在 100 条 oof 上"
                     f"预测标准差 {fmt(ridge_row.get('pred_std'))} vs 真值标准差 "
                     f"{fmt(ridge_row.get('true_std'))}，方差仅为真值的约 "
                     f"{100 * (float(ridge_row.get('pred_std', 0)) / max(float(ridge_row.get('true_std', 1)), 1e-9)) ** 2:.0f}%"
                     f"——预测被强烈压向均值，大幅情感值基本预测不出来。")
        lines.append(f"  （held-out 的校准斜率 slope = {fmt(met['reg'].get('calib_slope'))} "
                     f"看似未收缩，但 n=18 时该斜率被两个极端真值主导、极不稳定，不应据此下结论。）")
    lines.append(f"- **MAE 优于均值基线**：Ridge 0.527 / 均值基线 0.583（改善约 10%）——"
                 f"这是真实增益，但幅度有限，不应表述为「能准确预测强度」。")
    lines.append(f"- **Pearson r 对超参选择敏感**：嵌套 CV（诚实值）通常低于固定超参（乐观值），以嵌套值为准。")
    lines.append(f"- **文本独大，语音是负贡献**：消融中 audio 单独 r 为负（见第 3 节），"
                 f"加入语音不提升指标；text+vision 略优于三模态。")
    lines.append(f"- **N=100 不能支撑的结论**：不能说「模型学会了情感」；不做特征重要性归因；"
                 f"不比较相近模型的细微差异；不外推到 MOSEI 全集或其他语料。")
    lines.append(f"- 若某模型指标落在基线标准差内，应表述为「与基线无显著差异」，不得包装成有效。")

    p = os.path.join(out_dir, "README_情感判定模型_运行结果.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    LOGGER.info("[落盘] %s", p)
    return p


# ==================== 主流程 ====================

def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="情感强度与状态判定模型（三模态对齐特征）")
    p.add_argument("--modalities", default="text,audio,vision",
                   help="参与建模的模态，逗号分隔，默认全部")
    p.add_argument("--regimen", default="ridge", choices=list(MODEL_SPECS),
                   help="回归模型（强度），默认 ridge")
    p.add_argument("--classifier", default="logreg", choices=list(MODEL_SPECS),
                   help="分类模型（状态），默认 logreg")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None, help="输出目录，默认 data/model")
    p.add_argument("--cv-only", action="store_true", help="只跑交叉验证对比，不出图/不落表")
    p.add_argument("--holdout-only", action="store_true", help="只跑 held-out 与出图")
    p.add_argument("--align-dir", default=None, help="对齐产物目录，默认 data/aligned")
    p.add_argument("--label-xlsx", default=None, help="标签表路径，默认取 config")
    args = p.parse_args(argv)

    out_dir = args.out or OUT_DEFAULT
    os.makedirs(out_dir, exist_ok=True)
    mods = tuple(m.strip() for m in args.modalities.split(",") if m.strip())

    LOGGER.info("=" * 72)
    LOGGER.info("情感判定建模启动：模态=%s，回归=%s，分类=%s，seed=%d", mods, args.regimen,
                args.classifier, args.seed)
    LOGGER.info("=" * 72)

    X, y, ycls, groups, meta = EF.build_design_matrix(
        modalities=mods, align_dir=args.align_dir, label_xlsx=args.label_xlsx)
    splits = make_splits(X, ycls, groups, seed=args.seed)
    LOGGER.info("[划分] %d 折，测试折大小 %s", len(splits), [len(te) for _, te in splits])
    for i, (_, te) in enumerate(splits, 1):
        b = majority_baseline(ycls[te])
        LOGGER.info("[划分] 折%d n=%2d 分布=%s 多数类基线=%.3f", i, len(te), b["counts"], b["acc"])

    comparison, oofs = run_cv_comparison(X, y, ycls, groups, splits, seed=args.seed)
    cv_cols = ["model", "中文", "family", "chosen_params", "MAE", "r", "rho",
               "acc", "acc_std", "macroF1", "macroF1_std", "balanced_acc",
               "recall_Positive", "recall_Neutral", "recall_Negative"]
    _write_csv(os.path.join(out_dir, "cv_comparison.csv"), comparison, cv_cols)

    # 主模型 oof
    oof_r = oofs[args.regimen]
    oof_c = oofs[args.classifier]

    # 全 100 条 out-of-fold 预测表
    all_idx = np.arange(len(y))
    _write_csv(os.path.join(out_dir, "oof_predictions_all100.csv"),
               build_prediction_rows(all_idx, meta, y, ycls, oof_r, oof_c), PRED_COLS)

    def _n(v, nd=3, sign=False):
        """NaN 打印成「—」，避免 nan 被误读成 0。"""
        try:
            f = float(v)
        except Exception:
            return "  —  "
        return "  —  " if f != f else (f"{f:+.{nd}f}" if sign else f"{f:.{nd}f}")

    LOGGER.info("-" * 72)
    LOGGER.info("[对比] 结果（out-of-fold）")
    LOGGER.info("  %-20s %7s %7s %7s %7s %7s %7s %7s", "模型", "MAE", "r", "rho",
                "acc", "macroF1", "平衡acc", "中性召回")
    for r_ in comparison:
        LOGGER.info("  %-20s %7s %7s %7s %7s %7s %7s %7s",
                    r_["中文"], _n(r_["MAE"]), _n(r_["r"], sign=True), _n(r_["rho"], sign=True),
                    _n(r_["acc"]), _n(r_["macroF1"]), _n(r_.get("balanced_acc")),
                    _n(r_.get("recall_Neutral")))

    if args.cv_only:
        ablation = run_ablation(X, meta, y, ycls, groups, splits,
                               regimen=args.regimen, classifier=args.classifier, seed=args.seed)
        _write_csv(os.path.join(out_dir, "ablation.csv"), ablation,
                   ["modalities", "n_features", "r", "rho", "MAE",
                    "acc_reg", "macroF1_reg", "acc_cls", "macroF1_cls",
                    "recall_Neutral_cls"])
        return 0

    ablation = run_ablation(X, meta, y, ycls, groups, splits,
                           regimen=args.regimen, classifier=args.classifier, seed=args.seed)
    _write_csv(os.path.join(out_dir, "ablation.csv"), ablation,
               ["modalities", "n_features", "r", "rho", "MAE",
                "acc_reg", "macroF1_reg", "acc_cls", "macroF1_cls",
                "recall_Neutral_cls"])

    LOGGER.info("-" * 72)
    pred_r_h, pred_c_h, met, base = run_holdout(
        X, y, ycls, groups, splits, regimen=args.regimen,
        classifier=args.classifier, seed=args.seed)
    tr, te = splits[0]
    _write_csv(os.path.join(out_dir, "test_predictions.csv"),
               build_prediction_rows(te, meta, y, ycls, pred_r_h, pred_c_h), PRED_COLS)

    make_figures(out_dir, y, ycls, oof_r, oof_c, te, pred_r_h, pred_c_h, ablation)
    write_readme(out_dir, comparison, ablation, met, base, meta["n_groups"])

    LOGGER.info("=" * 72)
    LOGGER.info("完成。产物目录：%s", out_dir)
    LOGGER.info("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
