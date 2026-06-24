"""分割指标：Precision/Recall/F1/IoU/ODS + clDice。

包含两种评估协议：
  • strict   ：裸像素匹配 (TP/FP/FN by exact intersection)
  • tolerant ：DeepCrack/CrackForest 标准协议
                — 预测做 skeleton 细化（or 不细化）
                — 匹配时允许 ±tol 像素 (默认 2 px)
                — 论文中常用，能消除"边界 1-2 像素偏差"导致的虚假误差
"""
from __future__ import annotations
import numpy as np
import torch
import cv2
from skimage.morphology import skeletonize


def _to_np(t):
    return t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)


# ---------------- clDice ----------------
def cl_dice(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred.astype(bool); target = target.astype(bool)
    if pred.sum() == 0 or target.sum() == 0:
        return 0.0
    sk_p = skeletonize(pred); sk_t = skeletonize(target)
    tprec = (sk_p & target).sum() / (sk_p.sum() + eps)
    tsens = (sk_t & pred).sum() / (sk_t.sum() + eps)
    return float(2 * tprec * tsens / (tprec + tsens + eps))


# ---------------- tolerant matching ----------------
def _dilate(mask01: np.ndarray, radius: int) -> np.ndarray:
    """Square 邻域膨胀，3x3 kernel 迭代 radius 次。"""
    if radius <= 0:
        return mask01.astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    return cv2.dilate(mask01.astype(np.uint8), k, iterations=radius)


def _tp_fp_fn_tolerant(pred01: np.ndarray, gt01: np.ndarray, tol: int):
    """以 ±tol 像素容差计算 TP/FP/FN —— DeepCrack 协议。
       TP = 预测里被 GT 容差区域覆盖的像素
       FN = GT 里被 预测 容差区域覆盖之外的像素
       FP = 预测里不在 GT 容差区域内的像素
    """
    if tol <= 0:
        tp = int(((pred01 == 1) & (gt01 == 1)).sum())
        fp = int(((pred01 == 1) & (gt01 == 0)).sum())
        fn = int(((pred01 == 0) & (gt01 == 1)).sum())
        return tp, fp, fn

    gt_dil   = _dilate(gt01,   tol)
    pred_dil = _dilate(pred01, tol)
    tp = int(((pred01 == 1) & (gt_dil   == 1)).sum())
    fp = int(((pred01 == 1) & (gt_dil   == 0)).sum())
    fn = int(((gt01   == 1) & (pred_dil == 0)).sum())
    return tp, fp, fn


# ---------------- main metric ----------------
class SegMetric:
    """
    支持：
      • thresholds       : 阈值扫描列表
      • tolerance        : ±N pixel 容差（DeepCrack/CrackForest 标准 = 2）
      • thin_prediction  : 是否先把预测 skeleton 细化（裂缝评估常用）
    """

    def __init__(self,
                 thresholds=None,
                 tolerance: int = 2,
                 thin_prediction: bool = False):
        self.thr = thresholds if thresholds is not None else [0.5]
        self.tol = int(tolerance)
        self.thin = bool(thin_prediction)
        self.reset()

    def reset(self):
        self.tp = {t: 0 for t in self.thr}
        self.fp = {t: 0 for t in self.thr}
        self.fn = {t: 0 for t in self.thr}
        self.cldice_sum = 0.0
        self.n = 0

    def update(self, prob: torch.Tensor, target: torch.Tensor):
        """prob: [B,1,H,W] in [0,1]; target: [B,1,H,W] {0,1}."""
        prob_np = _to_np(prob); tgt_np = _to_np(target).astype(np.uint8)
        B = prob_np.shape[0]
        for b in range(B):
            t = tgt_np[b, 0]
            for thr in self.thr:
                p = (prob_np[b, 0] >= thr).astype(np.uint8)
                if self.thin and p.sum() > 0:
                    p = skeletonize(p.astype(bool)).astype(np.uint8)
                tp, fp, fn = _tp_fp_fn_tolerant(p, t, self.tol)
                self.tp[thr] += tp
                self.fp[thr] += fp
                self.fn[thr] += fn
            # clDice 始终在 0.5 阈值下算（不受 tol 影响）
            p_05 = (prob_np[b, 0] >= 0.5).astype(np.uint8)
            self.cldice_sum += cl_dice(p_05, t)
            self.n += 1

    def compute(self):
        eps = 1e-6
        results = {}
        best_f1 = -1; ods_thr = self.thr[0]
        for thr in self.thr:
            tp = self.tp[thr]; fp = self.fp[thr]; fn = self.fn[thr]
            P = tp / (tp + fp + eps); R = tp / (tp + fn + eps)
            F = 2 * P * R / (P + R + eps)
            IoU = tp / (tp + fp + fn + eps)
            results[f"P@{thr:.2f}"] = P
            results[f"R@{thr:.2f}"] = R
            results[f"F@{thr:.2f}"] = F
            results[f"IoU@{thr:.2f}"] = IoU
            if F > best_f1:
                best_f1 = F; ods_thr = thr
        results["ODS"]       = best_f1
        results["ODS_thr"]   = ods_thr
        results["clDice"]    = self.cldice_sum / max(self.n, 1)
        results["tolerance"] = self.tol
        results["thin_pred"] = self.thin
        return results
