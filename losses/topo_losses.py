"""
TopoSAM-Crack 的复合损失：
  L = w_dice * Dice
    + w_bce  * BCE_weighted_by_reliability
    + w_skel * SkeletonRecallLoss     (ECCV 2024)
    + w_cldice * clDice loss (CVPR 2021) — 同时约束 precision/recall 在 skeleton 上
    + w_dt   * DistanceTransformLoss
    + w_edge * EdgeBCE
    + w_cons * ConsistencyEMA  (在 trainer 里调用)
"""
from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize


# ---------------- 基础 ----------------
def dice_loss(prob: torch.Tensor, target: torch.Tensor,
              weight: Optional[torch.Tensor] = None, eps: float = 1e-6):
    if weight is not None:
        prob = prob * weight
        target = target * weight
    inter = (prob * target).sum(dim=(1, 2, 3))
    union = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1 - ((2 * inter + eps) / (union + eps)).mean()


def weighted_bce_loss(logit: torch.Tensor, target: torch.Tensor,
                      weight: torch.Tensor):
    """Pixel-wise reliability-weighted BCE."""
    bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    return (bce * weight).sum() / (weight.sum() + 1e-6)


# ---------------- Skeleton Recall (ECCV 2024) ----------------
def _np_skeleton(mask01_uint8: np.ndarray) -> np.ndarray:
    """non-differentiable, 对 GT/伪标签做。"""
    return skeletonize(mask01_uint8.astype(bool)).astype(np.float32)


def skeleton_recall_loss(prob: torch.Tensor, target: torch.Tensor,
                         dilate: int = 2, eps: float = 1e-6):
    """
    L = 1 - |prob ∩ dilate(skel(target))| / |skel(target)|
    target: {0,1} mask [B,1,H,W]
    """
    B = target.size(0)
    losses = []
    tgt_cpu = target.detach().cpu().numpy()
    for b in range(B):
        skel = _np_skeleton(tgt_cpu[b, 0])
        if skel.sum() < 1:
            losses.append(prob.new_zeros(()))
            continue
        skel_t = torch.from_numpy(skel).to(prob.device).unsqueeze(0).unsqueeze(0)
        if dilate > 0:
            skel_dil = F.max_pool2d(skel_t, 2 * dilate + 1, 1, dilate)
        else:
            skel_dil = skel_t
        p = prob[b:b+1]
        recall = (p * skel_t).sum() / (skel_t.sum() + eps)
        # 同时鼓励 prob 覆盖到加粗 skel 的区域，以兼顾细丝宽度
        cover = (p * skel_dil).sum() / (skel_dil.sum() + eps)
        losses.append(1 - 0.5 * (recall + cover))
    return torch.stack(losses).mean()


# ---------------- Distance Transform Loss ----------------
def _np_dt(mask01: np.ndarray) -> np.ndarray:
    """计算 binary mask 内部到边界的归一化 DT。"""
    if mask01.sum() < 1:
        return np.zeros_like(mask01, dtype=np.float32)
    dt = distance_transform_edt(mask01)
    return (dt / (dt.max() + 1e-6)).astype(np.float32)


def distance_transform_loss(prob: torch.Tensor, target: torch.Tensor,
                            reliable: Optional[torch.Tensor] = None):
    """L1 between DT(target) and DT(prob>0.5)."""
    B = target.size(0)
    tgt_np = target.detach().cpu().numpy()
    dt_t = np.stack([_np_dt(tgt_np[b, 0]) for b in range(B)], 0)
    dt_t = torch.from_numpy(dt_t).unsqueeze(1).to(prob.device)
    # prob 直接当作 soft mask 计算 1-阶矩近似
    pred_dt = prob                              # in [0,1]
    diff = (pred_dt - dt_t).abs()
    if reliable is not None:
        diff = diff * reliable
        return diff.sum() / (reliable.sum() + 1e-6)
    return diff.mean()


# ---------------- clDice Loss (CVPR 2021) ----------------
def _soft_skel(prob: torch.Tensor, iters: int = 5) -> torch.Tensor:
    """可微 skeleton 近似 (min-/max-pool)。"""
    img = prob
    skel = torch.zeros_like(prob)
    for _ in range(iters):
        eroded = -F.max_pool2d(-img, 3, 1, 1)
        opened = F.max_pool2d(eroded, 3, 1, 1)
        skel = skel + F.relu(img - opened) * (1 - skel)
        img = eroded
    return skel.clamp(0, 1)


def cldice_loss(prob: torch.Tensor, target: torch.Tensor,
                iters: int = 5, eps: float = 1e-6):
    """
    L_clDice = 1 - 2 * tprec * tsens / (tprec + tsens)
      tprec = sum(skel(pred) * target) / sum(skel(pred))
      tsens = sum(skel(target) * pred) / sum(skel(target))
    target: {0,1}  (会被当作 soft mask)
    """
    sk_p = _soft_skel(prob, iters)
    sk_t = _soft_skel(target, iters)
    tprec = (sk_p * target).sum() / (sk_p.sum() + eps)
    tsens = (sk_t * prob  ).sum() / (sk_t.sum() + eps)
    cldice = 2 * tprec * tsens / (tprec + tsens + eps)
    return 1 - cldice


# ---------------- Edge BCE ----------------
def edge_bce_loss(edge_logit: torch.Tensor, target: torch.Tensor, band: int = 2):
    """target 通过形态学得到 boundary band。"""
    with torch.no_grad():
        dil = F.max_pool2d(target, 2 * band + 1, 1, band)
        ero = -F.max_pool2d(-target, 2 * band + 1, 1, band)
        edge = (dil - ero).clamp(0, 1)
    return F.binary_cross_entropy_with_logits(edge_logit, edge)


# ---------------- 多尺度特征一致性（创新点） ----------------
def multi_scale_consistency_loss(
    stu_feats: list,              # Student 多尺度特征 [f1, f2, f3, ...]
    ema_feats: list,              # EMA Teacher 多尺度特征
    reliable_mask: torch.Tensor,  # [B,1,H,W] 可靠区域掩码
    w_unreliable: float = 2.0,    # 不可靠区域的权重放大倍数
    eps: float = 1e-6,
) -> torch.Tensor:
    """多尺度特征一致性损失。

    在多个特征尺度上对齐 Student 和 EMA Teacher 的中间表示。
    不可靠区域（reliable_mask=0）的权重提高 w_unreliable 倍，
    迫使 Student 在伪标签不可靠的区域也保持与 EMA Teacher 一致的特征。

    论文叙事：
        "We extend consistency regularization from the output space
         to the feature space, enforcing alignment at multiple levels
         of abstraction.  This multi-scale constraint prevents the
         student from developing domain-specific shortcuts that
         happen to produce correct outputs but rely on brittle features."
    """
    total_loss = 0.0
    count = 0
    for s_feat, e_feat in zip(stu_feats, ema_feats):
        if s_feat.shape != e_feat.shape:
            continue
        mask = F.interpolate(reliable_mask, size=s_feat.shape[2:],
                             mode="nearest")
        weight = mask * 1.0 + (1 - mask) * w_unreliable
        diff = (s_feat - e_feat.detach()).abs().mean(dim=1, keepdim=True)
        total_loss = total_loss + (diff * weight).sum() / (weight.sum() + eps)
        count += 1
    return total_loss / max(count, 1)


# # ---------------- 复合 ----------------
# class TopoSAMLoss(nn.Module):
#     def __init__(self, cfg_loss: dict):
#         super().__init__()
#         self.w = cfg_loss

#     def forward(self,
#                 out: dict,                           # student outputs
#                 pseudo: torch.Tensor,                # SAM pseudo {0,1}
#                 reliability: torch.Tensor,           # r ∈[0,1]
#                 reliable_mask: torch.Tensor):        # {0,1}
#         logit = out["mask_logit"]
#         prob = torch.sigmoid(logit)

#         l_dice = dice_loss(prob, pseudo, weight=reliable_mask)
#         l_bce = weighted_bce_loss(logit, pseudo, weight=reliability)
#         l_skel = skeleton_recall_loss(prob, pseudo)
#         l_cldice = (cldice_loss(prob, pseudo)
#                     if self.w.get("w_cldice", 0.0) > 0
#                     else logit.new_zeros(()))
#         l_dt = distance_transform_loss(prob, pseudo, reliable=reliable_mask)
#         l_edge = (edge_bce_loss(out["edge_logit"], pseudo)
#                   if "edge_logit" in out else logit.new_zeros(()))

#         loss = (self.w["w_dice"] * l_dice
#                 + self.w["w_bce"]  * l_bce
#                 + self.w["w_skelrecall"] * l_skel
#                 + self.w.get("w_cldice", 0.0) * l_cldice
#                 + self.w["w_dt"]   * l_dt
#                 + self.w["w_edge"] * l_edge)
#         return loss, {
#             "dice":    l_dice.detach(),
#             "bce":     l_bce.detach(),
#             "skel":    l_skel.detach(),
#             "cldice":  l_cldice.detach(),
#             "dt":      l_dt.detach(),
#             "edge":    l_edge.detach(),
#         }
# ---------------- 复合 ----------------

class TopoSAMLoss(nn.Module):
    def __init__(self, cfg_loss: dict):
        super().__init__()
        self.w = cfg_loss

    def forward(self,
                out: dict,
                pseudo: torch.Tensor,
                reliability: torch.Tensor,
                reliable_mask: torch.Tensor,
                ema_prob: Optional[torch.Tensor] = None):
        """
        新增参数：
            ema_prob: EMA 教师的预测概率（已 sigmoid），形状 [B,1,H,W]
                      仅当 w_consistency > 0 时需要传入
        """
        logit = out["mask_logit"]
        prob = torch.sigmoid(logit)

        # === 基础损失 ===
        l_dice = dice_loss(prob, pseudo, weight=reliable_mask)
        l_bce = weighted_bce_loss(logit, pseudo, weight=reliability)

        # === 拓扑相关损失（消融时权重可设为0） ===
        l_skel = skeleton_recall_loss(prob, pseudo)
        l_cldice = (cldice_loss(prob, pseudo)
                    if self.w.get("w_cldice", 0.0) > 0
                    else logit.new_zeros(()))
        l_dt = distance_transform_loss(prob, pseudo, reliable=reliable_mask)
        l_edge = (edge_bce_loss(out["edge_logit"], pseudo)
                  if "edge_logit" in out else logit.new_zeros(()))

        # === 新增：Consistency Loss（EMA 教师一致性）===
        l_cons = logit.new_zeros(())
        if self.w.get("w_consistency", 0.0) > 0 and ema_prob is not None:
            w_unreliable = (1.0 - reliable_mask).float()
            l_cons = ((prob - ema_prob).abs() * w_unreliable).sum() / \
                     (w_unreliable.sum() + 1e-6)

        # === 加权总损失 ===
        loss = (self.w["w_dice"] * l_dice
                + self.w["w_bce"] * l_bce
                + self.w.get("w_skelrecall", 0.0) * l_skel
                + self.w.get("w_cldice", 0.0) * l_cldice
                + self.w["w_dt"] * l_dt
                + self.w["w_edge"] * l_edge
                + self.w.get("w_consistency", 0.0) * l_cons)

        # === 返回日志字典 ===
        parts = {
            "dice": l_dice.detach(),
            "bce": l_bce.detach(),
            "skel": l_skel.detach(),
            "cldice": l_cldice.detach(),
            "dt": l_dt.detach(),
            "edge": l_edge.detach(),
            "cons": l_cons.detach(),
        }
        return loss, parts
