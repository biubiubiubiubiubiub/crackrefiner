"""
Confidence Gating Module:
    r(p) = σ( α·|2C_T(p) - 1|              # SAM logits 锐度
           + β·exp(-d_skel(p)/σ_s)          # 距 skeleton 的距离
           + γ·(1 - Var_MC(p)) )            # MC dropout 一致性
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def _soft_skel(mask: torch.Tensor, iters: int = 5) -> torch.Tensor:
    """morphological skeleton 的可微近似（min-/max-pool 风格）。"""
    skel = torch.zeros_like(mask)
    img = mask
    for _ in range(iters):
        eroded = -F.max_pool2d(-img, 3, 1, 1)
        opened = F.max_pool2d(eroded, 3, 1, 1)
        skel = skel + F.relu(img - opened)
        img = eroded
    return skel.clamp(0, 1)


def _dist_to_skel(mask: torch.Tensor, max_iter: int = 20) -> torch.Tensor:
    """近似 distance transform：迭代 dilation 计 hop。"""
    skel = (_soft_skel(mask) > 0.5).float()
    d = torch.full_like(skel, float(max_iter))
    cur = skel.clone()
    for i in range(max_iter):
        d = torch.where(cur > 0, torch.full_like(d, float(i)), d)
        cur = F.max_pool2d(cur, 3, 1, 1)
    return d


class ConfidenceGate:
    """无可学习参数的伪标签可靠性估计。"""

    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0,
                 sigma_s=5.0, tau=0.5):
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.sigma_s = sigma_s
        self.tau = tau

    @torch.no_grad()
    def __call__(self,
                 pseudo: torch.Tensor,   # [B,1,H,W] {0,1}
                 conf:   torch.Tensor,   # [B,1,H,W] ∈[0,1] SAM probability
                 var:    torch.Tensor    # [B,1,H,W] ∈[0,?] MC variance
                 ):
        sharp = (2 * conf - 1).abs()
        d = _dist_to_skel(pseudo)
        skel_term = torch.exp(-d / self.sigma_s)
        v_norm = (var - var.amin(dim=(2, 3), keepdim=True)) / \
                 (var.amax(dim=(2, 3), keepdim=True) - var.amin(dim=(2, 3), keepdim=True) + 1e-6)
        mc_term = 1.0 - v_norm
        r = torch.sigmoid(self.alpha * sharp + self.beta * skel_term + self.gamma * mc_term - 1.5)
        reliable_mask = (r > self.tau).float()
        return r, reliable_mask
