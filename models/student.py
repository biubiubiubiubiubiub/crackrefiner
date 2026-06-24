"""
SCSegamba-Lite Student（~3M 参数）：
- Encoder：Conv-stem + 4 个 VSS-Lite stage（双向 1D 扫描近似 SSM）
- Decoder：U-Net 风格 skip connection
- 双分支输出：mask logit + edge logit

为可独立运行，本实现 *不依赖* 第三方 mamba_ssm；
使用纯 PyTorch 写的 BiDirSSM (近似 S6 行为：dt + selective scan 用 GRU 替代)。
若你已安装 mamba_ssm，可在 BiDirSSM 中替换为官方实现以提速。
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------- 轻量 SSM 近似 --------------------
class BiDirSSM(nn.Module):
    """双向 1D selective scan 的轻量近似（GRU + depthwise conv）。

    显存优化：
      • 默认 d_state=8（原 16），输出通道减半，scan 显存 / 计算 ×0.5；
      • 训练时对 GRU 走 gradient checkpoint，把 activation 从 O(N) 降到 O(√N)；
      • chunked sequence: 当 seq_len > chunk_size 时，把 1D 序列切块逐段过 GRU
        并拼接，避免一次性把 BxNxd_state 的 activation 全保存。
    """

    def __init__(self, dim, expand: int = 2, d_state: int = 8,
                 chunk_size: int = 256, use_checkpoint: bool = True):
        super().__init__()
        d_inner = dim * expand
        self.in_proj = nn.Linear(dim, d_inner * 2)
        self.dwconv = nn.Conv1d(d_inner, d_inner, 3, padding=1, groups=d_inner)
        self.gru_f = nn.GRU(d_inner, d_state, batch_first=True)
        self.gru_b = nn.GRU(d_inner, d_state, batch_first=True)
        self.out_proj = nn.Linear(d_state * 2, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.SiLU()
        self.chunk_size = int(chunk_size)
        self.use_checkpoint = bool(use_checkpoint)

    # --- chunked GRU forward 帮助函数 ---
    @staticmethod
    def _chunked_gru(gru: nn.GRU, x: torch.Tensor, chunk: int) -> torch.Tensor:
        """x: (B, N, C) -> (B, N, H). 把长序列切成块，块间携带 hidden state."""
        N = x.size(1)
        if N <= chunk:
            y, _ = gru(x)
            return y
        outs = []
        h = None
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            y, h = gru(x[:, s:e].contiguous(), h)
            outs.append(y)
        return torch.cat(outs, dim=1)

    def _scan(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, d_inner) -> (B, N, 2*d_state)"""
        y_f = self._chunked_gru(self.gru_f, x, self.chunk_size)
        x_rev = torch.flip(x, [1])
        y_b = self._chunked_gru(self.gru_b, x_rev, self.chunk_size)
        y_b = torch.flip(y_b, [1])
        return torch.cat([y_f, y_b], dim=-1)

    def forward(self, x):  # x: [B, N, C]
        residual = x
        x = self.norm(x)
        x, gate = self.in_proj(x).chunk(2, dim=-1)
        x = self.dwconv(x.transpose(1, 2)).transpose(1, 2)
        x = self.act(x)

        if self.use_checkpoint and self.training:
            from torch.utils.checkpoint import checkpoint
            y = checkpoint(self._scan, x, use_reentrant=False)
        else:
            y = self._scan(x)

        y = self.out_proj(y)
        y = y * self.act(gate[..., :y.size(-1)] if gate.size(-1) >= y.size(-1) else gate)
        return residual + y


class VSSBlock(nn.Module):
    """2D vision block: H 方向 + W 方向 双 SSM。"""

    def __init__(self, dim):
        super().__init__()
        self.ssm_h = BiDirSSM(dim)
        self.ssm_w = BiDirSSM(dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x):                                # [B, C, H, W]
        B, C, H, W = x.shape
        # H 方向
        z = x.permute(0, 2, 3, 1).reshape(B * H, W, C)
        z = self.ssm_w(z).reshape(B, H, W, C)
        # W 方向
        z = z.permute(0, 2, 1, 3).reshape(B * W, H, C)
        z = self.ssm_h(z).reshape(B, W, H, C).permute(0, 2, 1, 3)
        z = z + self.mlp(z)
        return z.permute(0, 3, 1, 2).contiguous()


def conv_bn_act(ic, oc, k=3, s=1, p=1, act=True):
    layers = [nn.Conv2d(ic, oc, k, s, p, bias=False), nn.BatchNorm2d(oc)]
    if act: layers.append(nn.SiLU(inplace=True))
    return nn.Sequential(*layers)


class Down(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.conv = conv_bn_act(ic, oc, 3, 2, 1)
        self.vss = VSSBlock(oc)

    def forward(self, x):
        return self.vss(self.conv(x))


class Up(nn.Module):
    def __init__(self, ic, sc, oc):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.fuse = conv_bn_act(ic + sc, oc)
        self.vss = VSSBlock(oc)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                              align_corners=False)
        return self.vss(self.fuse(torch.cat([x, skip], 1)))


# -------------------- Edge-Aware Branch --------------------
class EdgeBranch(nn.Module):
    """HED 风格 multi-scale 边界分支。"""

    def __init__(self, channels_list):
        super().__init__()
        self.reduce = nn.ModuleList([nn.Conv2d(c, 32, 1) for c in channels_list])
        self.fuse = nn.Conv2d(32 * len(channels_list), 1, 1)

    def forward(self, feats, out_size):
        outs = []
        for r, f in zip(self.reduce, feats):
            outs.append(F.interpolate(r(f), size=out_size,
                                      mode="bilinear", align_corners=False))
        return self.fuse(torch.cat(outs, 1))


# -------------------- Student --------------------
class SCSegambaLite(nn.Module):
    def __init__(self, in_channels=3, base=32, num_classes=1, edge_branch=True):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 6
        self.stem = conv_bn_act(in_channels, c1)
        self.d1 = Down(c1, c2)
        self.d2 = Down(c2, c3)
        self.d3 = Down(c3, c4)
        self.bottleneck = VSSBlock(c4)

        self.u3 = Up(c4, c3, c3)
        self.u2 = Up(c3, c2, c2)
        self.u1 = Up(c2, c1, c1)

        self.head = nn.Conv2d(c1, num_classes, 1)
        self.edge = EdgeBranch([c1, c2, c3]) if edge_branch else None

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.d1(s0)
        s2 = self.d2(s1)
        s3 = self.d3(s2)
        b  = self.bottleneck(s3)
        u3 = self.u3(b,  s2)
        u2 = self.u2(u3, s1)
        u1 = self.u1(u2, s0)
        mask_logit = self.head(u1)
        out = {"mask_logit": mask_logit}
        if self.edge is not None:
            out["edge_logit"] = self.edge([u1, u2, u3], out_size=x.shape[-2:])
        return out


def build_student(cfg):
    """Factory.  Picks backbone by cfg['student']['name']:
        - 'scsegamba_lite'   : legacy BiSSM U-Net (~2.43M)              [keep]
        - 'scsegamba'        : SCSegamba-inspired (GBC + SASS + MFS)    [new]
    """
    s = cfg["student"]
    name = str(s.get("name", "scsegamba_lite")).lower()
    if name in ("scsegamba", "savss", "scsegamba_savss"):
        from .student_scsegamba import SCSegambaStudent
        return SCSegambaStudent(
            in_channels=s["in_channels"],
            base=s.get("base_channels", 32),
            num_classes=s["num_classes"],
            edge_branch=s.get("edge_branch", True),
            num_layers=s.get("num_layers", 4),
            d_state=s.get("d_state", 8),
            use_diag=s.get("use_diag", True),
            chunk_size=s.get("chunk_size", 256),
            use_checkpoint=s.get("use_checkpoint", True),
            mfs_embed_dim=s.get("mfs_embed_dim", 32),
        )
    # default = legacy lite
    return SCSegambaLite(
        in_channels=s["in_channels"],
        base=s["base_channels"],
        num_classes=s["num_classes"],
        edge_branch=s.get("edge_branch", True),
    )
