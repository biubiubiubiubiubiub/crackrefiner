"""TopoSAM-Crack 训练器（含 LACG + Hard Example Mining + Entropy Reg）。"""
from __future__ import annotations
import copy
import math
import os
import time
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from losses import build_loss
from models import ConfidenceGate, build_student
from models.adaptive_gate import (AdaptiveConfidenceGate, HardExampleTracker,
                                  gate_entropy_loss, build_adaptive_gate)
from metrics import SegMetric


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.model.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1 - self.decay)
            else:
                v.copy_(msd[k])


def build_optimizer(model, cfg):
    return torch.optim.AdamW(model.parameters(),
                             lr=cfg["train"]["lr"],
                             weight_decay=cfg["train"]["weight_decay"])


def build_scheduler(opt, cfg, steps_per_epoch):
    total = cfg["train"]["epochs"] * steps_per_epoch
    warmup = cfg["train"]["warmup_epochs"] * steps_per_epoch

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


# class Trainer:
#     def __init__(self, cfg, train_loader, val_loader=None, teacher=None):
#         """
#         teacher: 保留参数仅为向后兼容；TopoSAM-Crack 不在线调用 teacher，
#                  SAM3 伪标签全部离线生成（见 scripts/02_generate_sam3_pseudo.sh）。
#         """
#         self.cfg = cfg
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"

#         self.model = build_student(cfg).to(self.device)
#         self.ema = EMA(self.model, decay=cfg["train"]["ema_decay"])
#         self.gate = ConfidenceGate(**{k: cfg["gate"][k] for k in
#                                        ("alpha", "beta", "gamma", "sigma_s", "tau")})
#         # build_loss picks TopoSAMLoss (legacy 7-loss/lite) or
#         # RWFTTopoSAMLoss (new 4-loss) via cfg['loss']['type'].
#         self.loss_fn = build_loss(cfg["loss"]).to(self.device)
#         self.opt = build_optimizer(self.model, cfg)
#         self.sched = build_scheduler(self.opt, cfg, len(train_loader))
#         self.scaler = torch.amp.GradScaler("cuda", enabled=cfg["train"]["amp"])
#         self.train_loader = train_loader
#         self.val_loader = val_loader

#         self.out_dir = cfg["experiment"]["output_dir"]
#         os.makedirs(self.out_dir, exist_ok=True)
#         self.writer = SummaryWriter(os.path.join(self.out_dir, "tb"))
#         self.global_step = 0
#         self.start_epoch = 0
#         self.best_metric = -1.0

#         if cfg["train"].get("resume"):
#             self._load(cfg["train"]["resume"])

#     # ---------- step ----------
#     def _train_step(self, batch, epoch: int) -> Dict[str, torch.Tensor]:
#         img    = batch["image"].to(self.device, non_blocking=True)
#         pseudo = batch["pseudo"].to(self.device, non_blocking=True)
#         conf   = batch["conf"].to(self.device, non_blocking=True)
#         var    = batch["var"].to(self.device, non_blocking=True)

#         # ---- Confidence Gate ----
#         r, reliable = self.gate(pseudo, conf, var)

#         accum = max(1, int(self.cfg["train"].get("grad_accum_steps", 1)))

#         with torch.amp.autocast("cuda", enabled=self.cfg["train"]["amp"]):
#             out = self.model(img)
#             loss, parts = self.loss_fn(out, pseudo, r, reliable)
#             # ---- consistency with EMA teacher (在不可靠区域) ----
#             if self.cfg["loss"].get("w_consistency", 0.0) > 0:
#                 with torch.no_grad():
#                     ema_prob = torch.sigmoid(self.ema.model(img)["mask_logit"])
#                 stu_prob = torch.sigmoid(out["mask_logit"])
#                 w_unreliable = (1.0 - reliable)
#                 cons = ((stu_prob - ema_prob).abs() * w_unreliable).sum() / \
#                        (w_unreliable.sum() + 1e-6)
#                 loss = loss + self.cfg["loss"]["w_consistency"] * cons
#                 parts["cons"] = cons.detach()

#         # ---- 梯度累计 ----
#         self.scaler.scale(loss / accum).backward()
#         self._accum_count = getattr(self, "_accum_count", 0) + 1
#         if self._accum_count % accum == 0:
#             self.scaler.unscale_(self.opt)
#             torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
#             self.scaler.step(self.opt)
#             self.scaler.update()
#             self.opt.zero_grad(set_to_none=True)
#             self.sched.step()
#             self.ema.update(self.model)
#         parts["loss"] = loss.detach()
#         return parts
# ==================== trainer.py 修改后的关键部分 ====================

class Trainer:
    def __init__(self, cfg, train_loader, val_loader=None, teacher=None):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = build_student(cfg).to(self.device)
        self.ema = EMA(self.model, decay=cfg["train"]["ema_decay"])

        # ---- Gate: 自适应（LACG）或 传统（ConfidenceGate） ----
        self.gate_type = cfg["gate"].get("type", "legacy")
        if self.gate_type == "adaptive":
            self.gate = build_adaptive_gate(cfg["gate"]).to(self.device)
            self.gate_opt = torch.optim.AdamW(
                self.gate.parameters(),
                lr=cfg["gate"].get("lr", cfg["train"]["lr"] * 0.5),
                weight_decay=cfg["gate"].get("weight_decay", 0.0),
            )
            self.hard_tracker = HardExampleTracker(
                window=cfg["gate"].get("hard_window", 0.9))
            self.gate_w_entropy = cfg["gate"].get("w_entropy", 0.01)
            print(f"[Trainer] AdaptiveConfidenceGate  params="
                  f"{sum(p.numel() for p in self.gate.parameters()):,}")
        else:
            self.gate = ConfidenceGate(**{k: cfg["gate"][k] for k in
                                           ("alpha", "beta", "gamma", "sigma_s", "tau")})
            self.hard_tracker = None
            self.gate_opt = None
            self.gate_w_entropy = 0.0

        self.loss_fn = build_loss(cfg["loss"]).to(self.device)
        self.opt = build_optimizer(self.model, cfg)
        self.sched = build_scheduler(self.opt, cfg, len(train_loader))
        self.scaler = torch.amp.GradScaler("cuda", enabled=cfg["train"]["amp"])

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.out_dir = cfg["experiment"]["output_dir"]
        os.makedirs(self.out_dir, exist_ok=True)
        self.writer = SummaryWriter(os.path.join(self.out_dir, "tb"))
        self.global_step = 0
        self.start_epoch = 0
        self.best_metric = -1.0

        if cfg["train"].get("resume"):
            self._load(cfg["train"]["resume"])

    # ---------- 修改后的 _train_step ----------
    def _train_step(self, batch, epoch: int) -> Dict[str, torch.Tensor]:
        img    = batch["image"].to(self.device, non_blocking=True)
        pseudo = batch["pseudo"].to(self.device, non_blocking=True)
        conf   = batch["conf"].to(self.device, non_blocking=True)
        var    = batch["var"].to(self.device, non_blocking=True)

        # ---- Confidence Gate ----
        if self.gate_type == "adaptive":
            hard = (self.hard_tracker.get_hard_mask(top_k_ratio=0.3)
                    if self.hard_tracker.loss_ema is not None else None)
            r, reliable = self.gate(pseudo, conf, var, image=img, hard_mask=hard)
        else:
            r, reliable = self.gate(pseudo, conf, var)

        accum = max(1, int(self.cfg["train"].get("grad_accum_steps", 1)))

        with torch.amp.autocast("cuda", enabled=self.cfg["train"]["amp"]):
            out = self.model(img)

            # === 计算 EMA 概率（仅当需要 consistency 时） ===
            ema_prob = None
            if self.cfg["loss"].get("w_consistency", 0.0) > 0:
                with torch.no_grad():
                    ema_prob = torch.sigmoid(self.ema.model(img)["mask_logit"])

            # === 统一调用 loss_fn（已整合 consistency） ===
            loss, parts = self.loss_fn(out, pseudo, r, reliable, ema_prob=ema_prob)

            # === Gate 熵正则（防止 LACG 退化） ===
            if self.gate_type == "adaptive" and self.gate_w_entropy > 0:
                l_ent = gate_entropy_loss(r)
                loss = loss + self.gate_w_entropy * l_ent
                parts["gate_ent"] = l_ent.detach()

        # ---- 梯度累计 ----
        self.scaler.scale(loss / accum).backward()
        self._accum_count = getattr(self, "_accum_count", 0) + 1

        if self._accum_count % accum == 0:
            self.scaler.unscale_(self.opt)
            if self.gate_type == "adaptive":
                self.scaler.unscale_(self.gate_opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.scaler.step(self.opt)
            if self.gate_type == "adaptive":
                self.scaler.step(self.gate_opt)
            self.scaler.update()
            self.opt.zero_grad(set_to_none=True)
            if self.gate_type == "adaptive":
                self.gate_opt.zero_grad(set_to_none=True)
            self.sched.step()
            self.ema.update(self.model)

            # ---- 更新难例追踪 ----
            if self.hard_tracker is not None and "bce" in parts:
                self._update_hard_tracker(out, pseudo, r)

        parts["loss"] = loss.detach()
        return parts

    def _update_hard_tracker(self, out, pseudo, r):
        """基于逐像素 BCE 更新难例追踪器。"""
        with torch.no_grad():
            prob = torch.sigmoid(out["mask_logit"])
            bce_per_pixel = F.binary_cross_entropy(prob, pseudo, reduction="none")
            weighted = bce_per_pixel * r
            self.hard_tracker.update(weighted)
    # ---------- fit ----------
    def fit(self):
        cfg_t = self.cfg["train"]
        for epoch in range(self.start_epoch, cfg_t["epochs"]):
            t0 = time.time()
            self.model.train()
            for it, batch in enumerate(self.train_loader):
                parts = self._train_step(batch, epoch)
                self.global_step += 1
                if self.global_step % cfg_t["log_every"] == 0:
                    log = " ".join(f"{k}={v.item():.4f}" for k, v in parts.items())
                    print(f"[ep {epoch} it {it}/{len(self.train_loader)}] "
                          f"lr={self.opt.param_groups[0]['lr']:.2e} {log}")
                    for k, v in parts.items():
                        self.writer.add_scalar(f"train/{k}", v.item(), self.global_step)

            print(f"[ep {epoch}] time={time.time()-t0:.1f}s")

            # ----- val -----
            if self.val_loader is not None:
                metrics = self.validate()
                for k, v in metrics.items():
                    self.writer.add_scalar(f"val/{k}", v, epoch)
                cur = metrics.get("ODS", metrics.get("F@0.50", 0.0))
                if cur > self.best_metric:
                    self.best_metric = cur
                    self._save("best.pth", epoch)
                print(f"[ep {epoch}] val ODS={cur:.4f}  best={self.best_metric:.4f}")

            if (epoch + 1) % cfg_t["ckpt_every"] == 0:
                self._save(f"epoch_{epoch+1}.pth", epoch)

        self._save("last.pth", cfg_t["epochs"] - 1)

    # ---------- val ----------
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        metric = SegMetric(thresholds=[0.3, 0.4, 0.5, 0.6, 0.7])
        for batch in self.val_loader:
            img = batch["image"].to(self.device, non_blocking=True)
            gt = batch.get("gt")
            if gt is None:
                continue
            gt = gt.to(self.device)
            with torch.amp.autocast("cuda", enabled=self.cfg["train"]["amp"]):
                prob = torch.sigmoid(self.model(img)["mask_logit"])
            if prob.shape[-2:] != gt.shape[-2:]:
                prob = F.interpolate(prob, size=gt.shape[-2:],
                                     mode="bilinear", align_corners=False)
            metric.update(prob, gt)
        return metric.compute()

    # ---------- io ----------
    def _save(self, name, epoch):
        path = os.path.join(self.out_dir, name)
        payload = {
            "model": self.model.state_dict(),
            "ema":   self.ema.model.state_dict(),
            "opt":   self.opt.state_dict(),
            "epoch": epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "cfg":   self.cfg,
        }
        if self.gate_type == "adaptive":
            payload["gate"] = self.gate.state_dict()
            payload["gate_opt"] = self.gate_opt.state_dict()
        torch.save(payload, path)
        print(f"[save] {path}")

    def _load(self, path):
        ck = torch.load(path, map_location="cpu")
        self.model.load_state_dict(ck["model"])
        self.ema.model.load_state_dict(ck["ema"])
        self.opt.load_state_dict(ck["opt"])
        self.start_epoch = ck["epoch"] + 1
        self.global_step = ck["global_step"]
        self.best_metric = ck["best_metric"]
        if self.gate_type == "adaptive" and "gate" in ck:
            self.gate.load_state_dict(ck["gate"])
            self.gate_opt.load_state_dict(ck["gate_opt"])
        print(f"[resume] from {path} epoch={ck['epoch']}")
