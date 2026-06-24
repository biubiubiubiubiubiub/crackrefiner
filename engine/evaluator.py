"""跨域评估器（论文协议版）。

- 默认 `mode='eval_original'`：图像 resize 到训练分辨率推理，
  预测概率插值回 GT 原图尺寸，再算指标（与 DeepCrack/CrackForest 协议对齐）
- 默认 tolerance=2 像素：±2 px 容差匹配（裂缝评估标准协议）
- 阈值范围 0.05..0.99 全扫描，找真实 ODS

也支持严格模式：传 strict=True 即关闭 tolerance 与 original size。
"""
from __future__ import annotations
import json
import os

import torch
import torch.nn.functional as F

from datasets import build_loader
from metrics import SegMetric
from models import build_student


class Evaluator:
    def __init__(self, cfg, ckpt_path: str):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = build_student(cfg).to(self.device).eval()
        ck = torch.load(ckpt_path, map_location="cpu")
        # 优先使用 EMA
        state = ck.get("ema", ck["model"])
        self.model.load_state_dict(state)
        print(f"[Evaluator] loaded ckpt: {ckpt_path} (epoch={ck.get('epoch')})")

    # -------------------------------------------------------------
    @torch.no_grad()
    def evaluate_manifest(self,
                          manifest_path: str,
                          split: str = "test",
                          name: str = "",
                          tolerance: int = 2,
                          thin_pred: bool = False,
                          use_original_size: bool = True):
        """跨域评估单个数据集。"""
        mode = "eval_original" if use_original_size else "eval"
        loader = build_loader(self.cfg, manifest_path, split, mode=mode,
                              batch_size=1, shuffle=False)

        # 阈值扫描：0.05..0.95 步长 0.05 + 末端密集
        thresholds = [round(0.05 + 0.05 * i, 2) for i in range(19)]
        thresholds += [0.96, 0.97, 0.98, 0.99]
        metric = SegMetric(thresholds=sorted(set(thresholds)),
                           tolerance=tolerance, thin_prediction=thin_pred)

        for batch in loader:
            img = batch["image"].to(self.device, non_blocking=True)
            gt  = batch["gt"].to(self.device)
            prob = torch.sigmoid(self.model(img)["mask_logit"])
            # 插值回 GT 尺寸
            if prob.shape[-2:] != gt.shape[-2:]:
                prob = F.interpolate(prob, size=gt.shape[-2:],
                                     mode="bilinear", align_corners=False)
            metric.update(prob, gt)

        m = metric.compute()
        print(f"[Eval] {name or manifest_path}/{split}  "
              f"(tol={tolerance}, thin={thin_pred}, orig_size={use_original_size}):")
        for k, v in m.items():
            if isinstance(v, float):
                print(f"   {k}: {v:.4f}")
        return m

    # -------------------------------------------------------------
    def evaluate_crossdomain(self,
                             save_path: str | None = None,
                             tolerance: int = 2,
                             thin_pred: bool = False,
                             use_original_size: bool = True):
        cd_list = self.cfg.get("crossdomain", []) or []
        if not cd_list:
            print("[Eval] no cross-domain datasets configured, skip.")
            return {}
        results = {}
        for cd in cd_list:
            m = self.evaluate_manifest(
                cd["manifest"], cd["split"], cd["name"],
                tolerance=tolerance, thin_pred=thin_pred,
                use_original_size=use_original_size,
            )
            results[cd["name"]] = m
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(results, f, indent=2, default=float)
            print(f"[save] cross-domain metrics -> {save_path}")
        return results
