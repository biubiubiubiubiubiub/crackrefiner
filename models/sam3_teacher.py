"""
SAM3 Teacher wrapper（v2，修复版）：

关键修复
--------
v1 错误：把所有 nn.Dropout 强行设为 train() 试图做 MC sampling，
        但 SAM3 的 Dropout 在 attention/MLP/decoder 里都是结构性的，
        强行 train() 会把 instance scores 全部打到阈值之下，
        造成 mask 全 0。  → 这就是之前 mask.mean()=0 的根因。

v2 做法：
  • 全程 eval（与 labelme 行为一致），保证 prompt='crack' 能正确输出实例；
  • "不确定性" 改用 prompt ensemble + (optional) tile-flip TTA：
        var = Var_{prompts}( prob_map(prompt) )
  • prob_map 由多实例 mask 按 score 加权后做 pixel-wise max 得到。

真实 SAM3 API：
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    state  = processor.set_image(pil)
    output = processor.set_text_prompt(state=state, prompt='crack')
    masks, boxes, scores = output['masks'], output['boxes'], output['scores']
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import warnings

import cv2
import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# ---------------- helpers ----------------
def _to_pil(rgb: np.ndarray) -> Image.Image:
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _masks_to_np(masks, H: int, W: int) -> np.ndarray:
    """SAM3 输出的 masks → (N,H,W) float32 in [0,1]。"""
    if masks is None:
        return np.zeros((0, H, W), dtype=np.float32)
    if torch.is_tensor(masks):
        m = masks.detach().cpu().float().numpy()
    else:
        m = np.asarray(masks, dtype=np.float32)
    if m.ndim == 2:
        m = m[None]
    elif m.ndim == 4:
        m = m[:, 0]
    if m.size == 0:
        return np.zeros((0, H, W), dtype=np.float32)
    if m.shape[-2:] != (H, W):
        out = np.zeros((m.shape[0], H, W), dtype=np.float32)
        for i in range(m.shape[0]):
            out[i] = cv2.resize(m[i], (W, H),
                                interpolation=cv2.INTER_LINEAR)
        m = out
    return m.clip(0.0, 1.0)


def _scores_to_np(scores, n: int) -> np.ndarray:
    if scores is None:
        return np.ones(n, dtype=np.float32)
    if torch.is_tensor(scores):
        s = scores.detach().cpu().float().numpy()
    else:
        s = np.asarray(scores, dtype=np.float32)
    s = s.reshape(-1).clip(0.0, 1.0)
    if len(s) != n:
        return np.ones(n, dtype=np.float32)
    return s


def _ensure_all_eval(model: torch.nn.Module):
    """保证模型完全 eval；任何模块（含 Dropout）都不在 train 模式。"""
    model.eval()
    for m in model.modules():
        if m.training:
            m.eval()


# ---------------- main class ----------------
class SAM3Teacher:
    """SAM3 包装：text prompt(ensemble) → per-pixel crack prob + ensemble variance。"""

    def __init__(self,
                 ckpt: Optional[str] = None,
                 text_prompts: List[str] = ("crack",),
                 mc_samples: int = 1,          # 兼容老接口；v2 已不用 MC dropout
                 device: str = "cuda",
                 resolution: int = 1008,
                 confidence_threshold: float = 0.3,
                 load_from_HF: bool = False,
                 bpe_path: Optional[str] = None,
                 # ---- 新增 ----
                 ensemble_prompts: Optional[List[str]] = None,
                 enable_hflip_tta: bool = False,
                 autocast_dtype: str = "bfloat16",
                 binary_threshold: float = 0.5,
                 score_weight: bool = True,
                 max_mask_area_ratio: float = 0.30,   # 丢掉占图 >30% 的实例
                 min_mask_area_px: int   = 16):       # 丢掉 <16 像素的噪点
        self.device = device
        self.text_prompts = list(text_prompts) if text_prompts else ["crack"]
        self.confidence_threshold = float(confidence_threshold)
        self.binary_threshold = float(binary_threshold)
        self.score_weight = bool(score_weight)
        self.max_mask_area_ratio = float(max_mask_area_ratio)
        self.min_mask_area_px = int(min_mask_area_px)

        # ensemble_prompts：用于离线伪标签生成时同时跑多个等价 prompt，
        # 取 *prob 的 mean* 作为 conf_map，取 *prob 的 var* 作为 uncertainty。
        if ensemble_prompts is None or len(ensemble_prompts) == 0:
            self.ensemble_prompts = list(self.text_prompts)
        else:
            self.ensemble_prompts = list(ensemble_prompts)

        if mc_samples > 1:
            warnings.warn(
                "[SAM3Teacher] mc_samples>1 已在 v2 中废弃（会破坏 SAM3 推理）。"
                "请改用 ensemble_prompts 获取不确定性。"
            )

        self.enable_hflip_tta = bool(enable_hflip_tta)

        # autocast dtype
        self.autocast_dtype = {
            "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
            "float16":  torch.float16,  "fp16": torch.float16,
            "float32":  torch.float32,  "fp32": torch.float32,
        }.get(autocast_dtype.lower(), torch.bfloat16)

        print(f"[SAM3Teacher v2] build_sam3_image_model("
              f"checkpoint_path={ckpt}, load_from_HF={load_from_HF}, "
              f"device={device})")
        self.model = build_sam3_image_model(
            bpe_path=bpe_path,
            device=device,
            eval_mode=True,
            checkpoint_path=ckpt,
            load_from_HF=load_from_HF,
            enable_segmentation=True,
            enable_inst_interactivity=False,
            compile=False,
        )
        _ensure_all_eval(self.model)
        self.processor = Sam3Processor(
            self.model,
            resolution=resolution,
            device=device,
            confidence_threshold=self.confidence_threshold,
        )
        self.backend = "sam3"
        print(f"[SAM3Teacher v2] text_prompts={self.text_prompts}  "
              f"ensemble_prompts={self.ensemble_prompts}  "
              f"conf_thr={self.confidence_threshold}  "
              f"bin_thr={self.binary_threshold}  "
              f"score_weight={self.score_weight}  "
              f"max_area_ratio={self.max_mask_area_ratio}  "
              f"min_area_px={self.min_mask_area_px}  "
              f"hflip_tta={self.enable_hflip_tta}  "
              f"autocast={self.autocast_dtype}")

    # ---------------------------------------------------------------
    def _autocast(self):
        """SAM3 训练/推理默认 bf16，必须用 autocast 包住所有 backbone 调用，
        否则会出现 'mat1 and mat2 must have the same dtype, BFloat16 vs Float'。"""
        if self.autocast_dtype == torch.float32 or self.device == "cpu":
            # 关掉 autocast 的等价上下文管理器
            from contextlib import nullcontext
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.autocast_dtype)

    # ---------------------------------------------------------------
    @torch.no_grad()
    def _predict_prob(self, rgb: np.ndarray, prompt: str) -> np.ndarray:
        """对一张 rgb 用单个 text prompt 跑一次，返回 score-weighted union prob (H,W)."""
        H, W = rgb.shape[:2]
        pil = _to_pil(rgb)
        try:
            with self._autocast():
                state = self.processor.set_image(pil)
                out = self.processor.set_text_prompt(state=state, prompt=prompt)
        except Exception as e:
            warnings.warn(f"[SAM3] predict failed prompt={prompt!r}: {e}")
            return np.zeros((H, W), dtype=np.float32)
        masks_np = _masks_to_np(out.get("masks") if isinstance(out, dict) else None,
                                H, W)
        if masks_np.shape[0] == 0:
            return np.zeros((H, W), dtype=np.float32)
        s = _scores_to_np(out.get("scores") if isinstance(out, dict) else None,
                          masks_np.shape[0])

        # ---- 面积过滤：丢掉过大（整图）或过小（噪点）的实例 ----
        img_area = H * W
        keep_idx = []
        for i in range(masks_np.shape[0]):
            area = int((masks_np[i] > 0.5).sum())
            ratio = area / max(1, img_area)
            if ratio > self.max_mask_area_ratio:
                continue       # 整图巨型实例，丢
            if area < self.min_mask_area_px:
                continue       # 噪点，丢
            keep_idx.append(i)
        if not keep_idx:
            return np.zeros((H, W), dtype=np.float32)
        masks_np = masks_np[keep_idx]
        s = s[keep_idx]

        if self.score_weight:
            return (masks_np * s[:, None, None]).max(axis=0).astype(np.float32)
        else:
            return masks_np.max(axis=0).astype(np.float32)

    # ---------------------------------------------------------------
    @torch.no_grad()
    def infer(self, rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        离线伪标签生成。
        prob 由 ensemble_prompts × (optional hflip TTA) 取平均；
        var  由 ensemble 内部方差估计。
        """
        H, W = rgb.shape[:2]
        runs: List[np.ndarray] = []
        for prompt in self.ensemble_prompts:
            runs.append(self._predict_prob(rgb, prompt))
            if self.enable_hflip_tta:
                rgb_flip = rgb[:, ::-1].copy()
                p_flip = self._predict_prob(rgb_flip, prompt)
                runs.append(p_flip[:, ::-1].copy())

        if not runs:
            return (np.zeros((H, W), np.uint8),
                    np.zeros((H, W), np.float16),
                    np.zeros((H, W), np.float16))
        stack = np.stack(runs, 0)                  # (K,H,W)
        mean_p = stack.mean(0)
        var_p = stack.var(0) if stack.shape[0] > 1 else np.zeros_like(mean_p)
        mask = (mean_p > self.binary_threshold).astype(np.uint8)
        return mask, mean_p.astype(np.float16), var_p.astype(np.float16)

    # ===============================================================
    #  PromptLoop-Crack 扩展接口（保持与 v1 相同签名）
    # ===============================================================
    @torch.no_grad()
    def infer_with_prompts(
        self,
        rgb: np.ndarray,
        points: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        bbox: Optional[np.ndarray] = None,
        text: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        H, W = rgb.shape[:2]
        text = text or (self.text_prompts[0] if self.text_prompts else "crack")
        prob = self._predict_prob(rgb, text)

        if bbox is not None:
            x1, y1, x2, y2 = [int(round(v)) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W - 1, x2), min(H - 1, y2)
            if x2 > x1 and y2 > y1:
                roi = np.zeros_like(prob)
                roi[y1:y2 + 1, x1:x2 + 1] = 1.0
                prob = prob * roi
        bin_ = (prob > self.binary_threshold).astype(np.uint8)
        return bin_, prob

    @torch.no_grad()
    def batch_infer_with_prompts(
        self,
        rgb_batch: np.ndarray,
        points_batch: Optional[np.ndarray],
        point_labels_batch: Optional[np.ndarray],
        bbox_batch: Optional[np.ndarray],
        text: Optional[str] = None,
    ):
        B = rgb_batch.shape[0]
        bins, probs = [], []
        for b in range(B):
            m, p = self.infer_with_prompts(
                rgb_batch[b],
                points=None if points_batch is None else points_batch[b],
                point_labels=None if point_labels_batch is None else point_labels_batch[b],
                bbox=None if bbox_batch is None else bbox_batch[b],
                text=text,
            )
            bins.append(m); probs.append(p)
        return np.stack(bins, 0), np.stack(probs, 0)
