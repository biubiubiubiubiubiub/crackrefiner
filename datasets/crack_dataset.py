"""
统一的 Manifest 驱动 Dataset：

- mode='supervised'        : 返回 (image, gt)
- mode='weakly_pseudo'     : 返回 (image, pseudo_mask, conf_map, mc_var, gt) ，gt 仅用于评估
- mode='eval'              : 返回 (image, gt, meta)

Pseudo cache 路径约定：
    <pseudo_root>/<image_id>.npz   # 含 mask uint8, conf float16, var float16
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ---------- IO ----------
def _load_image(path: str, size: int) -> np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"image not found: {path}")
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"cv2 failed to decode image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    return img


def _load_mask(path: str, size: int) -> np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"mask not found: {path}")
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise IOError(f"cv2 failed to decode mask: {path}")
    m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.float32)
    
def _load_polygon_mask(path: str, img_h: int, img_w: int,
                       resize_to: int) -> np.ndarray:
    """
    把 YOLO polygon 标注 (.txt) 转为 pixel mask。
    txt 格式: 每行 "class x1 y1 x2 y2 ..."，坐标归一化到 [0,1]。

    返回与 image resize 后尺寸一致的 binary mask (float32, 0/1)。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"polygon label not found: {path}")
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:    # 至少 1 class + 3 个点 (6 坐标)
                continue
            # 跳过 class id（第 0 个），取后面的归一化坐标
            coords = [float(v) for v in parts[1:]]
            # 转成像素坐标
            pts = []
            for i in range(0, len(coords) - 1, 2):
                x = int(round(coords[i]     * img_w))
                y = int(round(coords[i + 1] * img_h))
                pts.append([x, y])
            if len(pts) >= 3:
                cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], color=1)
    # resize 到训练 size
    mask = cv2.resize(mask, (resize_to, resize_to),
                      interpolation=cv2.INTER_NEAREST)
    return mask.astype(np.float32)

# ---------- 数据增强（轻量） ----------
def _aug(image: np.ndarray, masks: Dict[str, np.ndarray], train: bool):
    if not train:
        return image, masks
    if np.random.rand() < 0.5:
        image = image[:, ::-1].copy()
        masks = {k: v[:, ::-1].copy() for k, v in masks.items()}
    if np.random.rand() < 0.5:
        image = image[::-1, :].copy()
        masks = {k: v[::-1, :].copy() for k, v in masks.items()}
    if np.random.rand() < 0.3:                       # 颜色抖动
        image = image.astype(np.float32)
        image *= np.random.uniform(0.8, 1.2)
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image, masks


# ---------- Dataset ----------
class CrackManifestDataset(Dataset):
    """Manifest-driven dataset for cracks."""

    def __init__(
        self,
        manifest_path: str,
        data_root: str,
        split: str = "train",
        image_size: int = 512,
        mode: str = "supervised",          # supervised | weakly_pseudo | eval
        pseudo_root: Optional[str] = None, # 仅 weakly_pseudo 用
        train_aug: bool = True,
        # ---- 弱监督专用：过滤异常伪标签 ----
        min_pseudo_ratio: float = 0.0005,  # 0.05%：低于此值视为"全空"，跳过
        max_pseudo_ratio: float = 0.35,    # 35%：高于此值视为 hallucination，跳过
    ):
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        self.data_root = data_root
        items_all: List[Dict[str, Any]] = self.manifest["splits"].get(split, [])
        if not items_all:
            raise ValueError(f"[{manifest_path}] split={split} 为空")
        self.image_size = image_size
        self.mode = mode
        self.pseudo_root = pseudo_root
        self.train_aug = train_aug and split == "train"

        # 弱监督模式下：扫一遍 pseudo cache 过滤异常样本
        self.items: List[Dict[str, Any]] = items_all
        if mode == "weakly_pseudo" and pseudo_root is not None and split == "train":
            kept, drop_empty, drop_full, drop_miss = [], 0, 0, 0
            for rec in items_all:
                pth = self._pseudo_path_static(pseudo_root, rec["image"])
                if not os.path.isfile(pth):
                    drop_miss += 1
                    continue
                try:
                    m = np.load(pth)["mask"]
                    r = float(m.mean()) if m.size else 0.0
                except Exception:
                    drop_miss += 1
                    continue
                if r < min_pseudo_ratio:
                    drop_empty += 1
                    continue
                if r > max_pseudo_ratio:
                    drop_full += 1
                    continue
                kept.append(rec)
            self.items = kept
            print(f"[Dataset filter] kept={len(kept)}/{len(items_all)}  "
                  f"drop_empty={drop_empty}  drop_full={drop_full}  "
                  f"drop_miss={drop_miss}  "
                  f"(thr=[{min_pseudo_ratio:.4f}, {max_pseudo_ratio:.2f}])")

        print(f"[Dataset] {self.manifest['dataset']}/{split} "
              f"n={len(self.items)} mode={mode}")

    @staticmethod
    def _pseudo_path_static(pseudo_root: str, image_rel: str) -> str:
        key = Path(image_rel).with_suffix("").as_posix().replace("/", "__")
        return os.path.join(pseudo_root, f"{key}.npz")

    def __len__(self):
        return len(self.items)

    def _pseudo_path(self, image_rel: str) -> str:
        assert self.pseudo_root is not None
        return self._pseudo_path_static(self.pseudo_root, image_rel)

    def __getitem__(self, idx):
        rec = self.items[idx]
        img_path = os.path.join(self.data_root, rec["image"])
        image = _load_image(img_path, self.image_size)

        gt = None
        if rec.get("gt"):
            gt_path = os.path.join(self.data_root, rec["gt"])
            try:
                # 自动识别：.txt 走 polygon，.png/.jpg 走 mask
                if gt_path.endswith(".txt"):
                    # 需要原始图像尺寸来计算 polygon → mask
                    _img_raw = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    H0, W0 = _img_raw.shape[:2]
                    gt = _load_polygon_mask(gt_path, H0, W0, self.image_size)
                else:
                    gt = _load_mask(gt_path, self.image_size)
            except Exception as e:
                print(f"[warn] gt load failed for {gt_path}: {e}")
                gt = None
        
        if self.mode == "supervised":
            assert gt is not None
            image, masks = _aug(image, {"gt": gt}, self.train_aug)
            return self._to_tensor(image, masks=masks, meta=rec)

        if self.mode == "weakly_pseudo":
            data = np.load(self._pseudo_path(rec["image"]))
            pseudo = cv2.resize(data["mask"].astype(np.float32),
                                (self.image_size, self.image_size),
                                interpolation=cv2.INTER_NEAREST)
            conf = cv2.resize(data["conf"].astype(np.float32),
                              (self.image_size, self.image_size),
                              interpolation=cv2.INTER_LINEAR)
            var = cv2.resize(data["var"].astype(np.float32),
                             (self.image_size, self.image_size),
                             interpolation=cv2.INTER_LINEAR)
            packed = {"pseudo": pseudo, "conf": conf, "var": var}
            if gt is not None:
                packed["gt"] = gt
            image, packed = _aug(image, packed, self.train_aug)
            return self._to_tensor(image, masks=packed, meta=rec)

        if self.mode == "eval":
            assert gt is not None
            return self._to_tensor(image, masks={"gt": gt}, meta=rec)

        if self.mode == "eval_original":
            assert rec.get("gt"), "eval_original needs gt"
            img_orig_path = os.path.join(self.data_root, rec["image"])
            img_full = cv2.imread(img_orig_path, cv2.IMREAD_COLOR)
            img_full = cv2.cvtColor(img_full, cv2.COLOR_BGR2RGB)
            H0, W0 = img_full.shape[:2]
            
            # ★ 这里改：支持 polygon
            gt_path = os.path.join(self.data_root, rec["gt"])
            if gt_path.endswith(".txt"):
                # polygon → mask（直接到原图尺寸）
                gt_full = np.zeros((H0, W0), dtype=np.uint8)
                with open(gt_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 7: continue
                        coords = [float(v) for v in parts[1:]]
                        pts = [[int(round(coords[i] * W0)),
                                int(round(coords[i + 1] * H0))]
                               for i in range(0, len(coords) - 1, 2)]
                        if len(pts) >= 3:
                            cv2.fillPoly(gt_full, [np.array(pts, dtype=np.int32)], 1)
                gt_full = gt_full.astype(np.float32)
            else:
                gt_full = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                gt_full = (gt_full > 127).astype(np.float32)
            
            # 图像 resize 到 image_size 推理
            img_t = cv2.resize(img_full, (self.image_size, self.image_size),
                               interpolation=cv2.INTER_LINEAR)
            img_t = torch.from_numpy(img_t).permute(2, 0, 1).float() / 255.0
            img_t = (img_t - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) \
                  /  torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            gt_t = torch.from_numpy(gt_full).float().unsqueeze(0)
            return {"image": img_t, "gt": gt_t, "meta": rec,
                    "orig_size": (H0, W0)}

    @staticmethod
    def _to_tensor(image, masks, meta):
        img_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        img_t = (img_t - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) \
              /  torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        out = {"image": img_t, "meta": meta}
        for k, v in masks.items():
            out[k] = torch.from_numpy(v).float().unsqueeze(0)  # 1xHxW
        return out


def build_loader(cfg, manifest_path, split, mode,
                 pseudo_root=None, batch_size=None, shuffle=None):
    ds = CrackManifestDataset(
        manifest_path=manifest_path,
        data_root=cfg["data"]["root"],
        split=split,
        image_size=cfg["data"]["image_size"],
        mode=mode,
        pseudo_root=pseudo_root,
        min_pseudo_ratio=cfg["data"].get("min_pseudo_ratio", 0.0005),
        max_pseudo_ratio=cfg["data"].get("max_pseudo_ratio", 0.35),
    )
    bs = batch_size if batch_size is not None else cfg["train"]["batch_size"]
    sh = shuffle if shuffle is not None else (split == "train")
    return DataLoader(
        ds, batch_size=bs, shuffle=sh,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True, drop_last=(split == "train"),
    )
