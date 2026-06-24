"""
可视化 Student 预测（带 GT 对比，论文 Figure 用）：
  image | SAM3 pseudo | Student prob | Student mask | overlay | GT overlay
用法:
  python -m scripts_py.run_visualize_pred \
       --config configs/toposam_crack500.yaml \
       --ckpt   outputs/toposam_crack500/best.pth \
       --manifest manifests/CRACK500.json \
       --split test --n 8 --shuffle
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from utils import load_yaml
from models import build_student


# ---------- helpers ----------
def _key_from_image(image_rel: str) -> str:
    return Path(image_rel).with_suffix("").as_posix().replace("/", "__")


def _norm_to_u8(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    if a.max() - a.min() < 1e-6:
        return np.zeros_like(a, dtype=np.uint8)
    a = (a - a.min()) / (a.max() - a.min())
    return (a * 255).astype(np.uint8)


def _heatmap(prob: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap(_norm_to_u8(prob), cv2.COLORMAP_JET)


def _overlay(rgb: np.ndarray, mask01: np.ndarray,
             color=(0, 0, 255), alpha=0.55):
    """rgb HxWx3 (RGB), mask01 HxW {0,1}; color in BGR for cv2 array but we operate in RGB."""
    m = (mask01 > 0.5).astype(np.uint8)[:, :, None]
    cl = np.zeros_like(rgb); cl[..., 0] = color[2]; cl[..., 1] = color[1]; cl[..., 2] = color[0]
    return (rgb * (1 - alpha * m) + cl * (alpha * m)).astype(np.uint8)


def _label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ---------- main ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",   required=True)
    p.add_argument("--ckpt",     required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--split",    default="test")
    p.add_argument("--n",        type=int, default=8)
    p.add_argument("--shuffle",  action="store_true")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--out_dir",  default="outputs/viz/pred")
    p.add_argument("--size",     type=int, default=384)
    p.add_argument("--bin_thr",  type=float, default=0.5,
                   help="阈值二值化 student 输出")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    data_root = cfg["data"]["root"]
    pseudo_root = cfg["data"].get("pseudo_cache")
    img_size = cfg["data"]["image_size"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 加载 manifest ----
    with open(args.manifest) as f:
        man = json.load(f)
    items_all = man["splits"].get(args.split, [])
    if not items_all:
        print(f"[empty] split={args.split}"); return
    if args.shuffle:
        import random
        random.seed(args.seed)
        items = random.sample(items_all, min(args.n, len(items_all)))
    else:
        items = items_all[: args.n]

    # ---- 加载模型 ----
    model = build_student(cfg).to(device).eval()
    ck = torch.load(args.ckpt, map_location="cpu")
    state = ck.get("ema", ck["model"])
    model.load_state_dict(state)
    print(f"[Viz] loaded {args.ckpt} (epoch={ck.get('epoch')})")

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for rec in items:
        ip = os.path.join(data_root, rec["image"])
        rgb = cv2.imread(ip, cv2.IMREAD_COLOR)
        if rgb is None:
            print(f"[skip] {ip}"); continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        H0, W0 = rgb.shape[:2]

        # student inference at training size, then upsample to original
        img_resized = cv2.resize(rgb, (img_size, img_size),
                                 interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        x = ((x - mean) / std).unsqueeze(0).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda",
                                                  enabled=device == "cuda"):
            prob = torch.sigmoid(model(x)["mask_logit"])
        prob_full = F.interpolate(prob, size=(H0, W0),
                                  mode="bilinear", align_corners=False)
        prob_np = prob_full[0, 0].float().cpu().numpy()
        pred_bin = (prob_np >= args.bin_thr).astype(np.float32)

        # GT
        gt01 = None
        if rec.get("gt"):
            gtp = os.path.join(data_root, rec["gt"])
            g = cv2.imread(gtp, cv2.IMREAD_GRAYSCALE)
            if g is not None:
                gt01 = (g > 127).astype(np.float32)
                if gt01.shape != (H0, W0):
                    gt01 = cv2.resize(gt01, (W0, H0),
                                      interpolation=cv2.INTER_NEAREST)

        # SAM3 pseudo (optional, only train split usually has cache)
        pseudo_mask = None
        if pseudo_root:
            npz_path = os.path.join(pseudo_root,
                                    f"{_key_from_image(rec['image'])}.npz")
            if os.path.isfile(npz_path):
                d = np.load(npz_path)
                pseudo_mask = d["mask"].astype(np.float32)
                if pseudo_mask.shape != (H0, W0):
                    pseudo_mask = cv2.resize(pseudo_mask, (W0, H0),
                                             interpolation=cv2.INTER_NEAREST)

        # ---- 拼接 6 列 ----
        cells = []
        cells.append(_label(rgb, f"image  {Path(rec['image']).name}"))
        if pseudo_mask is not None:
            cells.append(_label(_overlay(rgb, pseudo_mask, color=(255, 255, 0)),
                                f"SAM3 pseudo  r={pseudo_mask.mean():.4f}"))
        else:
            cells.append(_label(np.zeros_like(rgb), "no pseudo cache"))
        cells.append(_label(cv2.cvtColor(_heatmap(prob_np), cv2.COLOR_BGR2RGB),
                            f"Student prob  max={prob_np.max():.3f}"))
        cells.append(_label(cv2.cvtColor((pred_bin * 255).astype(np.uint8),
                                          cv2.COLOR_GRAY2RGB),
                            f"Student mask  thr={args.bin_thr:.2f}  "
                            f"r={pred_bin.mean():.4f}"))
        cells.append(_label(_overlay(rgb, pred_bin, color=(255, 0, 0)),
                            "Student overlay (red)"))
        if gt01 is not None:
            cells.append(_label(_overlay(rgb, gt01, color=(0, 255, 0)),
                                f"GT overlay (green)  r={gt01.mean():.4f}"))
        else:
            cells.append(_label(np.zeros_like(rgb), "no GT"))

        # resize & concat
        cells = [cv2.resize(c, (args.size, args.size)) for c in cells]
        rows.append(np.concatenate(cells, axis=1))

    if not rows:
        print("[no rows]"); return
    grid = np.concatenate(rows, axis=0)

    name = man.get("dataset", "data")
    out_path = os.path.join(args.out_dir,
                            f"{name}_{args.split}_n{len(rows)}_pred.jpg")
    cv2.imwrite(out_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print(f"[ok] {out_path}  size={grid.shape[1]}x{grid.shape[0]}")


if __name__ == "__main__":
    main()
