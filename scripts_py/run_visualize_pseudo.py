"""
可视化伪标签：把 image | SAM3 mask | conf heatmap | overlay | GT 拼到一张大图。
用法：
  python -m scripts_py.run_visualize_pseudo \
         --config configs/toposam_crack500.yaml \
         --manifest manifests/CRACK500.json \
         --split train --n 8
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

from utils import load_yaml


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


def _overlay(rgb: np.ndarray, mask01: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    color = np.zeros_like(rgb); color[..., 0] = 255          # 红
    m = (mask01 > 0.5).astype(np.uint8)[:, :, None]
    return (rgb * (1 - alpha * m) + color * (alpha * m)).astype(np.uint8)


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",   required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--split",    default="train")
    p.add_argument("--n",        type=int, default=8)
    p.add_argument("--shuffle",  action="store_true",
                   help="随机抽 n 张而不是前 n 张")
    p.add_argument("--seed",     type=int, default=0)
    p.add_argument("--out_dir",  default="outputs/viz/pseudo")
    p.add_argument("--size",     type=int, default=384, help="单格 resize 尺寸")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    data_root = cfg["data"]["root"]
    pseudo_root = cfg["data"]["pseudo_cache"]
    with open(args.manifest) as f:
        man = json.load(f)
    items_all = man["splits"].get(args.split, [])
    if args.shuffle:
        import random
        random.seed(args.seed)
        items = random.sample(items_all, min(args.n, len(items_all)))
    else:
        items = items_all[: args.n]
    if not items:
        print("[empty]"); return

    os.makedirs(args.out_dir, exist_ok=True)
    cols = 4 if not any(rec.get("gt") for rec in items) else 5
    rows = []
    for rec in items:
        ip = os.path.join(data_root, rec["image"])
        rgb = cv2.imread(ip, cv2.IMREAD_COLOR)
        if rgb is None:
            print(f"[skip] cannot read {ip}"); continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]

        npz_path = os.path.join(pseudo_root, f"{_key_from_image(rec['image'])}.npz")
        if not os.path.isfile(npz_path):
            print(f"[skip] no pseudo: {npz_path}"); continue
        d = np.load(npz_path)
        mask = d["mask"].astype(np.float32)
        conf = d["conf"].astype(np.float32)

        if mask.shape != (H, W):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            conf = cv2.resize(conf, (W, H), interpolation=cv2.INTER_LINEAR)

        cells = [
            _label(rgb, f"image  {Path(rec['image']).name}"),
            _label(cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB),
                   f"SAM3 mask  mean={mask.mean():.4f}"),
            _label(cv2.cvtColor(_heatmap(conf), cv2.COLOR_BGR2RGB),
                   f"conf  max={conf.max():.3f}"),
            _label(_overlay(rgb, mask), "overlay (red=SAM3)"),
        ]
        if rec.get("gt"):
            gp = os.path.join(data_root, rec["gt"])
            gt = cv2.imread(gp, cv2.IMREAD_GRAYSCALE)
            if gt is not None:
                gt01 = (gt > 127).astype(np.float32)
                if gt01.shape != (H, W):
                    gt01 = cv2.resize(gt01, (W, H), interpolation=cv2.INTER_NEAREST)
                cells.append(
                    _label(_overlay(rgb, gt01, alpha=0.55), "GT overlay (red=truth)")
                )

        # 统一 resize 到 args.size
        cells = [cv2.resize(c, (args.size, args.size)) for c in cells]
        rows.append(np.concatenate(cells, axis=1))

    if not rows:
        print("[no output]"); return
    grid = np.concatenate(rows, axis=0)
    out_path = os.path.join(args.out_dir,
                            f"{man['dataset']}_{args.split}_n{len(rows)}.jpg")
    cv2.imwrite(out_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print(f"[ok] {out_path}  ({grid.shape[1]}x{grid.shape[0]})")


if __name__ == "__main__":
    main()
