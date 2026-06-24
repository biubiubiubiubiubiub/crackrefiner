"""
统计 SAM3 伪标签的全局分布，用来判断质量与挑选阈值。
用法:
  python -m scripts_py.stat_pseudo --pseudo_dir outputs/pseudo/CRACK500
"""
import argparse
import glob
import os

import numpy as np
from tqdm import tqdm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pseudo_dir", required=True)
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.pseudo_dir, "*.npz")))
    if not files:
        print("[no files]"); return
    print(f"[stat] {len(files)} npz files in {args.pseudo_dir}")

    ratios, conf_max, conf_mean, var_mean, n_empty, n_full = [], [], [], [], 0, 0
    for f in tqdm(files):
        d = np.load(f)
        m = d["mask"].astype(np.float32)
        c = d["conf"].astype(np.float32)
        v = d["var"].astype(np.float32)
        r = m.mean()
        ratios.append(r); conf_max.append(c.max()); conf_mean.append(c.mean())
        var_mean.append(v.mean())
        if r < 1e-4:        n_empty += 1
        elif r > 0.3:       n_full  += 1
    ratios = np.asarray(ratios)
    conf_max = np.asarray(conf_max); conf_mean = np.asarray(conf_mean)
    var_mean = np.asarray(var_mean)

    def stat(name, arr):
        print(f"  {name:14s}  mean={arr.mean():.4f}  median={np.median(arr):.4f}  "
              f"min={arr.min():.4f}  max={arr.max():.4f}  std={arr.std():.4f}")

    print(f"\n=== 全量统计 ===")
    stat("mask_ratio",  ratios)
    stat("conf_max",    conf_max)
    stat("conf_mean",   conf_mean)
    stat("var_mean",    var_mean)

    print(f"\n=== 极端样本 ===")
    print(f"  完全空的 (ratio<1e-4) : {n_empty} ({100*n_empty/len(files):.2f}%)")
    print(f"  过密 (ratio>0.3)      : {n_full}  ({100*n_full /len(files):.2f}%)")

    print(f"\n=== 分位数（指导挑阈值）===")
    for q in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  P{q:2d} mask_ratio: {np.percentile(ratios, q):.4f}")


if __name__ == "__main__":
    main()
