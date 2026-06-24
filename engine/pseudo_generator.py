"""
对一个 manifest 的指定 split 用 SAM3 离线生成伪标签。
输出每张图一个 .npz: { mask: uint8, conf: float16, var: float16 }
路径：<pseudo_root>/<image_rel_no_ext>.npz （'/'->'__'）
"""
from __future__ import annotations
import json
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def _key_from_image(image_rel: str) -> str:
    return Path(image_rel).with_suffix("").as_posix().replace("/", "__")


def generate_pseudo_for_dataset(manifest_path: str,
                                data_root: str,
                                split: str,
                                out_dir: str,
                                teacher,
                                image_size: int | None = None,
                                overwrite: bool = False):
    with open(manifest_path) as f:
        man = json.load(f)
    items = man["splits"].get(split, [])
    if not items:
        print(f"[pseudo] empty split={split}")
        return
    os.makedirs(out_dir, exist_ok=True)
    print(f"[pseudo] generating {len(items)} pseudo labels for "
          f"{man['dataset']}/{split} -> {out_dir}")

    n_ok, n_miss, n_skip = 0, 0, 0
    for rec in tqdm(items):
        out_path = os.path.join(out_dir, f"{_key_from_image(rec['image'])}.npz")
        if (not overwrite) and os.path.exists(out_path):
            n_skip += 1
            continue
        img_path = os.path.join(data_root, rec["image"])
        if not os.path.isfile(img_path):
            print(f"[miss] {img_path}")
            n_miss += 1
            continue
        rgb = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if rgb is None:
            print(f"[decode_fail] {img_path}")
            n_miss += 1
            continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        if image_size:
            rgb = cv2.resize(rgb, (image_size, image_size),
                             interpolation=cv2.INTER_LINEAR)
        mask, conf, var = teacher.infer(rgb)
        np.savez_compressed(out_path, mask=mask, conf=conf, var=var)
        n_ok += 1
    print(f"[pseudo] done.  ok={n_ok}  miss={n_miss}  cached_skip={n_skip}")
    if n_ok == 0 and n_miss > 0:
        raise RuntimeError(
            "全部图像都加载失败！请检查 manifest 的 image 字段是否与 data_root 匹配。"
            "若旧 manifest 缺少 'CRACK500/' 等数据集前缀，请重跑 scripts/01_build_manifests.sh"
        )
