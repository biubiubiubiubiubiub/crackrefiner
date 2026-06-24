"""
为 5 个数据集生成统一 JSON manifest。

约定：manifest 里的相对路径以 **公共 data 根**（即 `cfg.data.root`）为基准，
      而不是某个数据集自己的子目录。这样：
        cfg.data.root = ./data
        rec.image     = CRACK500/traincrop/xxx.jpg
      Dataset 内部拼接出: ./data/CRACK500/traincrop/xxx.jpg ✓

用法（推荐 01 脚本自动调）：
  python -m datasets.build_manifest \
         --dataset CRACK500 --root ./data/CRACK500 --out manifests/CRACK500.json
"""
import argparse
import json
import os
from pathlib import Path


def _pair(img_dir: Path, gt_dir: Path, img_exts=("jpg", "png"), gt_ext="png"):
    """匹配 img/gt 同名对。"""
    items = []
    for ext in img_exts:
        for ip in sorted(img_dir.glob(f"*.{ext}")):
            gp = gt_dir / f"{ip.stem}.{gt_ext}"
            if gp.exists():
                items.append((str(ip), str(gp)))
    return items


def _entry(image, gt, data_root, has_gt=True,
           image_level_label=1, use_gt_in_training=False):
    """relpath 相对 *公共* data 根（不是各数据集自己的目录）。"""
    return {
        "image": os.path.relpath(image, data_root),
        "gt":    os.path.relpath(gt, data_root) if has_gt else None,
        "image_level_label": int(image_level_label),
        "source_image_id":   Path(image).stem.split("_")[0],
        "use_gt_in_training": use_gt_in_training,
    }


def build_crack500(root: Path, data_root: Path):
    """traincrop/ valcrop/ testcrop/ 每个目录里 img.jpg 与 mask.png 同名。"""
    splits = {}
    for split, sub in [("train", "traincrop"), ("val", "valcrop"), ("test", "testcrop")]:
        d = root / sub
        if not d.exists():
            print(f"[warn] {d} not found"); continue
        items = []
        for ip in sorted(d.glob("*.jpg")):
            gp = ip.with_suffix(".png")
            if gp.exists():
                items.append(_entry(str(ip), str(gp), str(data_root)))
        splits[split] = items
    return splits


def build_deepcrack(root: Path, data_root: Path):
    return {
        "train": [_entry(i, g, str(data_root))
                  for i, g in _pair(root / "train_img", root / "train_lab")],
        "test":  [_entry(i, g, str(data_root))
                  for i, g in _pair(root / "test_img",  root / "test_lab")],
    }


def build_cfd(root: Path, data_root: Path):
    items = [_entry(i, g, str(data_root))
             for i, g in _pair(root / "image", root / "gt")]
    return {"test": items}      # 跨域评估


def build_crackseg(root: Path, data_root: Path):
    """
    Ultralytics Crack-seg 数据集 (3717 / 112 / 200)。
    标注是 YOLO polygon 格式 (labels/*.txt)，
    Dataset 会按需把 polygon 转成 pixel mask（见 crack_dataset.py）。
    """
    splits = {}
    for split in ["train", "val", "test"]:
        img_dir = root / "images" / split
        lab_dir = root / "labels" / split
        if not img_dir.exists():
            print(f"[warn] {img_dir} not found"); continue
        items = []
        for ip in sorted(img_dir.glob("*.jpg")):
            # 标注文件与图像同名（.txt 替换 .jpg）
            tp = lab_dir / f"{ip.stem}.txt"
            if tp.exists():
                items.append(_entry(str(ip), str(tp), str(data_root)))
        splits[split] = items
    return splits


BUILDERS = {
    "CRACK500":  build_crack500,
    "DeepCrack": build_deepcrack,
    "CFD":       build_cfd,
    "CrackSeg":  build_crackseg,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(BUILDERS.keys()))
    p.add_argument("--root", required=True,
                   help="该数据集自己的根目录，如 ./data/CRACK500")
    p.add_argument("--data_root", default=None,
                   help="公共 data 根；默认取 --root 的父目录")
    p.add_argument("--out", required=True, help="output manifest .json")
    p.add_argument("--weakly_supervised", action="store_true", default=True)
    p.add_argument("--auto_split", nargs=3, type=float, default=None,
                   metavar=("TRAIN", "VAL", "TEST"),
                   help="对没有官方划分的数据集做自动划分，例如 --auto_split 0.8 0.1 0.1")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    root = Path(args.root).resolve()
    assert root.exists(), f"{root} not found"
    data_root = Path(args.data_root).resolve() if args.data_root else root.parent
    print(f"[info] dataset_root = {root}")
    print(f"[info] data_root    = {data_root}")

    splits = BUILDERS[args.dataset](root, data_root)

    # ----- 自动划分（适用于只有 test 一个 split 的数据集）-----
    if args.auto_split is not None:
        import random
        all_items = []
        for v in splits.values():
            all_items.extend(v)
        # 去重（防止 builder 同时把同一张放到多个 split）
        seen, uniq = set(), []
        for it in all_items:
            if it["image"] not in seen:
                seen.add(it["image"]); uniq.append(it)
        rng = random.Random(args.seed)
        rng.shuffle(uniq)
        n = len(uniq)
        r_tr, r_va, r_te = args.auto_split
        s = r_tr + r_va + r_te
        n_tr = int(round(n * r_tr / s))
        n_va = int(round(n * r_va / s))
        n_te = n - n_tr - n_va
        splits = {
            "train": uniq[:n_tr],
            "val":   uniq[n_tr:n_tr + n_va],
            "test":  uniq[n_tr + n_va:],
        }
        print(f"[auto_split] n={n}  train={n_tr}  val={n_va}  test={n_te}  "
              f"seed={args.seed}")

    manifest = {
        "dataset": args.dataset,
        "version": "v1.2",
        "data_root": str(data_root),
        "dataset_root": str(root),
        "weakly_supervised": bool(args.weakly_supervised),
        "auto_split": args.auto_split,
        "seed": args.seed,
        "splits": splits,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[ok] {args.dataset}: " +
          ", ".join(f"{k}={len(v)}" for k, v in splits.items()))
    if splits:
        for k, v in splits.items():
            if v:
                print(f"[ok] sample image path: {v[0]['image']}")
                break
    print(f"[ok] saved to {args.out}")


if __name__ == "__main__":
    main()
