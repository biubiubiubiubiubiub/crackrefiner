"""调用 SAM3 离线生成伪标签 + 置信图 + MC variance。"""
import argparse
import json
import shutil
import tempfile

from utils import load_yaml
from models import SAM3Teacher
from engine import generate_pseudo_for_dataset


def _maybe_subset(manifest_path: str, split: str, max_n: int) -> str:
    """若指定 --max，把 manifest 截取前 max_n 个样本写入临时文件并返回路径。"""
    with open(manifest_path) as f:
        man = json.load(f)
    items = man["splits"].get(split, [])
    if max_n <= 0 or max_n >= len(items):
        return manifest_path
    man = json.loads(json.dumps(man))   # deep copy
    man["splits"][split] = items[:max_n]
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=f".{split}.subset.json", delete=False
    )
    json.dump(man, tmp, indent=2); tmp.close()
    print(f"[subset] {split}: {len(items)} → {max_n} (tmp manifest={tmp.name})")
    return tmp.name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--image_size", type=int, default=None,
                   help="若提供，则 SAM 推理前 resize；None 表示原分辨率")
    p.add_argument("--max", type=int, default=0,
                   help="只跑前 N 个样本（用于冒烟测试）；0 表示全部")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    tcfg = cfg["teacher"]
    teacher = SAM3Teacher(
        ckpt=tcfg.get("ckpt"),
        text_prompts=tcfg["text_prompts"],
        mc_samples=tcfg.get("mc_samples", 1),
        device=tcfg.get("device", "cuda"),
        resolution=tcfg.get("resolution", 1008),
        confidence_threshold=tcfg.get("confidence_threshold", 0.3),
        load_from_HF=tcfg.get("load_from_HF", False),
        bpe_path=tcfg.get("bpe_path", None),
        ensemble_prompts=tcfg.get("ensemble_prompts", None),
        enable_hflip_tta=tcfg.get("enable_hflip_tta", False),
        autocast_dtype=tcfg.get("autocast_dtype", "bfloat16"),
        binary_threshold=tcfg.get("binary_threshold", 0.5),
        score_weight=tcfg.get("score_weight", True),
        max_mask_area_ratio=tcfg.get("max_mask_area_ratio", 0.30),
        min_mask_area_px=tcfg.get("min_mask_area_px", 16),
    )

    mp = _maybe_subset(args.manifest, args.split, args.max)
    generate_pseudo_for_dataset(
        manifest_path=mp,
        data_root=cfg["data"]["root"],
        split=args.split,
        out_dir=cfg["data"]["pseudo_cache"],
        teacher=teacher,
        image_size=args.image_size,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
