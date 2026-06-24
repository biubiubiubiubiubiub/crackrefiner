"""评估（论文协议版）。

默认使用论文标准评估协议：
  • 原图尺寸评估 (use_original_size=True)
  • ±2 px 容差 (tolerance=2)
  • 不细化预测 (thin_pred=False)

也提供 4 种模式对比：
  --protocol paper      ← 默认  (orig + tol=2, thin=False)
  --protocol strict     ← 裸像素 (resize512 + tol=0)
  --protocol skel       ← skeleton 细化后比较 (orig + tol=2, thin=True)
  --protocol legacy     ← 旧版兼容 (resize512 + tol=0)
"""
import argparse
import json
import os
from utils import load_yaml
from engine import Evaluator


PROTOCOLS = {
    "paper":  dict(tolerance=2, thin_pred=False, use_original_size=True),
    "strict": dict(tolerance=0, thin_pred=False, use_original_size=False),
    "skel":   dict(tolerance=2, thin_pred=True,  use_original_size=True),
    "legacy": dict(tolerance=0, thin_pred=False, use_original_size=False),
}


def _dataset_name_from_manifest(manifest_path: str) -> str:
    try:
        with open(manifest_path) as f:
            return json.load(f).get("dataset", os.path.basename(manifest_path))
    except Exception:
        return os.path.basename(manifest_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--save", default=None)
    p.add_argument("--protocol", choices=list(PROTOCOLS), default="paper")
    p.add_argument("--skip_indomain", action="store_true",
                   help="跳过 in-domain test 评估，只跑 cross-domain")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    opts = PROTOCOLS[args.protocol]
    print(f"[Eval] protocol={args.protocol}  opts={opts}")

    ev = Evaluator(cfg, args.ckpt)

    all_results = {}

    # 1) in-domain test（自动用 train_manifest 的 test split）
    if not args.skip_indomain:
        train_manifest = cfg["data"]["train_manifest"]
        in_name = _dataset_name_from_manifest(train_manifest)
        in_m = ev.evaluate_manifest(train_manifest, "test", in_name, **opts)
        all_results[f"{in_name}_indomain"] = in_m

    # 2) cross-domain
    cd_results = ev.evaluate_crossdomain(save_path=None, **opts)
    all_results.update(cd_results)

    # 3) 总保存
    if args.save:
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
        with open(args.save, "w") as f:
            json.dump(all_results, f, indent=2, default=float)
        print(f"[save] all metrics -> {args.save}")


if __name__ == "__main__":
    main()
