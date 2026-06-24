"""训练 TopoSAM-Crack Student（弱监督，离线 SAM3 pseudo .npz）。"""
import argparse
import json

from datasets import build_loader
from engine import Trainer
from utils import load_yaml, set_seed, count_params
from models import build_student


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_yaml(args.config)
    set_seed(cfg["experiment"]["seed"])

    # ---- data ----
    train_loader = build_loader(
        cfg, cfg["data"]["train_manifest"], split="train",
        mode="weakly_pseudo", pseudo_root=cfg["data"]["pseudo_cache"],
    )

    val_loader = None
    with open(cfg["data"]["val_manifest"]) as f:
        val_split_items = json.load(f)["splits"].get("val", [])
    if val_split_items:
        val_loader = build_loader(
            cfg, cfg["data"]["val_manifest"], split="val",
            mode="eval", batch_size=1, shuffle=False,
        )

    # ---- 打印 student 参数量 ----
    print(f"[student] trainable params: "
          f"{count_params(build_student(cfg))/1e6:.2f} M")

    trainer = Trainer(cfg, train_loader, val_loader)
    trainer.fit()


if __name__ == "__main__":
    main()
