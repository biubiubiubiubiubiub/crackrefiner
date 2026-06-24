from __future__ import annotations
import os
import random
import yaml
import numpy as np
import torch


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
