#!/usr/bin/env bash
set -e
export PYTHONPATH="$(pwd):${PYTHONPATH}"
# 减显存碎片，跑长序列 SSM 必备
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ★ 写死 TopoSAM 主训练 config
CONFIG="configs/toposam_crack500.yaml"
echo "[info] using CONFIG=$CONFIG"
echo "[info] expected output_dir=outputs/toposam_crack500"

python -m scripts_py.run_train --config $CONFIG
