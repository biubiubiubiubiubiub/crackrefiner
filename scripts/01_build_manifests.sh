#!/usr/bin/env bash
# 为 3 个数据集生成统一 manifest
#
# 协议：
#   CRACK500   : 官方 train/val/test 划分（1896/348/1124）
#   DeepCrack  : 官方 train/test 划分（300/237）— 跨域评估
#   CFD        : 全部 118 张当 test（跨域）
set -e
export PYTHONPATH="$(pwd):${PYTHONPATH}"

DATA_ROOT=${DATA_ROOT:-./data}
if [ ! -d "$DATA_ROOT" ]; then
  echo "[ERROR] DATA_ROOT=$DATA_ROOT 不存在。"
  exit 1
fi
echo "[info] DATA_ROOT=$DATA_ROOT"

OUT=manifests
mkdir -p $OUT
SEED=42

# ----- 官方划分 / 跨域专用 -----
python -m datasets.build_manifest --dataset CRACK500 \
       --root "$DATA_ROOT/CRACK500" --out "$OUT/CRACK500.json"

python -m datasets.build_manifest --dataset DeepCrack \
       --root "$DATA_ROOT/DeepCrack" --out "$OUT/DeepCrack.json"

python -m datasets.build_manifest --dataset CFD \
       --root "$DATA_ROOT/CFD" --out "$OUT/CFD.json"


echo "[ok] manifests written to $OUT/"
ls -l $OUT/
