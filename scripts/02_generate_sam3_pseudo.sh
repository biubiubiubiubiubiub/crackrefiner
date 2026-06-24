#!/usr/bin/env bash
set -e
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# 若 SAM3 是源码方式（未 pip install），手动加入仓库根目录
# 让 "import sam3" 命中 ./sam3/sam3/__init__.py 而不是外层的命名空间目录
SAM3_HOME=${SAM3_HOME:-./sam3}
if [ -d "$SAM3_HOME/sam3" ] && [ -f "$SAM3_HOME/sam3/__init__.py" ]; then
  export PYTHONPATH="$SAM3_HOME:${PYTHONPATH}"
  echo "[info] SAM3 source path added: $SAM3_HOME"
fi

CONFIG=${CONFIG:-configs/toposam_crack500.yaml}

# CRACK500 train split 生成伪标签（弱监督设定）
python -m scripts_py.run_generate_pseudo \
       --config $CONFIG \
       --manifest manifests/CRACK500.json \
       --split train

# val 也生成，便于无 GT 时做 sanity check
python -m scripts_py.run_generate_pseudo \
       --config $CONFIG \
       --manifest manifests/CRACK500.json \
       --split val
