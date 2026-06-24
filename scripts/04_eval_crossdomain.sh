#!/usr/bin/env bash
set -e
export PYTHONPATH="$(pwd):${PYTHONPATH}"
CONFIG=${CONFIG:-configs/toposam_crack500.yaml}
CKPT=${CKPT:-outputs/toposam_crack500/best.pth}
python -m scripts_py.run_eval \
       --config $CONFIG --ckpt $CKPT \
       --save outputs/toposam_crack500/crossdomain.json
