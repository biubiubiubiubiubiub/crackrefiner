#!/usr/bin/env bash
# =====================================================================
# CRACK500 增量 ablation: 串行跑 4 个配置（A → B → C → D）
#
# 设计逻辑（每一步只增加一项 loss，控制单变量）:
#   A: Dice + BCE                                         (2 loss, baseline)
#   B: A + clDice                                         (3 loss, +Topology)
#   C: B + SkelRecall                                     (4 loss, +Connectivity)  ★ 预期 CFD 跳变
#   D: C + DT + Edge                                      (6 loss, +Geometry)
#
# 已有 baseline 不重跑，直接复用:
#   E = outputs/toposam_crack500/             (7-loss Full)
#
# 时间预算 (RTX 4090D, 30 ep × 2.2 min/ep ≈ 66 min/config + 30 min eval):
#   train: 4 × 66 = 4h24min
#   eval : 4 × 30 = 2h00min
#   TOTAL: ≈ 6h30min  (一晚跑完)
# =====================================================================

set -e
export PYTHONPATH="$(pwd):${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 检查伪标签缓存（4 个配置共用，与 full / lite / rwft 完全一致）
if [ ! -d "outputs/pseudo/CRACK500" ]; then
    echo "[ERROR] outputs/pseudo/CRACK500/ not found."
    echo "        Run scripts/02_generate_sam3_pseudo.sh first."
    exit 1
fi
N_PSEUDO=$(ls outputs/pseudo/CRACK500/*.npz 2>/dev/null | wc -l)
echo "[info] found $N_PSEUDO pseudo labels under outputs/pseudo/CRACK500/"

mkdir -p outputs/ablation

# ---------- 单个配置训练+评估的函数 ----------
run_one() {
    TAG="$1"           # 例: A_dice_bce
    CFG="configs/ablation/${TAG}.yaml"
    OUT="outputs/ablation/${TAG}"

    echo ""
    echo "============================================================"
    echo "=== [$(date +%H:%M)] ABLATION ${TAG}  start train"
    echo "===   config : ${CFG}"
    echo "===   output : ${OUT}"
    echo "============================================================"

    python -m scripts_py.run_train --config "$CFG"

    echo ""
    echo "=== [$(date +%H:%M)] ABLATION ${TAG}  start cross-domain eval"
    python -m scripts_py.run_eval \
           --config "$CFG" \
           --ckpt  "${OUT}/best.pth" \
           --protocol paper \
           --save  "${OUT}/crossdomain_paper.json"

    echo "=== [$(date +%H:%M)] ABLATION ${TAG}  DONE ==="
}

# ---------- 顺序串行执行 ----------
run_one A_dice_bce
run_one B_plus_cldice
run_one C_plus_skelrecall
run_one D_plus_geom

# ---------- 汇总 ----------
echo ""
echo "============================================================"
echo "=== [$(date +%H:%M)]  ALL 4 ABLATIONS DONE  ==="
echo "============================================================"
echo "Results JSONs:"
for TAG in A_dice_bce B_plus_cldice C_plus_skelrecall D_plus_geom; do
    JSON="outputs/ablation/${TAG}/crossdomain_paper.json"
    if [ -f "$JSON" ]; then
        echo "  ✓ ${JSON}"
    else
        echo "  ✗ MISSING: ${JSON}"
    fi
done
echo ""
echo "Already-trained baselines for cross-reference:"
echo "  E (Full 7-loss) : outputs/toposam_crack500/crossdomain_paper.json"
echo "  F (Lite 4-loss) : outputs/toposam_crack500_lite/crossdomain_paper.json"
echo "  G (RWCT 4-loss) : outputs/toposam_crack500_rwft/crossdomain_paper.json"
