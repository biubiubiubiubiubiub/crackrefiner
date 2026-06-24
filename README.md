# CrackRefiner

> **TopoSAM‑Crack** — SAM3 Teacher → Confidence‑Gated, Topology‑Aware Lightweight Student for Road Crack Segmentation
>
> 训练数据：CRACK500（弱监督，仅用 SAM3 伪标签 + image‑level 标签训练）
> 跨域评估：DeepCrack / CFD     ·     同源独立训练：GAPs384

## 1. 环境
- Python 3.10 + PyTorch 2.3 + CUDA 12.x
- AutoDL RTX 4090D（24GB）

```bash
pip install -r requirements.txt
```

## 2. 数据 & 权重路径约定
- 数据集放在 `/root/autodl-tmp/data/{CRACK500,DeepCrack,CFD,GAPs384}` （或软链到 `data/`）
- SAM3 权重放在 `/checkpoint/sam3/sam3.pt`

## 3. 一键流程
```bash
# 1) 生成统一 manifest（5 个数据集都生成，但只用 CRACK500 训练）
bash scripts/01_build_manifests.sh

# 2) 用 SAM3 对 CRACK500 train 集离线生成伪标签 + 置信图 + MC variance
bash scripts/02_generate_sam3_pseudo.sh

# 3) 训练 TopoSAM‑Crack Student（CRACK500 弱监督）
bash scripts/03_train.sh

# 4) 跨域评估（DeepCrack / CFD）
bash scripts/04_eval_crossdomain.sh
```

## 4. 目录（扁平化，无嵌套包）
```
crackrefiner/                # 项目根 = 运行目录
├── configs/                 # YAML 配置
├── manifests/               # 生成的 JSON 索引（每个数据集一份）
├── scripts/                 # 一键 shell
├── scripts_py/              # 三个 python 入口（被 -m 调用）
├── datasets/                # Manifest 驱动的 Dataset
├── models/                  # SAM3 Teacher + Student + Gate
├── losses/                  # Dice/BCE/SkelRecall/DT/Edge
├── engine/                  # Pseudo label / Trainer / Evaluator
├── metrics/                 # ODS/IoU/clDice
├── utils/
└── outputs/                 # 训练日志 / 权重 / 伪标签缓存
```

> 所有 `python -m ...` 命令必须在 **项目根目录** 下执行（即本 README 所在目录），
> 因为各模块用顶层 import：`from datasets import ...`、`from engine import ...` 等。

## 5. 论文对应方法
- **Teacher**：SAM3 (frozen, ViT‑H, text="crack" + prompt ensemble)
- **Student**：SCSegamba‑Lite (~3M params)，VSS block + Edge‑Aware branch
- **Gate**：r(p) = σ(α|2C_T−1| + β·exp(−d_skel/σ_s) + γ(1−Var_ens))
- **Loss**：`L = L_dice + λ1·L_BCE_w + λ2·L_SkelRecall + λ3·L_DT + λ4·L_edge`

## 6. ⚠️ SAM3 使用陷阱（务必注意）
- **不要把 SAM3 的 Dropout 模块设为 train() 做 MC sampling**。SAM3 的 Dropout
  在 attention / mask decoder 内部是结构性的，开启 train 会把所有 instance scores
  打到阈值之下，导致 mask 全 0。
- 想得到不确定性，请用 **prompt ensemble**（`ensemble_prompts` 字段）或
  **flip TTA**（`enable_hflip_tta: true`），它们都在 eval 模式下完成，结果可靠。
- 若 wrapper 拿到 mask 全 0 但 labelme 能用，先跑 `scripts_py/diag_sam3_minimal.py`
  定位。
