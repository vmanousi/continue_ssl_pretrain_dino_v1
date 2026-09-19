#!/bin/bash
#SBATCH --job-name=hkv_dino_smoke
#SBATCH --partition=ampere
#SBATCH --qos=ampere-extd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/smoke/slurm_%j.out
#SBATCH --error=outputs/smoke/slurm_%j.err

# Short smoke test: 1 epoch over the 256-image subset (32 iterations at
# batch 8) -- does the whole continued-pretraining pipeline run on a GPU?
# main_dino.py has no DINOv2-style OFFICIAL_EPOCH_LENGTH override, so "1
# epoch" here means one full pass over whatever --data_path points at --
# that's why the smoke run uses the tiny subset, not the full corpus.
#   sbatch cluster/smoke_test.sh

set -euo pipefail
PROJECT=$HOME/continue_ssl_pretrain_dino_v1
cd "$PROJECT"
source .venv/bin/activate

mkdir -p outputs/smoke
nvidia-smi

torchrun --standalone --nproc_per_node=1 dino/main_dino.py \
  --arch vit_small \
  --patch_size 16 \
  --pretrained_backbone "$PROJECT/checkpoints/dino_deitsmall16_pretrain.pth" \
  --data_path "$PROJECT/data/smoke_subset" \
  --output_dir "$PROJECT/outputs/smoke" \
  --batch_size_per_gpu 8 \
  --epochs 1 \
  --warmup_epochs 0 \
  --saveckp_freq 1 \
  --num_workers 4

echo "smoke test finished: $(date)"
