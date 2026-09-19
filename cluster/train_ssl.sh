#!/bin/bash
#SBATCH --job-name=hkv_dino_ssl
#SBATCH --partition=ampere
#SBATCH --qos=ampere-extd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=6-00:00:00
#SBATCH --output=outputs/full_run/slurm_%j.out
#SBATCH --error=outputs/full_run/slurm_%j.err

# The full continued-pretraining run. 1 GPU (same reasoning as the DINOv2
# repo: this model is small, no need to fight multi-GPU/DDP complexity for it).
#
# DINO v1 has no OFFICIAL_EPOCH_LENGTH-style knob -- 1 epoch = 1 real pass over
# the corpus. To land on ~TARGET_ITERS total iterations (matching the DINOv2
# continued run's ~100k, for a comparable amount of training signal), we
# compute EPOCHS = TARGET_ITERS / (corpus_size / BS) here, and set warmup to
# 10% of that (matching DINO v1's own official proportion, 10/100 epochs).
#
#   BS=<from mem_probe> sbatch cluster/train_ssl.sh
#   BS=<...> TARGET_ITERS=100000 sbatch cluster/train_ssl.sh   # override target

set -euo pipefail
PROJECT=$HOME/continue_ssl_pretrain_dino_v1
cd "$PROJECT"
source .venv/bin/activate

: "${BS:?set BS to the batch_size_per_gpu chosen from cluster/mem_probe.sh}"
TARGET_ITERS=${TARGET_ITERS:-100000}

CORPUS_SIZE=$(ls "$PROJECT/data/hyperkvasir_imagefolder/unlabeled" | wc -l)
ITERS_PER_EPOCH=$(( CORPUS_SIZE / BS ))
EPOCHS=$(( (TARGET_ITERS + ITERS_PER_EPOCH - 1) / ITERS_PER_EPOCH ))
WARMUP_EPOCHS=$(( (EPOCHS + 9) / 10 ))   # ceil(10% of EPOCHS)
TOTAL_ITERS=$(( EPOCHS * ITERS_PER_EPOCH ))

echo "corpus=$CORPUS_SIZE  batch=$BS  iters/epoch=$ITERS_PER_EPOCH"
echo "-> epochs=$EPOCHS  warmup_epochs=$WARMUP_EPOCHS  total_iters=$TOTAL_ITERS  (target was $TARGET_ITERS)"

OUT="$PROJECT/outputs/full_run"
mkdir -p "$OUT"
nvidia-smi

torchrun --standalone --nproc_per_node=1 dino/main_dino.py \
  --arch vit_small --patch_size 16 \
  --pretrained_backbone "$PROJECT/checkpoints/dino_deitsmall16_pretrain.pth" \
  --data_path "$PROJECT/data/hyperkvasir_imagefolder" \
  --output_dir "$OUT" \
  --batch_size_per_gpu "$BS" \
  --epochs "$EPOCHS" \
  --warmup_epochs "$WARMUP_EPOCHS" \
  --saveckp_freq $(( EPOCHS / 10 )) \
  --num_workers 12

echo "done: $(date)"
