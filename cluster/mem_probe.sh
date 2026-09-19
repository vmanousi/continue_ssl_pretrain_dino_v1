#!/bin/bash
#SBATCH --job-name=hkv_dino_memprobe
#SBATCH --partition=ampere
#SBATCH --qos=ampere-extd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/memprobe/slurm_%j.out
#SBATCH --error=outputs/memprobe/slurm_%j.err

# Phase C2 -- find the largest batch_size_per_gpu that fits on one A100-40GB.
# Runs 1 epoch over the 256-image subset per candidate (drop_last=True caps
# meaningful candidates at <=256 for this subset size), records peak GPU mem
# from DINO v1's own "max mem: X" log line, catches CUDA OOM.
#   sbatch cluster/mem_probe.sh

set -uo pipefail
PROJECT=$HOME/continue_ssl_pretrain_dino_v1
cd "$PROJECT"
source .venv/bin/activate

mkdir -p outputs/memprobe
nvidia-smi

CANDIDATES="32 64 96 128 160 192 224 256"
RESULTS=()

for BS in $CANDIDATES; do
  echo ""
  echo "=================  batch_size_per_gpu = $BS  ================="
  OUT="outputs/memprobe/bs_$BS"
  rm -rf "$OUT"; mkdir -p "$OUT"
  LOG="$OUT/run.log"

  timeout 400 torchrun --standalone --nproc_per_node=1 dino/main_dino.py \
    --arch vit_small --patch_size 16 \
    --pretrained_backbone "$PROJECT/checkpoints/dino_deitsmall16_pretrain.pth" \
    --data_path "$PROJECT/data/smoke_subset" \
    --output_dir "$OUT" \
    --batch_size_per_gpu $BS \
    --epochs 1 \
    --warmup_epochs 0 \
    --saveckp_freq 1 \
    --num_workers 4 \
    > "$LOG" 2>&1
  RC=$?

  if grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG"; then
    VERDICT="OOM"
  elif [ $RC -eq 124 ]; then
    VERDICT="TIMEOUT(>400s)"
  elif [ $RC -ne 0 ]; then
    VERDICT="FAILED rc=$RC (see $LOG)"
  else
    MEM=$(grep -oE "max mem: [0-9]+" "$LOG" | tail -1 | grep -oE "[0-9]+")
    VERDICT="OK  peak_mem=${MEM}MB"
  fi
  echo ">>> BS=$BS : $VERDICT"
  RESULTS+=("BS=$BS : $VERDICT")
  rm -rf "$OUT"
done

echo ""
echo "=========================  SUMMARY  ========================="
for r in "${RESULTS[@]}"; do echo "$r"; done
echo "============================================================"
echo "memprobe finished: $(date)"
