#!/bin/bash
#SBATCH --job-name=hkv_dino_collapse
#SBATCH --partition=ampere
#SBATCH --qos=ampere-extd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=outputs/collapse_check/slurm_%j.out
#SBATCH --error=outputs/collapse_check/slurm_%j.err

# Phase C3 -- a real short run (~2000 iterations) on the FULL corpus at the
# probed production batch size, to check the loss is healthy: drifts down
# gradually, no NaN, no implausibly-fast collapse toward a degenerate value.
# DINO v1 logs one averaged loss value per EPOCH (not per-iteration like
# DINOv2), to output_dir/log.txt (JSON lines) -- coarser granularity, but
# with ~5 epochs here that's still 5 readable data points.
#
# NOTE: --output_dir is a SUBDIRECTORY of where SLURM writes its own
# slurm_%j.out/.err (this dir), so clearing it for a fresh run never touches
# the SLURM log (lesson learned from the DINOv2 repo's first version of this
# script, which rm -rf'd the SLURM log dir itself).
#
#   BS=<from mem_probe> sbatch cluster/collapse_check.sh

set -uo pipefail
PROJECT=$HOME/continue_ssl_pretrain_dino_v1
cd "$PROJECT"
source .venv/bin/activate

: "${BS:?set BS to the batch_size_per_gpu chosen from cluster/mem_probe.sh}"

OUT="$PROJECT/outputs/collapse_check/run"
rm -rf "$OUT"; mkdir -p "$OUT"
nvidia-smi

torchrun --standalone --nproc_per_node=1 dino/main_dino.py \
  --arch vit_small --patch_size 16 \
  --pretrained_backbone "$PROJECT/checkpoints/dino_deitsmall16_pretrain.pth" \
  --data_path "$PROJECT/data/hyperkvasir_imagefolder" \
  --output_dir "$OUT" \
  --batch_size_per_gpu "$BS" \
  --epochs 5 \
  --warmup_epochs 1 \
  --saveckp_freq 5 \
  --num_workers 8

echo ""
echo "=====================  LOSS TRAJECTORY (per epoch)  ====================="
python - <<PYEOF
import json
rows = [json.loads(l) for l in open("$OUT/log.txt")]
print(f"{'epoch':>6}  {'loss':>8}  {'lr':>10}")
for r in rows:
    print(f"{r['epoch']:>6}  {r['train_loss']:>8.4f}  {r['train_lr']:>10.2e}")
last = rows[-1]["train_loss"]
import math
bad = math.isnan(last) or last < 0.05
print()
print("VERDICT:", "!!! LOOKS LIKE COLLAPSE -- inspect closely" if bad
      else "healthy (final loss finite, not near-zero)")
PYEOF
echo "collapse_check finished: $(date)"
rm -f "$OUT"/checkpoint*.pth   # full-state checkpoints, not needed from this diagnostic run
