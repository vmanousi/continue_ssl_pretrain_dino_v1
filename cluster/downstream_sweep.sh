#!/bin/bash
#SBATCH --job-name=hkv_dino_ds_sweep
#SBATCH --partition=ampere
#SBATCH --qos=ampere-extd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=outputs/downstream/sweep/slurm_%j.out
#SBATCH --error=outputs/downstream/sweep/slurm_%j.err

# Phase D, step 1 -- linear-probe the 12 continued-SSL backbone checkpoints
# (plus the generic official checkpoint as reference) on GastroHUN, pick the
# one with the best validation macro-F1. Linear probe = backbone frozen,
# features cached, only a linear head trained -> fast, a few minutes each.
# Mirrors cluster/downstream_sweep.sh in the DINOv2 repo, but simpler: both
# generic and continued checkpoints are already in the same {"model": sd}
# format (see scripts/extract_backbones.py), so no --backbone-kind distinction.
#   sbatch cluster/downstream_sweep.sh

set -uo pipefail
PROJECT=$HOME/continue_ssl_pretrain_dino_v1
cd "$PROJECT"
source .venv/bin/activate
export PYTHONPATH="$PROJECT/dino:$PROJECT/downstream:${PYTHONPATH:-}"

OUT="$PROJECT/outputs/downstream/sweep"
mkdir -p "$OUT"
GENERIC="$PROJECT/checkpoints/dino_deitsmall16_pretrain.pth"
BACKBONES="$PROJECT/outputs/full_run/backbones"

echo "===== generic (official) reference ====="
python downstream/train_gastrohun.py --mode linear_probe \
  --backbone-weights "$GENERIC" \
  --output-dir "$OUT/generic" 2>&1 | tail -3

for CKPT in "$BACKBONES"/checkpoint*_backbone.pth; do
  NAME=$(basename "$CKPT" _backbone.pth)
  echo "===== $NAME ====="
  python downstream/train_gastrohun.py --mode linear_probe \
    --backbone-weights "$CKPT" \
    --output-dir "$OUT/$NAME" 2>&1 | tail -3
done

echo ""
echo "=========================  RANKING (val macro-F1)  ========================="
python - <<'PY'
import glob, json, os
rows = []
for p in glob.glob(os.path.expanduser("~/continue_ssl_pretrain_dino_v1/outputs/downstream/sweep/*/summary.json")):
    d = json.load(open(p))
    name = os.path.basename(os.path.dirname(p))
    rows.append((d["linear_probe"]["best_f1_macro"], name))
for f1, name in sorted(rows, reverse=True):
    print(f"  {f1:6.2f}   {name}")
best = max((r for r in rows if r[1] != "generic"), default=None)
if best:
    print(f"\nBEST continued checkpoint: {best[1]}  (val macro-F1 {best[0]:.2f})")
PY
echo "sweep finished: $(date)"
