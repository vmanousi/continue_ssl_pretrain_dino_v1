#!/bin/bash
# DINO v1's main_dino.py loads data via torchvision's ImageFolder, which needs
# at least one class subdirectory (data_path/<class>/<image>) -- our HyperKvasir
# corpus is a flat, unlabeled directory (SSL has no classes). Rather than patch
# main_dino.py's dataset loading (keep that stock, faithful to the original
# recipe), we build a symlink farm: one dummy "unlabeled" class pointing at the
# SAME 108,321 deduped images already prepared for the DINOv2 repo. Symlinks,
# not copies -- zero duplication of the 2.6 GB corpus, and READ-ONLY from the
# DINOv2 repo's data (never writes there -- full isolation).
#
# Also builds a tiny subset (256 images) for the smoke test / memory probe,
# since main_dino.py has no DINOv2-style OFFICIAL_EPOCH_LENGTH override --
# one "epoch" here is a real, full pass over whatever data_path points at.
#
#   bash scripts/prepare_hyperkvasir_imagefolder.sh

set -euo pipefail
DINOV2_REPO="${DINOV2_REPO:-$HOME/continue_ssl_pretrain_dinov2}"
SRC_ROOT="$DINOV2_REPO/data/hyperkvasir_ssl"
MANIFEST="$SRC_ROOT/train_deduped.txt"
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

[ -f "$MANIFEST" ] || { echo "MISSING $MANIFEST (is DINOV2_REPO set correctly?)"; exit 1; }

FULL_DIR="$PROJECT/data/hyperkvasir_imagefolder/unlabeled"
SMOKE_DIR="$PROJECT/data/smoke_subset/unlabeled"
mkdir -p "$FULL_DIR" "$SMOKE_DIR"

echo "Linking full corpus into $FULL_DIR ..."
n=0
while IFS= read -r rel; do
  ln -sf "$SRC_ROOT/$rel" "$FULL_DIR/$(basename "$rel")"
  n=$((n + 1))
done < "$MANIFEST"
echo "  linked $n images (full corpus)"

echo "Linking 256-image smoke/probe subset into $SMOKE_DIR ..."
head -n 256 "$MANIFEST" | while IFS= read -r rel; do
  ln -sf "$SRC_ROOT/$rel" "$SMOKE_DIR/$(basename "$rel")"
done
echo "  linked $(ls "$SMOKE_DIR" | wc -l) images (smoke subset)"

echo "done."
