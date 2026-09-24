"""DINO v1 ViT-S/16 backbone + linear head for GastroHUN.

Both experiment arms use the SAME architecture and the SAME code path; the
only difference is which weights go into the backbone. Unlike the DINOv2
downstream port, there is only ONE checkpoint format here: {"model": state_dict}
-- the official dino_deitsmall16_pretrain.pth already uses it, and
scripts/extract_backbones.py writes our continued-pretraining checkpoints in
the same format, so no "kind" distinction is needed.

`set_finetune_mode` is a port of finetuning_models.frozen_dino from
`gastrohun-dino`: unfreeze the last `pct`% of the 12 transformer blocks (plus
the patch-embed stem only if pct > 95). cls/pos tokens, the final norm and the
head stay trainable, exactly as in the original.
"""
import torch
import torch.nn as nn

from vision_transformer import vit_small

EMBED_DIM = 384
DEPTH = 12


def build_backbone(weights_path):
    model = vit_small(patch_size=16)
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    sd = ckpt["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"backbone load mismatch: missing={missing} unexpected={unexpected}")
    return model


class GastroHunClassifier(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(EMBED_DIM, num_classes)

    def forward(self, x):
        return self.head(self.backbone(x))  # backbone(x) -> normalised CLS token (384-d)


def set_frozen_mode(model):
    """Linear probe / warm-up: everything frozen except the linear head."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True


def set_finetune_mode(model, pct):
    """Port of frozen_dino(model_ft, pct)."""
    for p in model.parameters():
        p.requires_grad = True

    n_frozen = int((100 - pct) * DEPTH / 100)
    model.backbone.patch_embed.requires_grad_(pct > 95)
    for i, block in enumerate(model.backbone.blocks):
        block.requires_grad_(i >= n_frozen)
    return model


def trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]
