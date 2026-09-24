"""Extract clean backbone-only checkpoints from DINO v1's full-state saves.

Unlike DINOv2 (which writes a clean teacher_checkpoint.pth periodically via
do_test()), DINO v1's main_dino.py checkpoints are FULL training state:
{student, teacher, optimizer, epoch, args, dino_loss, fp16_scaler}. The
downstream checkpoint-sweep pipeline only wants the teacher backbone weights,
in the same {"model": state_dict} format as the official
dino_deitsmall16_pretrain.pth checkpoint (so the sweep can load every
candidate -- generic official, and every continued-pretraining epoch --
through the exact same code path).

teacher["backbone.*"] holds the ViT weights (150 tensors for vit_small);
teacher["head.*"] is the DINO projection head, discarded here since the
downstream task attaches its own classification head to the backbone only.

Usage (repo root, login node, no GPU needed):
    python scripts/extract_backbones.py
    python scripts/extract_backbones.py --run_dir outputs/full_run
"""
import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "dino"))


def extract_one(ckpt_path: Path, expected_tensors: int) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    teacher_sd = ckpt["teacher"]
    backbone_sd = {
        k[len("backbone."):]: v
        for k, v in teacher_sd.items()
        if k.startswith("backbone.")
    }
    assert len(backbone_sd) == expected_tensors, (
        f"{ckpt_path.name}: expected {expected_tensors} backbone tensors, got {len(backbone_sd)}"
    )
    return {"model": backbone_sd}, ckpt.get("epoch")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default=str(REPO_ROOT / "outputs" / "full_run"))
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = run_dir / "backbones"
    out_dir.mkdir(exist_ok=True)

    import vision_transformer as vits
    ref_model = vits.__dict__["vit_small"](patch_size=16)
    expected_tensors = len(ref_model.state_dict())

    ckpt_files = sorted(run_dir.glob("checkpoint*.pth"))
    assert ckpt_files, f"no checkpoint*.pth files found under {run_dir}"

    print(f"{'file':>20}  {'recorded_epoch':>14}  {'tensors':>8}  ->  out")
    for ckpt_path in ckpt_files:
        backbone_ckpt, epoch = extract_one(ckpt_path, expected_tensors)
        out_path = out_dir / f"{ckpt_path.stem}_backbone.pth"
        torch.save(backbone_ckpt, out_path)
        print(f"{ckpt_path.name:>20}  {str(epoch):>14}  {expected_tensors:>8}  ->  {out_path.name}")

    print(f"\ndone -- {len(ckpt_files)} backbone-only checkpoints written to {out_dir}")


if __name__ == "__main__":
    main()
