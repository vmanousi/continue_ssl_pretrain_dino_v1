"""Deep pre-flight check -- runs on the LOGIN NODE (no GPU). Verifies:
  1. the official backbone-only checkpoint loads cleanly into vit_small(patch16)
  2. the HyperKvasir ImageFolder symlink farm is in place and readable
  3. DataAugmentationDINO produces the expected 2 global (224) + 8 local (96) crops
  4. the default collate (no custom collate needed, unlike DINOv2's iBOT masking)
     batches a few samples into the right shapes

If this passes, the only things left untested are DDP/NCCL init and actual CUDA
kernels -- that's what the GPU smoke test covers.

Usage (repo root):
    python scripts/preflight_check.py
"""
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "dino"))


def section(msg):
    print("\n" + "=" * 70 + f"\n{msg}\n" + "=" * 70)


def main():
    section("1. package versions")
    import torchvision
    print("python      ", sys.version.split()[0])
    print("torch       ", torch.__version__)
    print("torchvision ", torchvision.__version__)

    section("2. checkpoint loads cleanly into vit_small(patch_size=16)")
    import vision_transformer as vits
    ckpt_path = REPO_ROOT / "checkpoints" / "dino_deitsmall16_pretrain.pth"
    assert ckpt_path.is_file(), f"missing {ckpt_path}"
    model = vits.__dict__["vit_small"](patch_size=16)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    assert not missing and not unexpected, f"missing={missing} unexpected={unexpected}"
    print(f"OK -- {len(state_dict)} tensors, 0 missing/unexpected, embed_dim={model.embed_dim}")

    section("3. HyperKvasir ImageFolder symlink farm")
    from torchvision import datasets
    full_dir = REPO_ROOT / "data" / "hyperkvasir_imagefolder"
    smoke_dir = REPO_ROOT / "data" / "smoke_subset"
    for name, d in (("full corpus", full_dir), ("smoke subset", smoke_dir)):
        assert d.is_dir(), f"missing {d} -- run scripts/prepare_hyperkvasir_imagefolder.sh"
        ds = datasets.ImageFolder(str(d))
        print(f"{name}: {len(ds)} images under {d}")

    section("4. DataAugmentationDINO -- crop counts/shapes")
    from main_dino import DataAugmentationDINO
    from PIL import Image
    aug = DataAugmentationDINO(global_crops_scale=(0.4, 1.0), local_crops_scale=(0.05, 0.4),
                               local_crops_number=8)
    ds = datasets.ImageFolder(str(smoke_dir))
    img, _ = ds[0]
    crops = aug(img)
    print(f"crops: {len(crops)} total "
          f"(expect 2 global + 8 local = 10) -- shapes: "
          f"{crops[0].shape} (global) ... {crops[-1].shape} (local)")
    assert len(crops) == 10
    assert crops[0].shape == (3, 224, 224)
    assert crops[-1].shape == (3, 96, 96)

    section("5. default collate over a small batch")
    from torch.utils.data import DataLoader

    class _Wrapped(torch.utils.data.Dataset):
        def __init__(self, base, transform):
            self.base, self.transform = base, transform

        def __len__(self):
            return len(self.base)

        def __getitem__(self, i):
            img, label = self.base[i]
            return self.transform(img), label

    loader = DataLoader(_Wrapped(ds, aug), batch_size=4, shuffle=False, num_workers=0)
    images, labels = next(iter(loader))
    print("collated: ", len(images), "crop positions, each shape", [im.shape for im in images[:2]], "...")
    assert len(images) == 10 and images[0].shape[0] == 4

    section("PRE-FLIGHT PASSED -- ready for the GPU smoke test")


if __name__ == "__main__":
    main()
