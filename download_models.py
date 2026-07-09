#!/usr/bin/env python3
"""
Download the two detection/segmentation checkpoints into pretrained_model/:
  - sam_vit_h_4b8939.pth            (SAM ViT-H, ~2.6 GB)
  - groundingdino_swinb_cogcoor.pth (Grounding DINO SwinB, ~938 MB)

Cross-platform alternative to download_models.sh (no wget needed).
CLIP / SegFormer / BERT download themselves from Hugging Face at runtime.
"""
import sys
import urllib.request
from pathlib import Path

MODELS = {
    "sam_vit_h_4b8939.pth":
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "groundingdino_swinb_cogcoor.pth":
        "https://github.com/IDEA-Research/GroundingDINO/releases/download/"
        "v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth",
}
DEST = Path(__file__).resolve().parent / "pretrained_model"


def _progress(block_num, block_size, total_size):
    if total_size > 0:
        done = min(100, block_num * block_size * 100 // total_size)
        sys.stdout.write(f"\r  {done:3d}%")
        sys.stdout.flush()


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for name, url in MODELS.items():
        out = DEST / name
        if out.exists() and out.stat().st_size > 0:
            print(f"[skip] {name} already present")
            continue
        print(f"[get ] {name}  <-  {url}")
        urllib.request.urlretrieve(url, out, _progress)
        print()
    print(f"Done. Checkpoints are in: {DEST}")


if __name__ == "__main__":
    main()
