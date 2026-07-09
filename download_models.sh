#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Download the two detection/segmentation checkpoints into pretrained_model/.
# The other models (CLIP, SegFormer, and Grounding DINO's BERT text encoder)
# download automatically from Hugging Face the first time you run the pipeline.
# Total size here: ~3.5 GB.
# ---------------------------------------------------------------------------
set -e
DIR="$(cd "$(dirname "$0")" && pwd)/pretrained_model"
mkdir -p "$DIR"

SAM_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
GDINO_URL="https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth"

echo "Downloading SAM ViT-H (~2.6 GB) ..."
wget -c -O "$DIR/sam_vit_h_4b8939.pth" "$SAM_URL"

echo "Downloading Grounding DINO SwinB (~938 MB) ..."
wget -c -O "$DIR/groundingdino_swinb_cogcoor.pth" "$GDINO_URL"

echo ""
echo "Done. Checkpoints are in: $DIR"
