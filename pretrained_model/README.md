# pretrained_model/

Model checkpoints go here (not tracked in git -- see `.gitignore`). Run
`download_models.sh` or `download_models.py` from the repo root to fetch them:

| File | Model | Size | Source |
|------|-------|------|--------|
| `sam_vit_h_4b8939.pth` | SAM ViT-H | 2.6 GB | Meta segment-anything |
| `groundingdino_swinb_cogcoor.pth` | Grounding DINO SwinB | 938 MB | IDEA-Research |

CLIP, SegFormer, Grounding DINO's BERT text encoder, and PaddleOCR PP-OCRv6's
detection/recognition models are not stored here either -- they download
themselves on first run (see the main README's Setup section).
