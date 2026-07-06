# UrbanchangeAI


This repository implements a building segmentation workflow for panoramic street-level imagery using:

- Grounding DINO via Hugging Face Transformers
- SAM2 for building instance segmentation
- Ground-floor detection

The workflow is designed for urban change analysis and façade-level interpretation.

## Workflow

```text
Panoramic image
    ↓
Grounding DINO
    ↓
SAM2
    ↓
Building instance masks
    ↓
Ground-floor extraction / detection

Ground-floor usage classification is not included in this version.

Installation
conda env create -f environment.yml
conda activate grounded_sam

Install SAM2 separately from the official Grounded-SAM-2 / SAM2 repository, then place the checkpoint in:

checkpoints/sam2.1_hiera_large.pt
Usage
Building instance segmentation
python scripts/01_grounded_sam_buildings.py
Ground-floor element detection
python scripts/02_groundfloor_detection.py
Outputs
outputs/
├── buildings/
│   ├── overlays/
│   ├── masks/
│   └── csv/
└── groundfloor/
    ├── overlays/
    ├── semantic/
    ├── masks/
    ├── csv/
    └── annotations/

---

## 8. `scripts/01_grounded_sam_buildings.py`

```python
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


INPUT_DIR = Path("input/sample_images")

OUT_DIR = Path("outputs/buildings")
OVERLAY_DIR = OUT_DIR / "overlays"
MASK_DIR = OUT_DIR / "masks"
CSV_DIR = OUT_DIR / "csv"

for d in [OUT_DIR, OVERLAY_DIR, MASK_DIR, CSV_DIR]:
    d.mkdir(parents=True, exist_ok=True)

LIMIT = 10

GROUNDING_MODEL_ID = "IDEA-Research/grounding-dino-base"
TEXT_PROMPT = "building."

BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.25

SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CHECKPOINT = "checkpoints/sam2.1_hiera_large.pt"

ALPHA = 0.45


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

processor = AutoProcessor.from_pretrained(GROUNDING_MODEL_ID)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
    GROUNDING_MODEL_ID
).to(device)
grounding_model.eval()

sam2_model = build_sam2(
    SAM2_CONFIG,
    SAM2_CHECKPOINT,
    device=device,
)

predictor = SAM2ImagePredictor(sam2_model)

image_files = sorted(
    list(INPUT_DIR.glob("*.jpg"))
    + list(INPUT_DIR.glob("*.jpeg"))
    + list(INPUT_DIR.glob("*.png"))
)[:LIMIT]

print(f"Images to process: {len(image_files)}")

all_rows = []

for image_path in image_files:
    print(f"\nProcessing: {image_path.name}")

    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    inputs = processor(
        images=image,
        text=TEXT_PROMPT,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[image.size[::-1]],
    )[0]

    boxes = results["boxes"].detach().cpu().numpy()
    scores = results["scores"].detach().cpu().numpy()
    labels = results["labels"]

    print(f"Detections: {len(boxes)}")

    if len(boxes) == 0:
        continue

    predictor.set_image(image_np)

    overlay_np = image_np.copy()

    np.random.seed(42)
    colors = np.random.randint(
        40,
        255,
        size=(len(boxes), 3),
        dtype=np.uint8,
    )

    image_masks = []
    rows = []

    for idx, (box, score, label) in enumerate(zip(boxes, scores, labels), start=1):
        masks, mask_scores, _ = predictor.predict(
            box=box,
            multimask_output=False,
        )

        mask = masks[0].astype(bool)
        mask_score = float(mask_scores[0])
        mask_area = int(mask.sum())

        if mask_area == 0:
            continue

        color = colors[idx - 1]

        overlay_np[mask] = (
            (1 - ALPHA) * image_np[mask] + ALPHA * color
        ).astype(np.uint8)

        ys, xs = np.where(mask)
        xmin, xmax = int(xs.min()), int(xs.max())
        ymin, ymax = int(ys.min()), int(ys.max())

        building_id = f"Building_{idx:03d}"

        image_masks.append(mask)

        row = {
            "image": image_path.name,
            "image_path": str(image_path),
            "stem": image_path.stem,
            "building_id": building_id,
            "label": label,
            "dino_score": float(score),
            "sam2_mask_score": mask_score,
            "mask_area": mask_area,
            "dino_box_xmin": float(box[0]),
            "dino_box_ymin": float(box[1]),
            "dino_box_xmax": float(box[2]),
            "dino_box_ymax": float(box[3]),
            "mask_bbox_xmin": xmin,
            "mask_bbox_ymin": ymin,
            "mask_bbox_xmax": xmax,
            "mask_bbox_ymax": ymax,
            "color_r": int(color[0]),
            "color_g": int(color[1]),
            "color_b": int(color[2]),
        }

        rows.append(row)
        all_rows.append(row)

        print(
            building_id,
            "dino_score=", round(float(score), 3),
            "sam_score=", round(mask_score, 3),
            "area=", mask_area,
        )

    stem = image_path.stem

    overlay_path = OVERLAY_DIR / f"{stem}_buildings_overlay.png"
    masks_path = MASK_DIR / f"{stem}_building_masks.npy"
    csv_path = CSV_DIR / f"{stem}_buildings.csv"

    Image.fromarray(overlay_np).save(overlay_path)

    if image_masks:
        np.save(masks_path, np.stack(image_masks, axis=0))

    df = pd.DataFrame(rows).sort_values("mask_area", ascending=False)
    df.to_csv(csv_path, index=False)

    print("Saved overlay:", overlay_path)
    print("Saved masks:", masks_path)
    print("Saved CSV:", csv_path)

combined_csv = OUT_DIR / "all_buildings.csv"
pd.DataFrame(all_rows).to_csv(combined_csv, index=False)

print("\nDone.")
print("Saved combined CSV:", combined_csv)
