from pathlib import Path
import json

import torch
import numpy as np
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# -------------------------------------------------
# CONFIG
# -------------------------------------------------

INPUT_DIR = Path(
    "/mnt/c/Users/psaro/Downloads/panorama_koukaki/gsv_koukaki_test/Archive/Test datasets/02_unique_gsv_dataset/latest"
)

OUT_DIR = (
    Path.home()
    / "UrbanChangeAI"
    / "results"
    / "Segmentation_test"
    / "GroundedSAM2_GroundFloor2"
)

OVERLAY_DIR = OUT_DIR / "overlays"
MASK_DIR = OUT_DIR / "masks"          # instance masks (.npy) + semantic PNG
SEM_DIR = OUT_DIR / "semantic"        # class-colored semantic masks
CSV_DIR = OUT_DIR / "csv"
JSON_DIR = OUT_DIR / "annotations"

for d in [OUT_DIR, OVERLAY_DIR, MASK_DIR, SEM_DIR, CSV_DIR, JSON_DIR]:
    d.mkdir(parents=True, exist_ok=True)

LIMIT = 10

GROUNDING_MODEL_ID = "IDEA-Research/grounding-dino-base"

# ground-floor element classes (GroundingDINO: lowercase, period-separated)
GF_PROMPT = "storefront . entrance door . glass window . cafe . restaurant . pillar . garage door ."
BUILDING_PROMPT = "building"

# keyword -> canonical class
CLASS_MAP = {
    "shop": "shop_retail",
    "storefront": "shop_retail",
    "store": "shop_retail",
    "entrance": "entrance_door",
    "door": "entrance_door",
    "garage": "entrance_door",
    "glass": "glass_window",
    "window": "glass_window",
    "cafe": "cafe_restaurant",
    "restaurant": "cafe_restaurant",
    "pillar": "pillar",
    "column": "pillar",
}
FALLBACK_CLASS = "other_ground_floor"

CLASS_COLORS = {
    "shop_retail":        (220,  70,  70),
    "entrance_door":      (235, 195,  50),
    "glass_window":       ( 90, 200,  90),
    "cafe_restaurant":    ( 60, 100, 235),
    "pillar":             (170, 130, 220),
    "other_ground_floor": (240, 150,  40),
}
CLASS_IDS = {c: i + 1 for i, c in enumerate(CLASS_COLORS)}  # 0 = background

BOX_THRESHOLD = 0.27
TEXT_THRESHOLD = 0.22

# ground-floor geometric filter (fractions of image height)
# equirectangular pano: horizon ~ 0.5 * H
GF_YMAX_MIN = 0.45   # box must extend below this line (i.e., into lower half)
GF_YMIN_MAX = 0.90   # box must start above this (reject pure-road detections)
MIN_BOX_AREA_FRAC = 0.0005  # reject tiny boxes
NMS_IOU = 0.55

SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CHECKPOINT = "checkpoints/sam2.1_hiera_large.pt"

ALPHA = 0.45


# -------------------------------------------------
# HELPERS
# -------------------------------------------------

def map_label(raw_label: str) -> str:
    raw = raw_label.lower()
    for kw, cls in CLASS_MAP.items():
        if kw in raw:
            return cls
    return FALLBACK_CLASS


def box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def nms(boxes, scores, iou_thr):
    order = np.argsort(scores)[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        order = np.array(
            [j for j in rest if box_iou(boxes[i], boxes[j]) < iou_thr],
            dtype=int,
        )
    return keep


def horizontal_overlap_frac(elem_box, bld_box):
    """fraction of element's width covered by the building box"""
    ix1 = max(elem_box[0], bld_box[0])
    ix2 = min(elem_box[2], bld_box[2])
    w = elem_box[2] - elem_box[0]
    if w <= 0:
        return 0.0
    return max(0.0, ix2 - ix1) / w


def detect(image, prompt, box_thr, text_thr):
    inputs = processor(
        images=image, text=prompt, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_thr,
        text_threshold=text_thr,
        target_sizes=[image.size[::-1]],
    )[0]
    return (
        results["boxes"].detach().cpu().numpy(),
        results["scores"].detach().cpu().numpy(),
        results["labels"],
    )


# -------------------------------------------------
# LOAD MODELS
# -------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

processor = AutoProcessor.from_pretrained(GROUNDING_MODEL_ID)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
    GROUNDING_MODEL_ID
).to(device)
grounding_model.eval()

sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
predictor = SAM2ImagePredictor(sam2_model)


# -------------------------------------------------
# INPUT IMAGES
# -------------------------------------------------

image_files = sorted(
    list(INPUT_DIR.glob("*.jpg")) +
    list(INPUT_DIR.glob("*.jpeg")) +
    list(INPUT_DIR.glob("*.png"))
)[:LIMIT]

print(f"Images to process: {len(image_files)}")


# -------------------------------------------------
# PROCESS
# -------------------------------------------------

all_rows = []

for image_path in image_files:
    print(f"\nProcessing: {image_path.name}")

    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    H, W = image_np.shape[:2]
    img_area = H * W

    # -----------------------------
    # 1. Building detection (for grouping)
    # -----------------------------
    b_boxes, b_scores, _ = detect(image, BUILDING_PROMPT, 0.30, 0.25)
    if len(b_boxes) > 0:
        keep_b = nms(b_boxes, b_scores, 0.5)
        b_boxes = b_boxes[keep_b]
        # sort left-to-right for stable building ids
        b_boxes = b_boxes[np.argsort(b_boxes[:, 0])]
    print(f"Buildings: {len(b_boxes)}")

    # -----------------------------
    # 2. Ground-floor element detection
    # -----------------------------
    boxes, scores, labels = detect(image, GF_PROMPT, BOX_THRESHOLD, TEXT_THRESHOLD)
    print(f"Raw detections: {len(boxes)}")

    if len(boxes) == 0:
        continue

    # geometric ground-floor filter
    keep_idx = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        area_frac = (x2 - x1) * (y2 - y1) / img_area
        if y2 < GF_YMAX_MIN * H:      # doesn't reach lower half -> upper floor
            continue
        if y1 > GF_YMIN_MAX * H:      # starts near bottom edge -> road artifact
            continue
        if area_frac < MIN_BOX_AREA_FRAC:
            continue
        keep_idx.append(i)

    boxes = boxes[keep_idx]
    scores = scores[keep_idx]
    labels = [labels[i] for i in keep_idx]

    # NMS to dedupe overlapping detections
    if len(boxes) > 0:
        keep_idx = nms(boxes, scores, NMS_IOU)
        boxes = boxes[keep_idx]
        scores = scores[keep_idx]
        labels = [labels[i] for i in keep_idx]

    print(f"Ground-floor detections after filtering: {len(boxes)}")
    if len(boxes) == 0:
        continue

    # -----------------------------
    # 3. SAM2 segmentation
    # -----------------------------
    predictor.set_image(image_np)

    overlay_np = image_np.copy()
    semantic_mask = np.zeros((H, W, 3), dtype=np.uint8)
    instance_mask = np.zeros((H, W), dtype=np.uint16)

    image_masks = []
    rows = []
    inst_id = 0

    # segment larger boxes first so small elements (doors) paint on top
    order = np.argsort(-(boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))

    for i in order:
        box, score, raw_label = boxes[i], scores[i], labels[i]
        cls = map_label(raw_label)

        masks, mask_scores, _ = predictor.predict(box=box, multimask_output=False)
        mask = masks[0].astype(bool)
        mask_area = int(mask.sum())
        if mask_area == 0:
            continue

        inst_id += 1
        color = np.array(CLASS_COLORS[cls], dtype=np.uint8)

        overlay_np[mask] = (
            (1 - ALPHA) * image_np[mask] + ALPHA * color
        ).astype(np.uint8)
        semantic_mask[mask] = color
        instance_mask[mask] = inst_id

        # assign to building
        building_id = "Building_000"
        if len(b_boxes) > 0:
            fracs = [horizontal_overlap_frac(box, bb) for bb in b_boxes]
            j = int(np.argmax(fracs))
            if fracs[j] > 0.3:
                building_id = f"Building_{j + 1:03d}"

        ys, xs = np.where(mask)
        image_masks.append(mask)

        row = {
            "image": image_path.name,
            "stem": image_path.stem,
            "building_id": building_id,
            "instance_id": inst_id,
            "class": cls,
            "raw_label": raw_label,
            "dino_score": float(score),
            "sam2_mask_score": float(mask_scores[0]),
            "mask_area": mask_area,
            "dino_box_xmin": float(box[0]),
            "dino_box_ymin": float(box[1]),
            "dino_box_xmax": float(box[2]),
            "dino_box_ymax": float(box[3]),
            "mask_bbox_xmin": int(xs.min()),
            "mask_bbox_ymin": int(ys.min()),
            "mask_bbox_xmax": int(xs.max()),
            "mask_bbox_ymax": int(ys.max()),
        }
        rows.append(row)
        all_rows.append(row)

        print(
            building_id, f"inst_{inst_id:03d}", cls,
            "dino=", round(float(score), 3),
            "sam=", round(float(mask_scores[0]), 3),
            "area=", mask_area,
        )

    stem = image_path.stem

    Image.fromarray(overlay_np).save(
        OVERLAY_DIR / f"{stem}_groundfloor_overlay.png"
    )
    Image.fromarray(semantic_mask).save(
        SEM_DIR / f"{stem}_groundfloor_semantic.png"
    )
    np.save(MASK_DIR / f"{stem}_instance_mask.npy", instance_mask)
    if image_masks:
        np.save(
            MASK_DIR / f"{stem}_binary_masks.npy",
            np.stack(image_masks, axis=0),
        )

    df = pd.DataFrame(rows).sort_values(
        ["building_id", "mask_area"], ascending=[True, False]
    )
    df.to_csv(CSV_DIR / f"{stem}_groundfloor.csv", index=False)

    with open(JSON_DIR / f"{stem}.json", "w") as f:
        json.dump(rows, f, indent=2)

    print("Saved:", stem)


# -------------------------------------------------
# COMBINED CSV
# -------------------------------------------------

combined_csv = OUT_DIR / "all_groundfloor_instances.csv"
pd.DataFrame(all_rows).to_csv(combined_csv, index=False)

print("\nDone. Combined CSV:", combined_csv)
