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

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = PROJECT_ROOT / "input" / "sample_images"
OUT_DIR = PROJECT_ROOT / "outputs" / "urban_scene_intelligence"

OVERLAY_DIR = OUT_DIR / "overlays"
CROP_DIR = OUT_DIR / "crops"
MASK_DIR = OUT_DIR / "masks"
CSV_DIR = OUT_DIR / "csv"
JSON_DIR = OUT_DIR / "json"

for d in [OUT_DIR, OVERLAY_DIR, CROP_DIR, MASK_DIR, CSV_DIR, JSON_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------
# Parameters
# -------------------------------------------------

LIMIT = 10

GROUNDING_MODEL_ID = "IDEA-Research/grounding-dino-base"

BUILDING_PROMPT = "building."

COMMERCIAL_UNIT_PROMPT = (
    "commercial storefront . "
    "shopfront . "
    "retail storefront . "
    "business frontage . "
    "commercial unit . "
    "ground floor shop . "
    "vacant storefront . "
    "closed storefront . "
)

BUILDING_BOX_THRESHOLD = 0.30
BUILDING_TEXT_THRESHOLD = 0.25

UNIT_BOX_THRESHOLD = 0.20
UNIT_TEXT_THRESHOLD = 0.16

GROUND_FLOOR_RATIO = 0.38

MIN_UNIT_AREA_FRAC = 0.00015
UNIT_NMS_IOU = 0.50
MIN_OVERLAP_WITH_GROUND_FLOOR = 0.08

SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "sam2.1_hiera_large.pt"

ALPHA = 0.50

UNIT_COLORS = [
    (255, 60, 60),
    (60, 180, 255),
    (255, 180, 40),
    (120, 220, 80),
    (190, 90, 255),
    (255, 120, 220),
    (80, 220, 200),
]


# -------------------------------------------------
# Helper functions
# -------------------------------------------------

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
    if len(boxes) == 0:
        return []

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


def detect(image, prompt, box_thr, text_thr, processor, model, device):
    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

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


def extract_groundfloor_mask(building_mask, ratio):
    ys, xs = np.where(building_mask)

    if len(xs) == 0 or len(ys) == 0:
        return None, None

    xmin, xmax = int(xs.min()), int(xs.max())
    ymin, ymax = int(ys.min()), int(ys.max())

    bbox_h = ymax - ymin + 1
    ground_ymin = int(ymax - bbox_h * ratio)
    ground_ymax = ymax

    ground_zone = np.zeros_like(building_mask, dtype=bool)
    ground_zone[ground_ymin:ground_ymax + 1, xmin:xmax + 1] = True

    ground_mask = building_mask & ground_zone

    info = {
        "building_bbox_xmin": xmin,
        "building_bbox_ymin": ymin,
        "building_bbox_xmax": xmax,
        "building_bbox_ymax": ymax,
        "ground_ymin": ground_ymin,
        "ground_ymax": ground_ymax,
    }

    return ground_mask, info


def box_overlap_with_mask(box, mask):
    x1, y1, x2, y2 = [int(v) for v in box]

    h, w = mask.shape
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))

    if x2 <= x1 or y2 <= y1:
        return 0.0

    box_area = (x2 - x1) * (y2 - y1)
    mask_area_in_box = int(mask[y1:y2, x1:x2].sum())

    return mask_area_in_box / box_area if box_area > 0 else 0.0


def box_center_inside_mask(box, mask):
    x1, y1, x2, y2 = box
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    h, w = mask.shape

    if cx < 0 or cx >= w or cy < 0 or cy >= h:
        return False

    return bool(mask[cy, cx])


def crop_from_mask(image, mask, padding=20):
    ys, xs = np.where(mask)

    if len(xs) == 0 or len(ys) == 0:
        return None, None

    w, h = image.size

    xmin = max(0, int(xs.min()) - padding)
    xmax = min(w, int(xs.max()) + padding)
    ymin = max(0, int(ys.min()) - padding)
    ymax = min(h, int(ys.max()) + padding)

    crop = image.crop((xmin, ymin, xmax, ymax))

    bbox = {
        "crop_xmin": xmin,
        "crop_ymin": ymin,
        "crop_xmax": xmax,
        "crop_ymax": ymax,
    }

    return crop, bbox


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT_DIR}")

    if not SAM2_CHECKPOINT.exists():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {SAM2_CHECKPOINT}")

    processor = AutoProcessor.from_pretrained(GROUNDING_MODEL_ID)

    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        GROUNDING_MODEL_ID
    ).to(device)
    grounding_model.eval()

    sam2_model = build_sam2(
        SAM2_CONFIG,
        str(SAM2_CHECKPOINT),
        device=device,
    )

    predictor = SAM2ImagePredictor(sam2_model)

    image_files = sorted(
        list(INPUT_DIR.glob("*.jpg")) +
        list(INPUT_DIR.glob("*.jpeg")) +
        list(INPUT_DIR.glob("*.png"))
    )

    if LIMIT is not None:
        image_files = image_files[:LIMIT]

    print(f"Images to process: {len(image_files)}")

    all_rows = []

    for image_path in image_files:
        print(f"\nProcessing: {image_path.name}")

        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        H, W = image_np.shape[:2]
        img_area = H * W

        # -----------------------------------------
        # 1. Building discovery
        # -----------------------------------------

        b_boxes, b_scores, b_labels = detect(
            image,
            BUILDING_PROMPT,
            BUILDING_BOX_THRESHOLD,
            BUILDING_TEXT_THRESHOLD,
            processor,
            grounding_model,
            device,
        )

        if len(b_boxes) > 0:
            keep = nms(b_boxes, b_scores, 0.50)
            b_boxes = b_boxes[keep]
            b_scores = b_scores[keep]
            b_labels = [b_labels[i] for i in keep]

            order = np.argsort(b_boxes[:, 0])
            b_boxes = b_boxes[order]
            b_scores = b_scores[order]
            b_labels = [b_labels[i] for i in order]

        print(f"Buildings detected: {len(b_boxes)}")

        if len(b_boxes) == 0:
            continue

        predictor.set_image(image_np)

        building_ground_masks = []

        # -----------------------------------------
        # 2. Building segmentation + ground-floor ROI
        # -----------------------------------------

        for b_idx, (b_box, b_score, b_label) in enumerate(
            zip(b_boxes, b_scores, b_labels),
            start=1,
        ):
            b_masks, b_mask_scores, _ = predictor.predict(
                box=b_box,
                multimask_output=False,
            )

            building_mask = b_masks[0].astype(bool)
            building_area = int(building_mask.sum())

            if building_area == 0:
                continue

            ground_mask, ground_info = extract_groundfloor_mask(
                building_mask,
                GROUND_FLOOR_RATIO,
            )

            if ground_mask is None or int(ground_mask.sum()) == 0:
                continue

            building_id = f"Building_{b_idx:03d}"

            building_ground_masks.append({
                "building_id": building_id,
                "building_box": b_box,
                "building_score": float(b_score),
                "building_mask_score": float(b_mask_scores[0]),
                "building_area": building_area,
                "building_mask": building_mask,
                "ground_mask": ground_mask,
                "ground_area": int(ground_mask.sum()),
                "ground_info": ground_info,
            })

        print(f"Ground-floor ROIs: {len(building_ground_masks)}")

        # -----------------------------------------
        # 3. Commercial unit proposals
        # -----------------------------------------

        u_boxes, u_scores, u_labels = detect(
            image,
            COMMERCIAL_UNIT_PROMPT,
            UNIT_BOX_THRESHOLD,
            UNIT_TEXT_THRESHOLD,
            processor,
            grounding_model,
            device,
        )

        print(f"Raw commercial unit detections: {len(u_boxes)}")

        if len(u_boxes) > 0:
            keep = nms(u_boxes, u_scores, UNIT_NMS_IOU)
            u_boxes = u_boxes[keep]
            u_scores = u_scores[keep]
            u_labels = [u_labels[i] for i in keep]

        print(f"After NMS: {len(u_boxes)}")

        overlay_np = image_np.copy()
        unit_instance_mask = np.zeros((H, W), dtype=np.uint16)

        unit_masks = []
        image_rows = []
        image_json = []

        global_unit_id = 0

        # -----------------------------------------
        # 4. Assign units to building ground-floor ROIs
        # -----------------------------------------

        for u_box, u_score, u_label in zip(u_boxes, u_scores, u_labels):
            area_frac = (
                (u_box[2] - u_box[0]) *
                (u_box[3] - u_box[1])
            ) / img_area

            if area_frac < MIN_UNIT_AREA_FRAC:
                continue

            best_building = None
            best_score = 0.0

            for b in building_ground_masks:
                overlap = box_overlap_with_mask(u_box, b["ground_mask"])
                center_inside = box_center_inside_mask(u_box, b["ground_mask"])

                assignment_score = overlap + (0.5 if center_inside else 0.0)

                if assignment_score > best_score:
                    best_score = assignment_score
                    best_building = b

            if best_building is None:
                continue

            if best_score < MIN_OVERLAP_WITH_GROUND_FLOOR:
                continue

            # SAM2 mask for proposed commercial unit
            u_masks_sam, u_mask_scores, _ = predictor.predict(
                box=u_box,
                multimask_output=False,
            )

            unit_mask = u_masks_sam[0].astype(bool)

            # constrain unit mask inside ground-floor ROI
            unit_mask = unit_mask & best_building["ground_mask"]

            unit_area = int(unit_mask.sum())

            if unit_area == 0:
                continue

            global_unit_id += 1

            building_id = best_building["building_id"]
            unit_id = f"Unit_{global_unit_id:03d}"

            color = np.array(
                UNIT_COLORS[(global_unit_id - 1) % len(UNIT_COLORS)],
                dtype=np.uint8,
            )

            overlay_np[unit_mask] = (
                (1 - ALPHA) * image_np[unit_mask]
                + ALPHA * color
            ).astype(np.uint8)

            unit_instance_mask[unit_mask] = global_unit_id
            unit_masks.append(unit_mask)

            ys, xs = np.where(unit_mask)

            crop, crop_bbox = crop_from_mask(image, unit_mask, padding=25)

            crop_path = ""

            if crop is not None:
                crop_filename = f"{image_path.stem}_{building_id}_{unit_id}.png"
                crop_path_obj = CROP_DIR / crop_filename
                crop.save(crop_path_obj)
                crop_path = str(crop_path_obj)

            row = {
                "image": image_path.name,
                "stem": image_path.stem,
                "building_id": building_id,
                "unit_id": unit_id,
                "raw_label": u_label,
                "dino_score": float(u_score),
                "sam2_unit_mask_score": float(u_mask_scores[0]),
                "assignment_score": float(best_score),
                "unit_area": unit_area,
                "unit_box_xmin": float(u_box[0]),
                "unit_box_ymin": float(u_box[1]),
                "unit_box_xmax": float(u_box[2]),
                "unit_box_ymax": float(u_box[3]),
                "unit_mask_xmin": int(xs.min()),
                "unit_mask_ymin": int(ys.min()),
                "unit_mask_xmax": int(xs.max()),
                "unit_mask_ymax": int(ys.max()),
                "crop_path": crop_path,
                "building_area": best_building["building_area"],
                "ground_area": best_building["ground_area"],
                **best_building["ground_info"],
            }

            if crop_bbox is not None:
                row.update(crop_bbox)

            image_rows.append(row)
            all_rows.append(row)

            image_json.append({
                "image": image_path.name,
                "building_id": building_id,
                "unit_id": unit_id,
                "raw_label": str(u_label),
                "dino_score": float(u_score),
                "sam2_unit_mask_score": float(u_mask_scores[0]),
                "assignment_score": float(best_score),
                "unit_area": unit_area,
                "crop_path": crop_path,
                "bbox": {
                    "xmin": float(u_box[0]),
                    "ymin": float(u_box[1]),
                    "xmax": float(u_box[2]),
                    "ymax": float(u_box[3]),
                },
            })

            print(
                building_id,
                unit_id,
                u_label,
                "dino=", round(float(u_score), 3),
                "assign=", round(float(best_score), 3),
                "area=", unit_area,
            )

        # -----------------------------------------
        # 5. Save outputs
        # -----------------------------------------

        stem = image_path.stem

        overlay_path = OVERLAY_DIR / f"{stem}_urban_scene_units_overlay.png"
        instance_path = MASK_DIR / f"{stem}_unit_instance_mask.npy"
        binary_path = MASK_DIR / f"{stem}_unit_binary_masks.npy"
        csv_path = CSV_DIR / f"{stem}_commercial_units.csv"
        json_path = JSON_DIR / f"{stem}_commercial_units.json"

        Image.fromarray(overlay_np).save(overlay_path)
        np.save(instance_path, unit_instance_mask)

        if unit_masks:
            np.save(binary_path, np.stack(unit_masks, axis=0))

        pd.DataFrame(image_rows).to_csv(csv_path, index=False)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(image_json, f, indent=2, ensure_ascii=False)

        if image_rows:
            summary = (
                pd.DataFrame(image_rows)
                .groupby("building_id")["unit_id"]
                .count()
                .reset_index(name="commercial_unit_count")
            )
        else:
            summary = pd.DataFrame(
                columns=["building_id", "commercial_unit_count"]
            )

        summary_path = CSV_DIR / f"{stem}_unit_count_by_building.csv"
        summary.to_csv(summary_path, index=False)

        print("Saved overlay:", overlay_path)
        print("Saved CSV:", csv_path)
        print("Saved JSON:", json_path)

    combined_csv = OUT_DIR / "all_commercial_units.csv"
    pd.DataFrame(all_rows).to_csv(combined_csv, index=False)

    print("\nDone.")
    print("Saved combined CSV:", combined_csv)


if __name__ == "__main__":
    main()
