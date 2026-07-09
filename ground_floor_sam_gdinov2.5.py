#!/usr/bin/env python3
"""
Building Segmentation & Ground-Floor Use Classification
Grounding DINO (SwinB) + SAM (ViT-H) + CLIP + PaddleOCR PP-OCRv6

Uses the two *original* detection/segmentation checkpoints directly:
  - pretrained_model/groundingdino_swinb_cogcoor.pth  -> open-vocabulary DETECTION
  - pretrained_model/sam_vit_h_4b8939.pth             -> promptable SEGMENTATION
plus CLIP (openai/clip-vit-base-patch32) for ground-floor USE classification and
PaddleOCR PP-OCRv6 (PP-OCRv6_medium_det + PP-OCRv6_medium_rec, el/en multilingual)
to read any sign/storefront text out of each building's ground-floor mask.

Output is rendered in the project's reference style (see output/01.png, output/02.png):
  * each building is segmented (SAM) and drawn with a translucent instance colour
    + a WHITE dashed "Building Boundary (Segmented)" outline,
  * the GROUND-FLOOR region of each building is drawn as a YELLOW dashed
    "Ground Floor Region (Evaluated)" band (the bottom GF_FRACTION of the facade),
  * a "B#" tag sits on top of each building,
  * a label box per building gives its Ground Floor Use, any OCR'd sign text,
    and Confidence (CLIP),
  * a legend (top-left) and summary (top-right) frame the panel.

Pipeline per image:
  1. Grounding DINO detects each building/facade (prompt "building. house.").
  2. SAM box-prompts each detection -> that building's exact silhouette.
  3. Ground floor = bottom GF_FRACTION of the building mask's height.
  4. CLIP classifies the ground-floor crop into a use category + confidence.
  5. PaddleOCR PP-OCRv6 reads any sign/storefront text on the ground-floor mask.
  6. Reference-style annotated image is saved to results/.
  7. (optional, --use-vlm) A local VLM (Qwen2.5-VL) re-inspects each building's
     ground-floor band, splits it into separate shop UNITS where warranted, and
     re-classifies each unit's use, fusing its verdict with CLIP's.

Run (use the quasar env python, which has torch/transformers/the two packages):
    python ground_floor_sam_gdino.py [IMAGE] [--out OUT.png] [--prompt "building. house."]
                                     [--gf-fraction 0.30] [--box-threshold 0.35]
                                     [--text-threshold 0.25]
                                     [--use-vlm] [--vlm-model Qwen/Qwen2.5-VL-3B-Instruct]
                                     [--no-vlm-4bit]
"""

import argparse
import json
import os
import re
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(BASE_DIR / ".cache"))   # BERT + CLIP downloads stay in-project

PRETRAINED_DIR = BASE_DIR / "pretrained_model"
GDINO_CKPT = PRETRAINED_DIR / "groundingdino_swinb_cogcoor.pth"
SAM_CKPT = PRETRAINED_DIR / "sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"
CLIP_ID = "openai/clip-vit-base-patch32"
SEGFORMER_ID = "nvidia/segformer-b5-finetuned-ade-640-640"   # ADE20K semantic guard

# ADE20K classes that make a SAM mask a real facade vs. a non-building surface.
# This is what rejects the arcade ceiling / concrete pillar / floor that SAM
# otherwise happily segments inside a covered passage (stoa).
FACADE_IDS = {0, 1, 8, 14, 25, 43, 48, 58, 86}   # wall, building, window, door, house, signboard, skyscraper, screen-door, awning
NON_FACADE_IDS = {2, 3, 4, 5, 6, 11, 13, 42}     # sky, floor, tree, ceiling, road, sidewalk, earth, column/pillar
MIN_FACADE_FRAC = 0.50                            # a mask below this share of facade pixels is dropped
                                                  # (real facades score >90%; arcade ceiling/pillar <=45%)
# Single-storey (whole segment = ground floor) vs multi-storey (bottom band only).
# Storey count can't be read reliably from one 2D image (a 1-storey shop close up
# looks like a 5-storey block far away), but the scene type can: under a covered
# arcade (stoa) the upper floors are hidden by the roof, so the strip just above a
# facade is CEILING, whereas an open street shows SKY there.
CEILING_ID, SKY_ID = 5, 2
ARCADE_STRIP_FRAC = 0.15                           # height of the strip checked just above each facade
ARCADE_CEIL_MIN = 0.15                             # min ceiling share in that strip to call it covered
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

RESULTS_DIR = BASE_DIR / "results" / "ground_floor_sam_gdino"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Grounding DINO / ground-floor defaults --------------------------------------
DEFAULT_PROMPT = "building. house."   # short prompt keeps Grounding DINO recall high;
                                      # extra phrases (storefront/shop) were found to REDUCE it
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
GF_FRACTION = 0.30                    # bottom-fraction fallback when no shopfront features are found
MAX_IMAGE_SIDE = 1536                 # downscale longest side before inference; 0 = keep original
MIN_GROUND_FLOOR_PIXELS = 300         # below this, don't attempt CLIP classification

# Feature-based ground floor: a second open-vocabulary Grounding DINO pass finds
# the actual shopfront features (display window, door, shutter, sign, awning).
# The ground floor is then the band where those features are, scanned up from the
# street -- not a blind fraction of the building height (which cut shops in half).
FEATURE_PROMPT = "shop window. storefront. door. roller shutter. signboard. awning. shop sign."
FEATURE_BOX_THRESHOLD = 0.25
FEATURE_TEXT_THRESHOLD = 0.20
FEAT_SUPPORT_MIN = 0.12               # min per-row feature coverage (of building width) to be ground floor
FEAT_GAP_FRAC = 0.08                  # tolerate feature gaps up to this fraction of building height

# Reject degenerate detections/segmentations. A single building is never the
# whole 360-deg panorama; SAM box-prompted with a giant box just segments the
# dominant ground/sky plane, so both the box and the resulting mask are capped.
MAX_BOX_AREA_FRAC = 0.55             # drop Grounding DINO boxes larger than this (whole-scene boxes)
MAX_BOX_WIDTH_FRAC = 0.60            # ... or wider than this fraction of the image
MIN_BUILDING_AREA_FRAC = 0.004       # drop SAM masks smaller than this
MAX_BUILDING_MASK_FRAC = 0.45        # drop SAM masks bigger than this (ground / road / sky planes)
MIN_FACADE_ASPECT = 0.30             # drop tall/thin masks (arcade columns/pillars/poles); a
                                     # concrete pillar is ~0.23 wide-to-tall, real facades >=0.43,
                                     # and SegFormer labels the pillar "building" so shape is the only tell
# Panoramas are ~2:1; run detection on overlapping square windows so "building"
# localises to one facade instead of matching the entire arcade/street scene.
TILE_ASPECT_TRIGGER = 1.6            # tile detection when width/height exceeds this
TILE_OVERLAP = 0.5                   # window stride = tile_size * (1 - overlap)
NMS_IOU = 0.5                        # de-duplicate boxes across windows
NMS_CONTAINMENT = 0.85              # ... and drop a box mostly swallowed by a bigger kept one

# CLIP ground-floor use classification (mirrors pipeline.py) -------------------
USE_CLASSES = [
    "Food & Beverage", "Retail/Utility", "Health and wellness establishments",
    "Beauty and fashion boutiques", "Professional services", "Parking lot",
    "Accommodation", "Workshops", "Open Public Space", "Closed Public Space",
    "Other", "Vacant",
]
COMMERCIAL_USE_CLASSES = {
    "Food & Beverage", "Retail/Utility", "Health and wellness establishments",
    "Beauty and fashion boutiques", "Professional services", "Accommodation", "Workshops",
}
CLIP_TEMPLATES = [
    "a street-level photo of the ground floor of a building used as a {}",
    "a photo of a {} storefront",
    "a photo of a {}",
]
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

# PaddleOCR PP-OCRv6 (detection + recognition, "transformers" backend so it
# reuses the torch/transformers stack already loaded here) reads sign/
# storefront text out of each building's ground-floor mask. PP-OCRv6_medium_rec
# is a single multilingual recognizer covering both Greek (el) and English
# (en), so no per-language model switching is needed. Purely additive: if the
# pipeline can't be loaded (missing paddleocr / no download), OCR is skipped
# and every building just gets an empty sign_text.
OCR_DET_MODEL = "PP-OCRv6_medium_det"
OCR_REC_MODEL = "PP-OCRv6_medium_rec"
OCR_MIN_SCORE = 0.5                   # drop low-confidence recognitions
OCR_TEXT_BLOCKLIST = ("googl",)       # Street View "(c) 2024 Google" watermark (recognizer sometimes
                                      # drops the trailing "e"); "googl" substring-matches both forms

# -- Step 7: VLM ground-floor correction & unit splitting ---------------------
# Grounding DINO/SAM give one ground-floor MASK per building, and CLIP gives it
# ONE use-class for the whole band. Real facades often hold several adjoining
# shops (one mask, several businesses), and CLIP's zero-shot label is brittle
# on cluttered storefronts. A small local VLM re-inspects the ground-floor crop
# (with the CLIP verdict + OCR text as context) and returns a corrected split
# into shop UNITS, each with its own use-class + confidence + evidence.
#
# This is purely additive and OPTIONAL (--use-vlm): if the model can't be
# loaded (missing transformers version, OOM, no internet), it is disabled and
# every building just keeps its single CLIP-only verdict, same as before.
#
# Sized for an 8GB card: Qwen2.5-VL-3B-Instruct in 4-bit (bitsandbytes NF4)
# uses roughly 2.5-3.5GB VRAM, on top of the ~4-5GB the GDINO+SAM+SegFormer+
# CLIP stack already occupies in fp32 -- tight but workable for testing on a
# handful of images. Drop to VLM_MODEL_ID="Qwen/Qwen2.5-VL-3B-Instruct" (kept
# below) if a 7B model OOMs; the 7B AWQ checkpoint is a viable upgrade on a
# 12GB+ card.
VLM_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
VLM_4BIT = True                       # NF4 quantization via bitsandbytes; set False only on >=16GB cards
VLM_MAX_UNITS = 6                     # sanity cap: one facade is never split into more shops than this
VLM_CROP_MAX_SIDE = 896               # resize a crop before feeding the VLM
VLM_MIN_UNIT_FRAC = 0.04              # drop degenerate VLM unit splits narrower than this fraction of width
VLM_UNIT_PAD_FRAC = 0.06              # pad each unit's own crop a bit so edge content isn't cut off

# Splitting + 12-way classification + OCR-grounding + anti-copying in ONE
# compound call was unreliable on this 3B/4-bit model in practice: it tended
# to echo nearby prompt text back as "evidence" (the few-shot example, or the
# CLIP-baseline sentence) instead of reasoning from the image, and legible OCR
# text (e.g. "nail salon") often went unused. Splitting into two much simpler,
# single-purpose calls per building is more reliable:
#   1. SPLIT: purely spatial/visual -- "where are the unit boundaries?" --
#      no classification, no OCR, nothing to copy.
#   2. CLASSIFY: one call PER unit, on a tight crop of just that unit, with
#      the OCR text as supporting context -- one focused decision at a time.
VLM_SPLIT_MAX_NEW_TOKENS = 256
VLM_CLASSIFY_MAX_NEW_TOKENS = 150

VLM_SPLIT_PROMPT_TEMPLATE = """You are looking at one building's ground-floor storefront band, cropped from a street-level photo. Image width represents the full horizontal extent of this building's ground floor.

Identify how many separate shop units are visible side by side (a single facade may hold several adjoining shops, separated by pillars, doors, gaps, or a clear change in window/signage style).

For each unit, moving left to right, give its horizontal extent as a fraction of the image width (0.0 = left edge, 1.0 = right edge).

Rules:
- If it really is one shop, return exactly one unit spanning x_start=0.0 to x_end=1.0.
- Units must be ordered left to right, non-overlapping, and together should span close to the full width.
- Return at most {max_units} units.
- Return ONLY a JSON array of objects, each with just "x_start" and "x_end" (numbers 0.0-1.0). No prose,
  no markdown fences, no other fields.
"""

VLM_CLASSIFY_PROMPT_TEMPLATE = """You are looking at a close-up crop of ONE ground-floor shop unit from a street-level photo.

OCR text detected somewhere on this building's ground floor (may or may not belong to this specific unit;
may be empty, partial, garbled, or contain unrelated street clutter like watermarks or plate numbers): "{ocr_text}"

If a word or phrase in that OCR text clearly names a business, product, or service and looks like it belongs
to what you see in THIS crop (e.g. "nail salon", "pharmacy", "bakery", "ATM", "hotel", "garage"), use it as
your primary evidence, even when the generic visual impression alone would suggest something else. Ignore
OCR fragments that are clearly noise (street numbers, watermarks, unrelated stray words) or don't seem to
belong to this crop.

Classify this unit into EXACTLY ONE of these categories: {classes}

Guidance for mapping what you see/read to these categories:
  - food, drink, cafe, restaurant, taverna, bakery, bar -> "Food & Beverage"
  - general shops, kiosks, minimarkets, hardware, utility stores -> "Retail/Utility"
  - pharmacy, clinic, dentist, gym, wellness -> "Health and wellness establishments"
  - hairdresser, nail salon, cosmetics, clothing, jewelry -> "Beauty and fashion boutiques"
  - bank, ATM, office, real estate, insurance, legal -> "Professional services"
  - open-air parking, garage door, car park -> "Parking lot"
  - hotel, hostel, short-term rooms -> "Accommodation"
  - repair shop, workshop, small manufacturing -> "Workshops"
  - a plaza, square, park, or other open space with no building use -> "Open Public Space"
  - a closed/gated public building (school, church, civic building) -> "Closed Public Space"
  - clearly empty, no signage, shuttered with no business indication -> "Vacant"
  - anything that doesn't fit the above -> "Other"

Respond with ONLY a JSON object (no prose, no markdown fences) with exactly these keys: "use_class" (one of
the categories above, copied exactly), "confidence" (a number 0.0-1.0), and "evidence" (a short phrase
describing the specific thing YOU saw or read in THIS crop -- do not repeat these instructions back).
"""

PALETTE = [
    (60, 220, 90), (60, 140, 235), (235, 70, 130), (215, 175, 40),
    (200, 90, 220), (235, 130, 40), (90, 195, 205), (170, 90, 60),
]
GROUND_FLOOR_COLOR = (255, 221, 40)                       # yellow -> "Ground Floor Region (Evaluated)"
BUILDING_BOUNDARY_COLOR = (235, 235, 235)                 # white  -> "Building Boundary (Segmented)"
VLM_UNIT_COLOR = (255, 90, 90)                            # red    -> VLM-drawn unit separators
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def log(msg):
    print(msg, flush=True)


def _font(size):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def _wrap_text(text, font, max_w, draw):
    words, lines, cur = text.split(), [], ""
    for wd in words:
        trial = (cur + " " + wd).strip()
        if draw.textlength(trial, font=font) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = wd
    if cur:
        lines.append(cur)
    return lines


def _truncate_text(text, font, max_w, draw):
    """One-line truncation with a trailing ellipsis -- guarantees the returned
    string fits max_w, so a single-line caller never silently loses text past
    the cutoff (unlike _wrap_text, which drops overflow words on the floor)."""
    if draw.textlength(text, font=font) <= max_w:
        return text
    while text and draw.textlength(text + "...", font=font) > max_w:
        text = text[:-1]
    return (text + "...") if text else "..."


# ------------------------------------------------------------------------------
# Compatibility shim: groundingdino-py 0.4.0 targets transformers 4.x; its
# BertModelWarper uses BertModel.get_head_mask / return_dict, removed in
# transformers 5.x. Replace it with a minimal warper that drives the (still
# present) embeddings + encoder directly and builds the additive attention mask
# itself (Grounding DINO passes a custom 3-D per-token mask; only
# last_hidden_state is consumed downstream). In-memory rebind, nothing on disk.
# ------------------------------------------------------------------------------
def _patch_groundingdino_for_transformers5():
    import groundingdino.models.GroundingDINO.bertwarper as bw
    import groundingdino.models.GroundingDINO.groundingdino as gd

    class _BertModelWarper(torch.nn.Module):
        def __init__(self, bert_model):
            super().__init__()
            self.config = bert_model.config
            self.embeddings = bert_model.embeddings
            self.encoder = bert_model.encoder
            self.pooler = bert_model.pooler

        def _extended_mask(self, attention_mask, dtype):
            ext = attention_mask[:, None, :, :] if attention_mask.dim() == 3 \
                else attention_mask[:, None, None, :]
            ext = ext.to(dtype=dtype)
            return (1.0 - ext) * torch.finfo(dtype).min

        def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                    position_ids=None, **kwargs):
            shape, device = input_ids.size(), input_ids.device
            if attention_mask is None:
                attention_mask = torch.ones(shape, device=device)
            if token_type_ids is None:
                token_type_ids = torch.zeros(shape, dtype=torch.long, device=device)
            dtype = next(self.encoder.parameters()).dtype
            emb = self.embeddings(input_ids=input_ids, position_ids=position_ids,
                                  token_type_ids=token_type_ids)
            enc = self.encoder(emb, attention_mask=self._extended_mask(attention_mask, dtype))
            last = enc.last_hidden_state if hasattr(enc, "last_hidden_state") else enc[0]
            return {"last_hidden_state": last}

    bw.BertModelWarper = _BertModelWarper
    gd.BertModelWarper = _BertModelWarper


# ------------------------------------------------------------------------------
class GroundFloorSegmenter:
    def __init__(self, device, use_vlm=False, vlm_model_id=VLM_MODEL_ID, vlm_4bit=VLM_4BIT):
        self.device = device
        _patch_groundingdino_for_transformers5()
        import groundingdino
        from groundingdino.util.inference import load_model as gdino_load
        from segment_anything import sam_model_registry, SamPredictor
        from transformers import CLIPModel, CLIPTokenizer, SegformerForSemanticSegmentation

        cfg = os.path.join(os.path.dirname(groundingdino.__file__),
                           "config", "GroundingDINO_SwinB_cfg.py")
        log(f"Loading Grounding DINO SwinB ({GDINO_CKPT.name}) ...")
        self.gdino = gdino_load(cfg, str(GDINO_CKPT), device=device)

        log(f"Loading SAM {SAM_MODEL_TYPE} ({SAM_CKPT.name}) ...")
        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=str(SAM_CKPT)).to(device)
        self.sam = SamPredictor(sam)

        log(f"Loading SegFormer semantic guard ({SEGFORMER_ID}) ...")
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_ID).eval().to(device)

        log(f"Loading CLIP ({CLIP_ID}) ...")
        self.clip = CLIPModel.from_pretrained(CLIP_ID).eval().to(device)
        self.clip_tok = CLIPTokenizer.from_pretrained(CLIP_ID)
        self.use_text_embeds = self._encode_use_classes()

        self.ocr = None
        self._load_ocr()

        # VLM is loaded lazily and is entirely optional -- a failure here
        # (missing package, OOM, no internet) just disables step 7, it never
        # crashes the rest of the pipeline.
        self.vlm = None
        self.vlm_processor = None
        self.vlm_model_id = vlm_model_id
        self.vlm_4bit = vlm_4bit
        if use_vlm:
            self._load_vlm()

    def _load_ocr(self):
        try:
            from paddleocr import PaddleOCR
            log(f"Loading PaddleOCR ({OCR_DET_MODEL} + {OCR_REC_MODEL}, el/en multilingual) ...")
            paddle_device = "gpu" if self.device == "cuda" else self.device
            self.ocr = PaddleOCR(
                text_detection_model_name=OCR_DET_MODEL,
                text_recognition_model_name=OCR_REC_MODEL,
                engine="transformers",             # reuses our torch/transformers stack, no paddlepaddle needed
                device=paddle_device,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        except Exception as e:                   # missing package / no network / OOM -> OCR is optional
            log(f"  !! PaddleOCR unavailable, sign-text reading disabled: {e}")
            self.ocr = None

    # -- Step 7 setup: load the VLM (optional, quantized for small GPUs) ------
    def _load_vlm(self):
        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            quant_kwargs = {}
            if self.vlm_4bit and self.device == "cuda":
                from transformers import BitsAndBytesConfig
                quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            else:
                quant_kwargs["torch_dtype"] = torch.float16 if self.device == "cuda" else torch.float32
            log(f"Loading VLM ({self.vlm_model_id}, 4bit={self.vlm_4bit}) for ground-floor correction ...")
            self.vlm_processor = AutoProcessor.from_pretrained(self.vlm_model_id, min_pixels=256 * 28 * 28,
                                                               max_pixels=VLM_CROP_MAX_SIDE * VLM_CROP_MAX_SIDE)
            self.vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.vlm_model_id,
                device_map=self.device if self.device == "cuda" else None,
                **quant_kwargs,
            ).eval()
            if self.device != "cuda":
                self.vlm = self.vlm.to(self.device)
        except Exception as e:                   # missing package / wrong transformers version / OOM
            log(f"  !! VLM unavailable, ground-floor correction (step 7) disabled: {e}")
            self.vlm, self.vlm_processor = None, None

    def _encode_use_classes(self):
        with torch.no_grad():
            all_embeds = []
            for cls in USE_CLASSES:
                tok = self.clip_tok([t.format(cls) for t in CLIP_TEMPLATES],
                                    padding=True, return_tensors="pt").to(self.device)
                emb = self.clip.text_projection(self.clip.text_model(**tok).pooler_output)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                all_embeds.append(emb.mean(dim=0))
            text_embeds = torch.stack(all_embeds)
            return text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    # -- Step 1: Grounding DINO detection --------------------------------------
    def _detect_one(self, image_rgb, prompt, box_threshold, text_threshold):
        """Grounding DINO on a single image -> [(xyxy, score), ...] in its coords."""
        from groundingdino.util.inference import predict
        import groundingdino.datasets.transforms as T

        transform = T.Compose([
            T.RandomResize([800], max_size=1333), T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_tensor, _ = transform(Image.fromarray(image_rgb), None)
        boxes, logits, _ = predict(
            model=self.gdino, image=image_tensor, caption=prompt,
            box_threshold=box_threshold, text_threshold=text_threshold, device=self.device)
        h, w = image_rgb.shape[:2]
        out = []
        for (cx, cy, bw_, bh), score in zip(boxes.cpu().numpy(), logits.cpu().numpy()):
            x0, y0 = int((cx - bw_ / 2) * w), int((cy - bh / 2) * h)
            x1, y1 = int((cx + bw_ / 2) * w), int((cy + bh / 2) * h)
            x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w - 1), min(y1, h - 1)
            if x1 > x0 and y1 > y0:
                out.append(([x0, y0, x1, y1], float(score)))
        return out

    @staticmethod
    def _iou_containment(a, b):
        ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
        ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
        aa = (a[2] - a[0]) * (a[3] - a[1])
        ab = (b[2] - b[0]) * (b[3] - b[1])
        union = aa + ab - inter
        iou = inter / union if union > 0 else 0.0
        contain = inter / min(aa, ab) if min(aa, ab) > 0 else 0.0
        return iou, contain

    def detect(self, image_rgb, prompt, box_threshold, text_threshold, tile=True):
        """Detect buildings. Panoramas (wide aspect) are scanned with overlapping
        square windows so each facade is detected on its own instead of the whole
        scene collapsing into one giant box; degenerate (near-full-frame) boxes
        are dropped, then boxes from all windows are de-duplicated."""
        h, w = image_rgb.shape[:2]
        # Detect on the whole image (good recall on normal photos & wide facades)
        # AND, for panoramas, on overlapping square windows (localises each facade
        # so the scene never collapses into one giant box). Union, then filter/NMS.
        raw = list(self._detect_one(image_rgb, prompt, box_threshold, text_threshold))
        if tile and w / h > TILE_ASPECT_TRIGGER:
            tile_w = h                                   # square windows, full height
            stride = max(1, int(tile_w * (1 - TILE_OVERLAP)))
            xs = list(range(0, max(1, w - tile_w + 1), stride))
            if not xs or xs[-1] != w - tile_w:
                xs.append(max(0, w - tile_w))
            for x in xs:
                crop = image_rgb[:, x:x + tile_w]
                for box, score in self._detect_one(crop, prompt, box_threshold, text_threshold):
                    raw.append(([box[0] + x, box[1], box[2] + x, box[3]], score))

        # drop whole-scene / oversized boxes
        kept = []
        for box, score in raw:
            bw_, bh_ = box[2] - box[0], box[3] - box[1]
            if bw_ * bh_ > MAX_BOX_AREA_FRAC * h * w or bw_ > MAX_BOX_WIDTH_FRAC * w:
                continue
            kept.append((box, score))

        # NMS across windows: highest score first, drop duplicates / contained boxes
        kept.sort(key=lambda bs: -bs[1])
        deduped = []
        for box, score in kept:
            if all(not (self._iou_containment(box, kb)[0] > NMS_IOU or
                        self._iou_containment(box, kb)[1] > NMS_CONTAINMENT) for kb, _ in deduped):
                deduped.append((box, score))
        return deduped

    # -- Semantic guard: ADE20K label map (SegFormer) --------------------------
    def semantic_label_map(self, image_rgb):
        h, w = image_rgb.shape[:2]
        resized = cv2.resize(image_rgb, (640, 640), interpolation=cv2.INTER_LINEAR)
        x = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.segformer(pixel_values=x).logits
            logits = torch.nn.functional.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
            return logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    # -- Step 2 + 3: SAM segmentation + ground-floor clip ----------------------
    def _building_mask(self, box, h, w):
        """SAM box-prompt -> one clean facade mask. Ask for 3 candidate masks and
        take the highest-quality one that is NOT a background plane (a facade never
        fills ~half the frame), then hard-clip it to the detection box and keep the
        largest connected component so ground / road spill is removed."""
        x0, y0, x1, y1 = box
        masks, scores, _ = self.sam.predict(box=np.asarray(box), multimask_output=True)
        rect = np.zeros((h, w), dtype=bool)
        rect[y0:y1, x0:x1] = True

        best, best_score = None, -1.0
        for mi in range(masks.shape[0]):
            m = masks[mi].astype(bool)
            if m.sum() > MAX_BUILDING_MASK_FRAC * h * w:      # ground / sky / road plane
                continue
            if scores[mi] > best_score:
                best, best_score = m, float(scores[mi])
        if best is None:                                       # all degenerate -> smallest, clipped
            areas = [masks[k].sum() for k in range(masks.shape[0])]
            best = masks[int(np.argmin(areas))].astype(bool)

        m = best & rect                                        # never leave the detection box
        num, lab = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
        if num > 2:
            sizes = [(lab == k).sum() for k in range(1, num)]
            m = lab == (1 + int(np.argmax(sizes)))
        return m

    def _ground_floor_region(self, mask, feat_mask, single_storey, gf_fraction):
        """Ground floor = the band where shopfront FEATURES are, scanned up from
        the street. Arcade single-storey shops -> the whole segment. Multi-storey
        -> the contiguous bottom band that has enough feature support (detected
        shop windows / doors / shutters / signs). If no features land on the
        building, fall back to the bottom-fraction heuristic."""
        rows = np.where(mask.any(axis=1))[0]
        y_top, y_bot = int(rows.min()), int(rows.max())
        bh = y_bot - y_top + 1
        gf = mask.copy()
        if single_storey:
            return gf, "whole (arcade)"

        width = mask.sum(axis=1).astype(np.float32)
        support = (feat_mask & mask).sum(axis=1) / np.clip(width, 1, None)
        if (feat_mask & mask).sum() >= MIN_GROUND_FLOOR_PIXELS:
            gap_tol = max(3, int(FEAT_GAP_FRAC * bh))
            gf_top, gap = y_bot, 0
            for r in range(y_bot, y_top - 1, -1):           # walk up from the street
                if width[r] == 0:
                    continue
                if support[r] >= FEAT_SUPPORT_MIN:
                    gf_top, gap = r, 0
                else:
                    gap += 1
                    if gap > gap_tol:                        # features have ended -> top of ground floor
                        break
            if y_bot - gf_top >= 0.06 * bh:                  # a usable feature band was found
                gf[:gf_top] = False
                return gf, "feature-based"
        band_top = int(round(y_bot - gf_fraction * bh))      # fallback: no features on this facade
        gf[:band_top] = False
        return gf, f"bottom {int(gf_fraction * 100)}% (no features)"

    def segment(self, image_rgb, boxes, gf_fraction, feature_boxes=None):
        h, w = image_rgb.shape[:2]
        self.sam.set_image(image_rgb)
        label_map = self.semantic_label_map(image_rgb)
        facade_lut = np.zeros(256, dtype=bool)
        facade_lut[list(FACADE_IDS)] = True
        feat_mask = np.zeros((h, w), dtype=bool)             # union of detected shopfront-feature boxes
        for fb, _ in (feature_boxes or []):
            fx0, fy0, fx1, fy1 = fb
            feat_mask[fy0:fy1, fx0:fx1] = True
        results = []
        for box, score in boxes:
            building_mask = self._building_mask(box, h, w)
            area = building_mask.sum()
            if area < MIN_BUILDING_AREA_FRAC * h * w or area > MAX_BUILDING_MASK_FRAC * h * w:
                continue                                       # too small, or still a background plane
            # semantic guard: drop masks that are mostly ceiling / pillar / floor /
            # sky (SAM segments these happily inside a covered arcade, but they are
            # not building facades). Keep only masks with enough real facade pixels.
            facade_frac = facade_lut[label_map[building_mask]].mean()
            if facade_frac < MIN_FACADE_FRAC:
                continue
            rows = np.where(building_mask.any(axis=1))[0]
            cols = np.where(building_mask.any(axis=0))[0]
            y_top, y_bot = int(rows.min()), int(rows.max())
            x_lo, x_hi = int(cols.min()), int(cols.max())
            if (x_hi - x_lo + 1) / (y_bot - y_top + 1) < MIN_FACADE_ASPECT:
                continue                                       # tall, thin column / pillar / pole

            # Covered arcade? Inspect the strip directly above the facade: ceiling
            # there means the upper storeys are hidden by the arcade roof, so the
            # whole visible segment is the ground floor. Sky there means an open
            # multi-storey street facade -> keep only the bottom band.
            strip = label_map[max(0, y_top - int(ARCADE_STRIP_FRAC * h)):y_top, x_lo:x_hi + 1]
            ceil_frac = float((strip == CEILING_ID).mean()) if strip.size else 0.0
            sky_frac = float((strip == SKY_ID).mean()) if strip.size else 0.0
            single_storey = ceil_frac > sky_frac and ceil_frac > ARCADE_CEIL_MIN

            gf_mask, gf_method = self._ground_floor_region(
                building_mask, feat_mask, single_storey, gf_fraction)
            results.append({"box": box, "score": score, "mask": building_mask,
                            "gf_mask": gf_mask, "single_storey": single_storey,
                            "gf_method": gf_method})
        return results

    # -- Step 4: CLIP ground-floor use classification --------------------------
    def classify_use(self, image_rgb, region_mask):
        ys, xs = np.where(region_mask)
        if len(ys) < MIN_GROUND_FLOOR_PIXELS:
            return "unknown", 0.0, "low"
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        crop = image_rgb[y0:y1 + 1, x0:x1 + 1]
        ch, cw = crop.shape[:2]
        side = max(ch, cw)
        padded = cv2.copyMakeBorder(crop, (side - ch) // 2, side - ch - (side - ch) // 2,
                                    (side - cw) // 2, side - cw - (side - cw) // 2,
                                    borderType=cv2.BORDER_REPLICATE)
        resized = cv2.resize(padded, (224, 224), interpolation=cv2.INTER_LINEAR)
        x = (resized.astype(np.float32) / 255.0 - CLIP_MEAN) / CLIP_STD
        x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embed = self.clip.visual_projection(self.clip.vision_model(pixel_values=x).pooler_output)
            embed = embed / embed.norm(dim=-1, keepdim=True)
            sims = (self.clip.logit_scale.exp() * embed @ self.use_text_embeds.T).softmax(dim=-1)[0]
        top = int(sims.argmax())
        conf = float(sims[top])
        bucket = "high" if conf >= 0.42 else ("medium" if conf >= 0.25 else "low")
        return USE_CLASSES[top], conf, bucket

    # -- Step 5: PaddleOCR PP-OCRv6 sign/storefront text reading ---------------
    def read_sign_text(self, image_rgb, region_mask, full_res_image=None, scale=1.0):
        """Crop one building's ground-floor mask and read any sign/storefront
        text on it with PaddleOCR PP-OCRv6 (text detection + el/en multilingual
        recognition). Returns "" if OCR wasn't loaded, the crop is too small,
        or no text was found above OCR_MIN_SCORE.

        If full_res_image/scale are given, the crop is re-extracted from the
        ORIGINAL (pre-downscale) panorama instead of the working-resolution
        image_rgb -- signage is a tiny fraction of an 8192x4096 GSV panorama
        and is already illegible after the pipeline's MAX_IMAGE_SIDE downscale
        (empirically: OCR on the downscaled crop finds nothing; the identical
        region at full resolution reads real shop text)."""
        if self.ocr is None:
            return ""
        ys, xs = np.where(region_mask)
        if len(ys) < MIN_GROUND_FLOOR_PIXELS:
            return ""
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        if full_res_image is not None and scale != 1.0:
            fh, fw = full_res_image.shape[:2]
            fy0, fy1 = int(y0 / scale), min(fh, int((y1 + 1) / scale))
            fx0, fx1 = int(x0 / scale), min(fw, int((x1 + 1) / scale))
            crop_rgb = full_res_image[fy0:fy1, fx0:fx1]
        else:
            crop_rgb = image_rgb[y0:y1 + 1, x0:x1 + 1]
        crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        res = next(iter(self.ocr.predict(crop_bgr)), None)
        if res is None:
            return ""
        fields = res.json["res"]
        texts, scores = fields.get("rec_texts", []), fields.get("rec_scores", [])
        kept = [t.strip() for t, s in zip(texts, scores)
               if s >= OCR_MIN_SCORE and t.strip()
               and not any(b in t.lower() for b in OCR_TEXT_BLOCKLIST)]
        return " ".join(kept)

    # -- Step 7: VLM ground-floor correction & unit splitting ------------------
    def _vlm_generate(self, crop_rgb, prompt_text, max_new_tokens):
        """One VLM call on a single crop with a given text prompt. Returns the
        raw text response, or None if the VLM isn't loaded / generation failed."""
        if self.vlm is None or self.vlm_processor is None:
            return None
        pil_img = Image.fromarray(crop_rgb)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": pil_img},
            {"type": "text", "text": prompt_text},
        ]}]
        try:
            text_prompt = self.vlm_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = self.vlm_processor(text=[text_prompt], images=[pil_img],
                                        padding=True, return_tensors="pt").to(self.vlm.device)
            with torch.no_grad():
                out_ids = self.vlm.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            trimmed = out_ids[:, inputs["input_ids"].shape[1]:]
            return self.vlm_processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        except Exception as e:
            log(f"    !! VLM generation failed: {e}")
            return None

    @staticmethod
    def _parse_vlm_split(raw_text):
        """Pull a JSON array of {x_start, x_end} out of the split call's raw
        text. Returns [] if nothing parseable was found -- caller falls back
        to a single whole-width unit."""
        if not raw_text:
            return []
        match = re.search(r"\[.*\]", raw_text, re.DOTALL)
        if not match:
            return []
        try:
            raw_units = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
        cleaned = []
        for u in raw_units if isinstance(raw_units, list) else []:
            try:
                x0, x1 = float(u["x_start"]), float(u["x_end"])
            except (KeyError, TypeError, ValueError):
                continue
            x0, x1 = max(0.0, min(x0, 1.0)), max(0.0, min(x1, 1.0))
            if x1 - x0 < VLM_MIN_UNIT_FRAC:
                continue
            cleaned.append({"x_start": x0, "x_end": x1})
        cleaned.sort(key=lambda u: u["x_start"])
        return cleaned

    @staticmethod
    def _parse_vlm_classification(raw_text):
        """Pull a single {use_class, confidence, evidence} JSON object out of
        the classify call's raw text. Returns None if unparseable/invalid."""
        if not raw_text:
            return None
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not match:
            return None
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        try:
            cls = str(obj["use_class"])
            conf = float(obj.get("confidence", 0.5))
            evidence = str(obj.get("evidence", ""))
        except (KeyError, TypeError, ValueError):
            return None
        if cls not in USE_CLASSES:
            return None
        return {"use_class": cls, "confidence": max(0.0, min(conf, 1.0)), "evidence": evidence}

    def vlm_correct_building(self, image_rgb, building, full_res_image=None, scale=1.0):
        """Runs step 7 on one building in two simpler VLM calls instead of one
        compound one (see the comment above VLM_SPLIT_MAX_NEW_TOKENS for why).
        Mutates and returns `building` with:
          - vlm_units: one dict per shop unit (pixel-space x0,x1 within the gf
            crop, use_class, confidence, evidence), or [] if the VLM is
            disabled or every call failed/was unparseable.
          - use_class / confidence / bucket: overwritten with the VLM's verdict
            for the LARGEST unit if the VLM produced a confident, valid result;
            otherwise left as CLIP's original verdict (nothing is lost).

        1. One call on the WHOLE ground-floor crop finds unit boundaries only
           (purely spatial/visual, no classification).
        2. One call PER unit, on a tight crop of just that unit, classifies it.

        If full_res_image/scale are given, crops are re-extracted from the
        ORIGINAL (pre-downscale) panorama instead of the working-resolution
        image_rgb -- same reasoning as read_sign_text: at MAX_IMAGE_SIDE
        downscale, small signage/detail is already illegible.
        """
        building["vlm_units"] = []
        if self.vlm is None:
            return building
        ys, xs = np.where(building["gf_mask"])
        if len(ys) < MIN_GROUND_FLOOR_PIXELS:
            return building
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        use_full_res = full_res_image is not None and scale != 1.0
        src = full_res_image if use_full_res else image_rgb
        src_scale = scale if use_full_res else 1.0

        def to_src(px_x0, px_y0, px_x1, px_y1):
            return (int(px_x0 / src_scale), int(px_y0 / src_scale),
                    int(px_x1 / src_scale), int(px_y1 / src_scale))

        def fit_crop(arr):
            ch, cw = arr.shape[:2]
            if max(ch, cw) > VLM_CROP_MAX_SIDE:
                f = VLM_CROP_MAX_SIDE / max(ch, cw)
                arr = cv2.resize(arr, (max(1, int(cw * f)), max(1, int(ch * f))),
                                 interpolation=cv2.INTER_AREA)
            return arr

        fx0, fy0, fx1, fy1 = to_src(x0, y0, x1 + 1, y1 + 1)
        whole_crop = fit_crop(src[fy0:fy1, fx0:fx1])

        split_prompt = VLM_SPLIT_PROMPT_TEMPLATE.format(max_units=VLM_MAX_UNITS)
        split_raw = self._vlm_generate(whole_crop, split_prompt, VLM_SPLIT_MAX_NEW_TOKENS)
        bounds = self._parse_vlm_split(split_raw) or [{"x_start": 0.0, "x_end": 1.0}]

        ocr_text = building.get("sign_text", "")
        classify_prompt = VLM_CLASSIFY_PROMPT_TEMPLATE.format(ocr_text=ocr_text or "", classes=USE_CLASSES)
        crop_w = x1 - x0 + 1
        units = []
        for b in bounds:
            # int(...) the WHOLE expression, not just round()'s output: x0/x1 are
            # numpy.int64 (from np.where(...).min()/.max()), and numpy.int64 + int
            # stays numpy.int64 -- which json.dump can't serialize, crashing the
            # _vlm.json sidecar write below for every image that actually gets a
            # parsed unit.
            px_start = int(x0 + round(b["x_start"] * crop_w))
            px_end = int(x0 + round(b["x_end"] * crop_w))
            pad = max(1, int((px_end - px_start) * VLM_UNIT_PAD_FRAC))
            ux0, uy0, ux1, uy1 = to_src(max(x0, px_start - pad), y0,
                                        min(x1 + 1, px_end + pad), y1 + 1)
            unit_crop = src[uy0:uy1, ux0:ux1]
            if unit_crop.size == 0:
                continue
            unit_crop = fit_crop(unit_crop)
            cls_raw = self._vlm_generate(unit_crop, classify_prompt, VLM_CLASSIFY_MAX_NEW_TOKENS)
            result = self._parse_vlm_classification(cls_raw)
            if result is None:
                continue
            units.append({**b, **result, "px_start": px_start, "px_end": px_end})

        if not units:
            return building                              # VLM disabled/failed/unparseable -> keep CLIP verdict
        building["vlm_units"] = units

        # Fuse: the LARGEST unit's class becomes the building's headline
        # use_class (for backward compatibility with the single-label
        # render/summary), while the full split is kept in vlm_units for
        # anyone consuming the raw JSON. Take the VLM label only when it
        # agrees with CLIP, or disagrees with reasonably high confidence, so a
        # low-confidence VLM call never overrides a strong CLIP verdict.
        largest = max(units, key=lambda u: u["x_end"] - u["x_start"])
        agree = largest["use_class"] == building.get("use_class")
        if not agree and largest["confidence"] < 0.55:
            return building                              # low-confidence disagreement -> keep CLIP
        building["clip_use_class"] = building.get("use_class")
        building["clip_confidence"] = building.get("confidence")
        building["use_class"] = largest["use_class"]
        building["confidence"] = largest["confidence"]
        building["bucket"] = ("high" if largest["confidence"] >= 0.7
                              else "medium" if largest["confidence"] >= 0.45 else "low")
        building["vlm_corrected"] = not agree
        return building


# ------------------------------------------------------------------------------
# Reference-style rendering (output/01.png, output/02.png)
# ------------------------------------------------------------------------------
def _dashed_polygon(canvas, contour, color, thickness=3, dash=14, gap=9):
    pts = contour.reshape(-1, 2)
    n = len(pts)
    for i in range(n):
        p1, p2 = pts[i].astype(np.float64), pts[(i + 1) % n].astype(np.float64)
        seg_len = np.linalg.norm(p2 - p1)
        if seg_len < 1e-3:
            continue
        step = dash + gap
        for j in range(max(1, int(seg_len / step)) + 1):
            t0 = min(1.0, j * step / seg_len)
            t1 = min(1.0, t0 + dash / seg_len)
            a = tuple((p1 + (p2 - p1) * t0).astype(int))
            b = tuple((p1 + (p2 - p1) * t1).astype(int))
            cv2.line(canvas, a, b, color, thickness, cv2.LINE_AA)


def _outer_contour(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    return cv2.approxPolyDP(biggest, 0.003 * cv2.arcLength(biggest, True), True)


def render(image_rgb, buildings, gf_fraction):
    h, w = image_rgb.shape[:2]
    label_strip_h = 230

    # translucent per-instance building overlay
    overlay = image_rgb.copy()
    for i, b in enumerate(buildings):
        overlay[b["mask"]] = PALETTE[i % len(PALETTE)]
    blended = cv2.addWeighted(overlay, 0.40, image_rgb, 0.60, 0)

    canvas = np.zeros((h + label_strip_h, w, 3), dtype=np.uint8)
    canvas[:h] = blended
    canvas[h:] = (18, 24, 33)

    for i, b in enumerate(buildings):
        color = PALETTE[i % len(PALETTE)]
        bc = _outer_contour(b["mask"])
        if bc is not None:
            cv2.polylines(canvas[:h], [bc], True, color, 2, cv2.LINE_AA)            # instance outline
            _dashed_polygon(canvas[:h], bc, BUILDING_BOUNDARY_COLOR, 2, dash=10, gap=8)  # white boundary
        gc = _outer_contour(b["gf_mask"])
        if gc is not None:
            _dashed_polygon(canvas[:h], gc, GROUND_FLOOR_COLOR, thickness=3)         # yellow ground floor
        # VLM unit separators: thin red dashed vertical lines at each internal
        # unit boundary (skip the outer two edges, which the yellow band already marks).
        units = b.get("vlm_units") or []
        if len(units) > 1:
            ys_gf, _ = np.where(b["gf_mask"])
            if len(ys_gf):
                y_top, y_bot = int(ys_gf.min()), int(ys_gf.max())
                for u in units[:-1]:
                    xline = u["px_end"]
                    for yy in range(y_top, y_bot, 12):
                        cv2.line(canvas[:h], (xline, yy), (xline, min(yy + 7, y_bot)),
                                 VLM_UNIT_COLOR, 2, cv2.LINE_AA)

    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil, "RGBA")
    f_tag, f_label, f_small, f_title = _font(26), _font(22), _font(17), _font(20)

    # B# tags on top of each building
    for i, b in enumerate(buildings):
        color = PALETTE[i % len(PALETTE)]
        ys, xs = np.where(b["mask"])
        cx, top_y = int(xs.mean()), max(int(ys.min()) - 34, 4)
        tag = f"B{i + 1}"
        tw = draw.textlength(tag, font=f_tag)
        draw.rectangle([cx - tw / 2 - 8, top_y, cx + tw / 2 + 8, top_y + 32], fill=color + (235,))
        draw.text((cx - tw / 2, top_y + 3), tag, font=f_tag, fill=(255, 255, 255, 255))

    # bottom label boxes, left-to-right, non-overlapping
    order = sorted(range(len(buildings)), key=lambda i: np.where(buildings[i]["mask"])[1].min())
    n = max(len(order), 1)
    margin = 12
    box_w = min(300, max(150, (w - margin * (n + 1)) / n))
    box_h = 210
    centers = [int(np.where(buildings[i]["mask"])[1].mean()) for i in order]
    x0s = [c - box_w / 2 for c in centers]
    for k in range(1, n):
        x0s[k] = max(x0s[k], x0s[k - 1] + box_w + margin)
    overflow = (x0s[-1] + box_w) - (w - margin) if n else 0
    if overflow > 0:
        x0s[-1] -= overflow
        for k in range(n - 2, -1, -1):
            x0s[k] = min(x0s[k], x0s[k + 1] - box_w - margin)
    x0s = [max(x0, margin) for x0 in x0s]

    for k, i in enumerate(order):
        b = buildings[i]
        color = PALETTE[i % len(PALETTE)]
        x0 = x0s[k]; x1 = x0 + box_w; y0 = h + 10; y1 = y0 + box_h
        draw.rectangle([x0, y0, x1, y1], fill=color + (235,), outline=(255, 255, 255, 180), width=1)
        pad = 10
        draw.text((x0 + pad, y0 + 8), f"B{i + 1}", font=f_label, fill=(255, 255, 255, 255))
        n_units = len(b.get("vlm_units") or [])
        gf_label = "Ground Floor Use:" if n_units <= 1 else f"Ground Floor Use ({n_units} units, VLM):"
        gf_label = _truncate_text(gf_label, f_small, box_w - 2 * pad, draw)
        draw.text((x0 + pad, y0 + 40), gf_label, font=f_small, fill=(255, 255, 255, 255))
        ty = y0 + 62
        for line in _wrap_text(b["use_class"].title(), f_label, box_w - 2 * pad, draw)[:2]:
            draw.text((x0 + pad, ty), line, font=f_label, fill=(255, 255, 255, 255))
            ty += 24
        if b.get("vlm_corrected"):
            corr_line = _truncate_text(f'CLIP said: {b.get("clip_use_class", "")}',
                                       f_small, box_w - 2 * pad, draw)
            draw.text((x0 + pad, ty), corr_line, font=f_small, fill=(255, 160, 160, 255))
            ty += 20
        sign_text = b.get("sign_text", "")
        if sign_text:
            sign_line = _truncate_text(f'Sign: "{sign_text}"', f_small, box_w - 2 * pad, draw)
            draw.text((x0 + pad, y1 - 52), sign_line, font=f_small, fill=(255, 235, 160, 255))
        draw.text((x0 + pad, y1 - 26),
                  f"Confidence: {b['bucket'].title()} ({b['confidence']:.0%})",
                  font=f_small, fill=(235, 235, 235, 255))

    # legend (top-left): the two boundary types, matching output/02.png
    legend_items = [
        (BUILDING_BOUNDARY_COLOR, "Building Boundary (Segmented)"),
        (GROUND_FLOOR_COLOR, "Ground Floor Region (Evaluated)"),
    ]
    if any(len(b.get("vlm_units") or []) > 1 for b in buildings):
        legend_items.append((VLM_UNIT_COLOR, "Shop Unit Split (VLM)"))
    legend_h = 40 + 28 * len(legend_items)
    draw.rectangle([10, 10, 340, 10 + legend_h], fill=(15, 20, 28, 210))
    draw.text((20, 16), "BUILDING SEGMENTATION", font=f_title, fill=(255, 255, 255, 255))
    for j, (col, name) in enumerate(legend_items):
        yy = 48 + j * 28
        draw.line([(20, yy + 8), (30, yy + 8)], fill=col + (255,), width=3)   # dashed swatch
        draw.line([(36, yy + 8), (46, yy + 8)], fill=col + (255,), width=3)
        draw.text((54, yy), name, font=f_small, fill=(230, 230, 230, 255))

    # summary (top-right)
    n_comm = sum(1 for b in buildings if b["use_class"] in COMMERCIAL_USE_CLASSES)
    n_feat = sum(1 for b in buildings if b.get("gf_method") == "feature-based")
    n_arcade = sum(1 for b in buildings if b.get("single_storey"))
    n_ocr = sum(1 for b in buildings if b.get("sign_text"))
    n_vlm_corrected = sum(1 for b in buildings if b.get("vlm_corrected"))
    n_multi_unit = sum(1 for b in buildings if len(b.get("vlm_units") or []) > 1)
    lines = ["SUMMARY", f"Buildings segmented: {len(buildings)}",
             f"Commercial ground floors: {n_comm}",
             f"Residential / other: {len(buildings) - n_comm}",
             f"Buildings with sign text (OCR): {n_ocr}",
             f"GF: {n_feat} feature-based, {n_arcade} arcade,",
             f"    {len(buildings) - n_feat - n_arcade} fraction-fallback"]
    if n_vlm_corrected or n_multi_unit:
        lines.append(f"VLM: {n_vlm_corrected} reclassified, {n_multi_unit} multi-unit")
    sw = 330; sh = 34 + 22 * len(lines)
    draw.rectangle([w - sw - 10, 10, w - 10, 10 + sh], fill=(15, 20, 28, 210))
    for i, line in enumerate(lines):
        draw.text((w - sw + 2, 16 + i * 22), line,
                  font=f_title if i == 0 else f_small, fill=(255, 255, 255, 255))

    return np.array(pil.convert("RGB"))


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


# ------------------------------------------------------------------------------
def process_image(seg, image_path, out_path, prompt, box_threshold, text_threshold,
                  gf_fraction, max_side, use_vlm):
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        log(f"  !! could not read image: {image_path}")
        return
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    orig_rgb = image_rgb                                   # keep full-res copy for OCR (see read_sign_text)
    h, w = image_rgb.shape[:2]
    scale = 1.0
    if max_side and max(h, w) > max_side:
        scale = max_side / max(h, w)
        image_rgb = cv2.resize(image_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    boxes = seg.detect(image_rgb, prompt, box_threshold, text_threshold)
    if not boxes:
        log(f"  {image_path.name}: nothing detected (try lowering --box-threshold)")
        return
    feature_boxes = seg.detect(image_rgb, FEATURE_PROMPT, FEATURE_BOX_THRESHOLD, FEATURE_TEXT_THRESHOLD)
    buildings = seg.segment(image_rgb, boxes, gf_fraction, feature_boxes)
    for b in buildings:
        b["use_class"], b["confidence"], b["bucket"] = seg.classify_use(image_rgb, b["gf_mask"])
        b["sign_text"] = seg.read_sign_text(image_rgb, b["gf_mask"], full_res_image=orig_rgb, scale=scale)
        if use_vlm:
            seg.vlm_correct_building(image_rgb, b, full_res_image=orig_rgb, scale=scale)

    annotated = render(image_rgb, buildings, gf_fraction)
    cv2.imwrite(str(out_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    combined = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    for b in buildings:
        combined[b["gf_mask"]] = 255
    cv2.imwrite(str(out_path.with_name(out_path.stem + "_mask.png")), combined)

    # sidecar JSON with the full per-unit VLM breakdown (kept separate from the
    # rendered PNG so the raw structured data survives even when the render
    # only shows the headline label per building).
    if use_vlm:
        sidecar = [{
            "building": i + 1,
            "clip_use_class": b.get("clip_use_class", b["use_class"]),
            "clip_confidence": b.get("clip_confidence", b["confidence"]),
            "final_use_class": b["use_class"],
            "final_confidence": b["confidence"],
            "vlm_corrected": b.get("vlm_corrected", False),
            "sign_text": b.get("sign_text", ""),
            "vlm_units": b.get("vlm_units", []),
        } for i, b in enumerate(buildings)]
        with open(out_path.with_name(out_path.stem + "_vlm.json"), "w") as f:
            json.dump(sidecar, f, indent=2)

    n_feat = sum(1 for b in buildings if b.get("gf_method") == "feature-based")
    n_arcade = sum(1 for b in buildings if b.get("single_storey"))
    n_vlm_corrected = sum(1 for b in buildings if b.get("vlm_corrected"))
    n_multi_unit = sum(1 for b in buildings if len(b.get("vlm_units") or []) > 1)
    extra = f", VLM: {n_vlm_corrected} reclassified/{n_multi_unit} multi-unit" if use_vlm else ""
    log(f"  {image_path.name}: {len(buildings)} building(s), "
        f"ground floor: {n_feat} feature-based, {n_arcade} arcade-whole, "
        f"{len(buildings) - n_feat - n_arcade} fraction-fallback{extra} -> {out_path.name}")


def _collect_images(path):
    p = Path(path)
    if p.is_dir():
        return sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    return [p]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?",
                    default=str(BASE_DIR / "data" / "panos" / "00000-pano.jpg"),
                    help="input image, OR a folder to batch-process every image in it")
    ap.add_argument("--out", default=None,
                    help="output path (single image) or folder (batch); default results/ground_floor_sam_gdino/")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, help="Grounding DINO text prompt")
    ap.add_argument("--box-threshold", type=float, default=BOX_THRESHOLD)
    ap.add_argument("--text-threshold", type=float, default=TEXT_THRESHOLD)
    ap.add_argument("--gf-fraction", type=float, default=GF_FRACTION,
                    help="fraction of each building's height kept as ground floor (0-1)")
    ap.add_argument("--max-side", type=int, default=MAX_IMAGE_SIDE,
                    help="downscale longest side to this before inference (0 = keep original)")
    ap.add_argument("--use-vlm", action="store_true",
                    help="enable step 7: VLM ground-floor unit splitting + reclassification")
    ap.add_argument("--vlm-model", default=VLM_MODEL_ID,
                    help="HF model id for the VLM correction step (default: %(default)s)")
    ap.add_argument("--no-vlm-4bit", dest="vlm_4bit", action="store_false", default=VLM_4BIT,
                    help="disable 4-bit (bitsandbytes NF4) quantization for the VLM -- needs >=16GB VRAM")
    args = ap.parse_args()

    images = _collect_images(args.image)
    if not images:
        log(f"No images found at {args.image}")
        return
    out_dir = Path(args.out) if (args.out and (len(images) > 1 or Path(args.out).suffix == "")) else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log("=" * 78)
    log("BUILDING SEGMENTATION & GROUND-FLOOR USE CLASSIFICATION")
    log("Grounding DINO (SwinB) + SAM (ViT-H) + CLIP + PaddleOCR PP-OCRv6" +
        (" + VLM" if args.use_vlm else "") + "  |  " +
        f"{len(images)} image(s), device={device}")
    log("=" * 78)
    seg = GroundFloorSegmenter(device, use_vlm=args.use_vlm,
                               vlm_model_id=args.vlm_model, vlm_4bit=args.vlm_4bit)

    max_side = args.max_side if args.max_side else None
    for i, image_path in enumerate(images, 1):
        if args.out and len(images) == 1 and not out_dir:
            out_path = Path(args.out)
        else:
            out_path = (out_dir or RESULTS_DIR) / (image_path.stem + "_ground_floor.png")
        log(f"[{i}/{len(images)}] {image_path.name}")
        try:
            process_image(seg, image_path, out_path, args.prompt, args.box_threshold,
                          args.text_threshold, args.gf_fraction, max_side, args.use_vlm)
        except Exception as e:                 # one bad image shouldn't kill the batch
            log(f"  !! failed on {image_path.name}: {e}")
    log(f"\nDone. Results in {out_dir or RESULTS_DIR}")


if __name__ == "__main__":
    main()