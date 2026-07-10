# Ground-Floor Segmentation & Use Classification

Detect buildings in a street-level (equirectangular / normal) photo, segment each
one, isolate its **ground floor**, classify the ground-floor **use**
(retail / cafe / pharmacy / residential …), and read any **sign/storefront text**
on it. Output is an annotated image plus a ground-floor mask.

Uses five models: **Grounding DINO (SwinB)** for detection, **SAM (ViT-H)** for
segmentation, **SegFormer** as a semantic guard, **CLIP** for use classification,
and **PaddleOCR PP-OCRv6** (multilingual, Greek + English) for sign text.

---

## Folder layout

```
ground_floor_pipeline/
├── ground_floor_sam_gdinov2.py   # the current pipeline (single file) -- adds PaddleOCR sign text
├── ground_floor_sam_gdinov2.5.py # optional next step -- adds a local VLM correction pass, see below
├── ground_floor_sam_gdino.py     # earlier version, kept for reference (no OCR step)
├── download_models.sh            # download the 2 checkpoints (bash / wget)
├── download_models.py            # download the 2 checkpoints (pure python)
├── requirements.txt
├── README.md
├── pretrained_model/           # the 2 .pth checkpoints go here
└── data/panos/                 # put your input images here (optional)
└── results/                    # created automatically; annotated outputs land here
```

---

## Setup

```bash
# 1. environment (Python 3.11 recommended)
conda create -n groundfloor python=3.11 -y
conda activate groundfloor

# 2. install torch matching your CUDA FIRST, e.g. CUDA 12.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# then the rest:
pip install -r requirements.txt

# 3. download the two checkpoints into pretrained_model/
bash download_models.sh          # or:  python download_models.py
```

Two checkpoints are downloaded (~3.5 GB total):

| File | Model | Size | Source |
|------|-------|------|--------|
| `sam_vit_h_4b8939.pth` | SAM ViT-H | 2.6 GB | Meta segment-anything |
| `groundingdino_swinb_cogcoor.pth` | Grounding DINO SwinB | 938 MB | IDEA-Research |

**CLIP**, **SegFormer**, Grounding DINO's **BERT** text encoder, and **PaddleOCR
PP-OCRv6**'s detection + recognition models are *not* in the zip — they download
themselves on first run (CLIP/SegFormer/BERT from Hugging Face into `./.cache`;
PaddleOCR's models via PaddleX into `~/.paddlex/official_models`).

---

## Usage

```bash
# single image
python ground_floor_sam_gdinov2.py path/to/street.jpg

# a whole folder (batch — models load once, then every image is processed)
python ground_floor_sam_gdinov2.py data/panos

# useful flags
python ground_floor_sam_gdinov2.py data/panos \
    --out results/my_run \        # output folder
    --box-threshold 0.30 \        # lower = detect more buildings (more recall)
    --gf-fraction 0.30 \          # bottom-fraction fallback height (0-1)
    --max-side 1536               # downscale long side before inference (0 = keep)
```

Each image produces two files in `results/ground_floor_sam_gdino/`:
`<name>_ground_floor.png` (annotated) and `<name>_ground_floor_mask.png` (binary
ground-floor mask).

---

## How it works (brief)

1. **Detection** — Grounding DINO (prompt `"building. house."`) runs on the whole
   image **plus overlapping square tiles** (so a 360° pano doesn't collapse into
   one giant box); degenerate/duplicate boxes are dropped.
2. **Segmentation** — each building box is a **SAM** box-prompt → facade mask,
   cleaned by three guards: clip-to-box + largest connected component, a
   **SegFormer** semantic guard (drop masks that are mostly ceiling / floor / sky),
   and an aspect guard (drop tall-thin arcade **pillars**).
3. **Ground floor** — per building, in priority order:
   - **Arcade check**: if the strip just above the facade is *ceiling* (covered
     passage, upper floors hidden) → the whole segment is the ground floor.
   - **Feature-based**: a *second* Grounding DINO pass detects shopfront features
     (`"shop window. storefront. door. roller shutter. signboard. awning."`); the
     ground floor is the band where those features are, scanned up from the street.
   - **Fallback**: if no features land on the building, take the bottom `GF_FRACTION`.
4. **Classification** — CLIP labels the ground-floor crop (retail / cafe /
   pharmacy / residential …) with a confidence bucket.
5. **Sign text** — **PaddleOCR PP-OCRv6** (detection + a single multilingual
   recognizer covering Greek *and* English, no per-language switching needed)
   reads any sign/storefront text from the same ground-floor mask. The crop is
   re-extracted from the *original full-resolution* image, not the downscaled
   working copy — signage is a small fraction of an 8192×4096 panorama and is
   already illegible after the `--max-side` downscale. A small blocklist drops
   the Street View "© 2024 Google" watermark text if OCR happens to pick it up.
6. **Render** — the annotated image is drawn: building overlay, white-dashed
   **building boundary**, yellow-dashed **ground-floor region**, B# tags,
   per-building use + sign text + confidence, legend, and summary.

---

## Optional next step: VLM-corrected ground floors (v2.5)

`ground_floor_sam_gdinov2.5.py` is the same pipeline plus an optional 7th step:
a local vision-language model (**Qwen2.5-VL-3B-Instruct**, 4-bit quantized by
default) re-inspects each building's ground-floor crop, decides whether it's
really *one* shop or several adjoining units, and classifies each unit using
the ground-floor crop, CLIP's baseline guess, and the PaddleOCR sign text as
context. It also switches to a different 12-class use taxonomy (Food &
Beverage, Retail/Utility, Health and wellness establishments, Beauty and
fashion boutiques, Professional services, Parking lot, Accommodation,
Workshops, Open Public Space, Closed Public Space, Other, Vacant) instead of
v2's 14-class list.

```bash
pip install -r requirements.txt   # now also installs accelerate + bitsandbytes

python ground_floor_sam_gdinov2.5.py data/panos/street.jpg --use-vlm

# useful flags
python ground_floor_sam_gdinov2.5.py data/panos/street.jpg --use-vlm \
    --vlm-model Qwen/Qwen2.5-VL-3B-Instruct \  # swap in a different Qwen2.5-VL checkpoint
    --no-vlm-4bit                              # disable 4-bit quantization -- needs >=16GB VRAM
```

It runs in two VLM calls per building (a spatial split, then one focused
classification call per unit) rather than one compound call, which was found
to be more reliable in testing. With `--use-vlm`, each run also writes a
`<name>_ground_floor_vlm.json` sidecar with the full per-unit breakdown
(bounding fractions, use class, confidence, evidence) alongside the usual PNG
outputs.

**Sizing**: 4-bit NF4 quantization uses roughly 2.5–3.5 GB of VRAM on top of
the ~4–5 GB the rest of the stack (GDINO + SAM + SegFormer + CLIP + PaddleOCR)
already occupies — tested working on an 8 GB card. `--use-vlm` is entirely
optional and additive: if the VLM (or `bitsandbytes`/`accelerate`) can't be
loaded, it's silently disabled and every building keeps its CLIP-only verdict.

**Honest caveat from testing**: a 3B, 4-bit-quantized VLM is not fully
reliable at this task. It sometimes classifies correctly using the
provided OCR text, sometimes ignores clear OCR signal in favor of a generic
visual guess, and its "evidence" field is occasionally unrelated to the actual
decision. Treat `--use-vlm` output as a second opinion worth spot-checking,
not a ground truth upgrade over the base CLIP classification.

---

## Notes & limitations

- **transformers 5.x**: `groundingdino-py` was written for transformers 4.x; the
  script includes an in-memory compatibility shim (`_patch_groundingdino_for_transformers5`)
  so no downgrade is needed. Nothing on disk is modified.
- **No CUDA extension needed**: `groundingdino-py` runs its deformable-attention in
  pure PyTorch, so a missing `groundingdino._C` build is harmless.
- **Scale ambiguity**: from a single 2D image the exact storey line of a *multi-storey*
  building can't be read reliably (a 1-storey shop close-up looks like a 5-storey block
  far away). The feature/arcade cues handle most shopfronts; multi-storey buildings can
  occasionally over- or under-cover. Metric depth / LiDAR would remove this ambiguity.
- **OCR is optional at runtime**: if `paddleocr` isn't installed (or the model download
  fails), PaddleOCR is silently skipped and every building just gets an empty sign text —
  everything else in the pipeline still runs normally.
- **GPU recommended** (ViT-H is heavy on CPU). Model load ≈ 1–2 min once; then a few
  seconds per image.
