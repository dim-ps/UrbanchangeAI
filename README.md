# GroundedSAM2 Urban Change

A research prototype for building instance segmentation and experimental ground-floor detection from panoramic street-level imagery.

---

## Workflow

```
Panoramic image
        │
        ▼
Grounding DINO
        │
        ▼
SAM2
        │
        ▼
Building Instance Segmentation
        │
        ▼
Experimental Ground-Floor Detection
```

---

## Current Status

| Module | Status |
|---------|--------|
| Building Detection | ✅ |
| Building Instance Segmentation | ✅ |
| Ground-floor Detection | 🟡 Experimental |
| Ground-floor Use Classification | ⏳ Planned |

---

## Repository Structure

```
GroundedSAM2-UrbanChange/

├── scripts/
│   └── ground.py
│
├── input/
│   └── sample_images/
│
├── outputs/
│
├── checkpoints/
│
├── docs/
│
├── README.md
├── requirements.txt
├── environment.yml
└── .gitignore
```

---

## Installation

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate grounded_sam
```

Install the official SAM2 repository:

```bash
pip install -e .
```

---

## Checkpoints

Download the SAM2 checkpoint:

```
sam2.1_hiera_large.pt
```
It can be downloaded from:

```
https://lgrl-net.aegean.gr:5051/d/s/18wen89ZJqAu19askwmpYj6G1FCdtbBc/kgB2qOdzYG2Om6sVVSuYPrfaaBQsC7DI-5b_gUwoKVA0
```

and place it inside:

```
checkpoints/
```

---

## Input Images

Place panoramic images in:

```
input/sample_images/
```
They can be downloaded from:

```
https://lgrl-net.aegean.gr:5051/d/s/18weT1IFgHvCqaik8McmHVtIVshuyShA/opp758Qp2D1MTb5tPbm4CCELKnsTFn06-L7BgpUUKVA0
```

Supported formats:

- jpg
- jpeg
- png

---

## Run

```bash
python scripts/ground.py
```

---

## Outputs

The generated outputs include:

- Building detections
- Ground-floor detections
- Overlay images
- Semantic masks
- Instance masks
- CSV files
- JSON annotations

All outputs are written to:

```
outputs/
```

---

## License

Research prototype developed for urban change analysis.
