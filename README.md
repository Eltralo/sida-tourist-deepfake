# SIDA-7B Fine-Tuning for Tourist Deepfake Detection



Detection and localisation of synthetic tourist photographs in Russian social media
using a domain-adapted **SIDA-7B** multimodal model.

---

## Problem statement

Tourist photographs on Russian social media platforms (ВКонтакте, Telegram)
are increasingly fabricated using generative AI:

| Class | Description | Count |
|-------|-------------|-------|
| **real** | Authentic personal archive photographs (2016–2026) | 438 |
| **full_synt** | Fully AI-generated tourist scenes (4 models) | 428 |
| **tempered** | Real person + AI-generated Russian landmark background | 433 |
| **Total** | | **1 299** |

---

## Results

| Metric | Baseline SIDA-7B | Fine-tuned (this work) |
|--------|-----------------|------------------------|
| Overall Accuracy | 44.0% | **94.67%** |
| Macro F1 | 0.363 | **0.946** |
| REAL accuracy | 91% | 94% |
| FAKE accuracy | 5% | **100%** |
| TAMPERED accuracy | 5% | 90% |

Best epoch: **11** · Early stopping at epoch 18 · Seed: 42

---

## Repository structure

```
sida-tourist-deepfake/
│
├── dataset_generation/          # Dataset construction pipeline
│   ├── _api_client.py           # Shared ruGPT.io API helpers
│   ├── split_dataset.py         # Train/test split (seed=42, 1000/300)
│   ├── full_synthetic/
│   │   ├── gen_flux2pro.py      # Flux 2 Pro  — 152 images
│   │   ├── gen_seedream45.py    # Seedream 4.5 — 123 images
│   │   ├── gen_zimage.py        # Z-Image      — 122 images
│   │   └── gen_imagen4.py       # Imagen 4     —  31 images
│   └── tempered/
│       ├── gen_nano_banana.py   # Nano Banana (background replacement) — 241 images
│       └── gen_flux2pro_edit.py # Flux 2 Pro edit (background replacement) — 192 images
│
├── detection/
│   ├── finetune.py              # Two-phase fine-tuning (Phase 1: features, Phase 2: head)
│   ├── inference.py             # Inference with SAM mask generation
│   ├── baseline_eval.py         # Baseline evaluation (unmodified SIDA-7B)
│   └── eval_metrics.py          # Metrics: accuracy, Macro F1, confusion matrix
│
├── ck/
│   ├── cls_head_new.pth         # ✅ Fine-tuned classification head weights (33 MB)
│   ├── sam_vit_h_4b8939.pth     # SAM ViT-H checkpoint (30 MB)
│   └── SIDA-7B/                 # ⬇️  Download separately (see Setup)
│
├── model/                       # SIDA model architecture (original authors)
├── utils/                       # Utility functions (original authors)
│
├── environment.yml              # Conda environment (Python 3.10, CUDA 12.6)
├── requirements.txt             # pip dependencies
└── .gitignore
```

> **Streamlit demo** — planned, not yet implemented.
> Will be added in `streamlit_app/` in a future release.

---

## Setup

### 1. Clone and create environment

```bash
git clone https://github.com/YOUR_USERNAME/sida-tourist-deepfake.git
cd sida-tourist-deepfake

conda env create -f environment.yml
conda activate sida_modern
```

### 2. Download base model

```bash
pip install huggingface_hub
huggingface-cli download Peterande/SIDA-7B --local-dir ck/SIDA-7B
```

### 3. Hardware requirements

| Component | Minimum | Used in this work |
|-----------|---------|-------------------|
| GPU | 24 GB VRAM | NVIDIA RTX 5090 (32 GB) |
| RAM | 32 GB | 64 GB |
| Storage | 20 GB | SSD |

---

## Reproducing the dataset

All generation scripts require a `RUGPT_API_KEY` environment variable
(your ruGPT.io API key). **Never hard-code API keys in source files.**

```bash
export RUGPT_API_KEY="your_key_here"
```

### Fully synthetic images

```bash
cd dataset_generation/full_synthetic

python gen_flux2pro.py    # Flux 2 Pro   → dataset/full_synthetic/flux2pro/
python gen_seedream45.py  # Seedream 4.5 → dataset/full_synthetic/seedream45/
python gen_zimage.py      # Z-Image      → dataset/full_synthetic/zimage/
python gen_imagen4.py     # Imagen 4     → dataset/full_synthetic/imagen4/
```

> Each script generates a test image first and asks for confirmation
> before running the full batch.

### Tempered images (background replacement)

```bash
cd dataset_generation/tempered

# Source: your real photographs must be in dataset/real/
python gen_nano_banana.py    # → dataset/tempered/nano_banana/
python gen_flux2pro_edit.py  # → dataset/tempered/flux2pro_edit/
```

### Train/test split

```bash
cd dataset_generation
python split_dataset.py
# Output: photo/train/{real,full_synt,tempered}/
#         photo/test/ {real,full_synt,tempered}/
```

---

## Fine-tuning

```bash
cd detection
python finetune.py
```

**Phase 1** (run once, ~20 min): extracts and caches [CLS] feature vectors
from all images using the frozen SIDA-7B backbone.

**Phase 2** (seconds per epoch): trains the classification head on cached features.

```bash
# Force re-extraction (skip cache):
python finetune.py --no-cache

# Custom paths:
python finetune.py \
    --model-dir ../ck/SIDA-7B \
    --train-dir ../photo/train \
    --test-dir  ../photo/test \
    --save-head ../ck/cls_head_new.pth
```

### Architecture

```
SIDA-7B backbone (frozen, 7.35B parameters — 99.886%)
    ↓
[CLS] hidden-state vector  h_cls ∈ ℝ⁴⁰⁹⁶
    ↓
cls_head (trainable, 8.4M parameters — 0.114%)
    Linear(4096 → 2048) → ReLU → Dropout(0.1) → Linear(2048 → 3)
    ↓
REAL / FAKE / TAMPERED
```

Key change vs original SIDA code: **Dropout increased from p=0.0 to p=0.1**
to regularise training on the small (1 000 image) dataset.

---

## Inference

```bash
cd detection

# Full inference with SAM mask generation for TAMPERED predictions:
python inference.py

# Baseline (unmodified SIDA-7B, no fine-tuning):
python baseline_eval.py

# Metrics only (from existing results.json):
python eval_metrics.py
```

Results are saved to `results_final/`:
- `report.txt` — human-readable report
- `results.json` — per-image predictions
- `masks/` — binary masks for TAMPERED predictions
- `overlays/` — original images with mask overlay

---

## Models used

### Fully synthetic dataset

| Model | Developer | Architecture | Images |
|-------|-----------|-------------|--------|
| Flux 2 Pro | Black Forest Labs | Rectified-flow DiT, 32B | 152 |
| Seedream 4.5 | ByteDance | Scalable DiT | 123 |
| Z-Image | Alibaba Tongyi MAI | S³-DiT, 6B ([arXiv:2511.22699](https://arxiv.org/abs/2511.22699)) | 122 |
| Imagen 4 | Google DeepMind | Cascaded latent diffusion | 31 |

### Tempered dataset (background replacement)

| Model | Developer | Method | Images |
|-------|-----------|--------|--------|
| Nano Banana | Google DeepMind | img2img (Gemini 2.5 Flash Image) | 241 |
| Flux 2 Pro edit | Black Forest Labs | img2img, 2K resolution | 192 |

All models accessed via [ruGPT.io](https://rugpt.io) — a Russian API aggregator
providing ruble-denominated access to international generative models.

> **Note on Imagen 4**: ruGPT.io exposes Imagen 4 under the alias `"dall-e-3"`.
> All Imagen 4 outputs carry an invisible **SynthID** watermark
> ([arXiv:2510.09263](https://arxiv.org/abs/2510.09263)).

---

## Citation

If you use this work, please cite:

```bibtex
@mastersthesis{eltara2026tourist,
  title  = {Detection of Synthetic Tourist Images in Russian Social Media
             Using Domain-Adapted SIDA-7B},
  author = {Eltara},
  year   = {2026},
  school = {Your University}
}
```

Original SIDA paper:

```bibtex
@article{huang2024sida,
  title  = {SIDA: Social Media Image Deepfake Detection, Localization
             and Explanation with Large Multimodal Model},
  author = {Huang, Zhenglin and others},
  year   = {2024},
  eprint = {2412.04292},
  url    = {https://arxiv.org/abs/2412.04292}
}
```

---

## License

Detection code and fine-tuning scripts: **MIT License**

Base SIDA-7B model: see [original repository](https://github.com/Peterande/SIDA)

Dataset images: not included in this repository.
