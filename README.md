# Система проверки пользовательских фотографий в туристических сервисах с применением методов глубокого машинного обучения

## Оглавление  

[1. Описание проекта](.README.md#Описание-проекта)  
[2. Формализация задачи ](.README.md#Какой-кейс-решаем)  
[3. Краткая информация о данных](.README.md#Краткая-информация-о-данных)  
[4. Этапы работы над проектом](.README.md#Результат) 
[4. Результаты](.README.md#Результат)    
[5. Структура репозитория](.README.md#Результат) 
[6. Установка и использование](.README.md#Выводы) 

### Описание проекта  

Данный проект выполнен в рамках магистерской диссертации НИЯУ МИФИ и представляет собой реализацию проекта проверки пользовательских фотографий в туристических сервисах с применением дообучения предобученной мультимодальной модели SIDA-7B. 


## Формализация задачи

В связи с взрывным развитием генеративных моделей возникает потребность в детекции синтетических и частично - синтетических изображений в контексте туристического домена

## Краткая информация о данных 

Датасет для дообучения модели формировался из личного архива автора.
Синтетические и частично - синтетические изображения получены посредством API - доступа ([ruGPT.io](https://rugpt.io)).Список использованных моделей приведен в таблице.

### Набор данных полностью синтетических изображений

| Модель | Компания | Архитектура | Количество изображений |
|-------|-----------|-------------|--------|
| Flux 2 Pro | Black Forest Labs | Rectified-flow DiT, 32B | 152 |
| Seedream 4.5 | ByteDance | Scalable DiT | 123 |
| Z-Image | Alibaba Tongyi MAI | S³-DiT, 6B  | 122 |
| Imagen 4 | Google DeepMind | Cascaded latent diffusion | 31 |

### Частично - синтетические  изображения с заменой фона

| Модель | Компания | Метод | Количество изображений |
|-------|-----------|--------|--------|
| Nano Banana | Google DeepMind | img2img (Gemini 2.5 Flash Image) | 241 |
| Flux 2 Pro edit | Black Forest Labs | img2img, 2K resolution | 192 |

### Итоговый набор данных

| Класс | Характеристика | Изображение|
|-------|-------------|-------|
| **Подлинные изображения** | Личный архив автора | 438 |
| **полностью синтетические изображения** | Синтетические фотографии туристических сцен | 428 |
| **частично-синтетические изображения** | Реальный человек и замена фона с достопримечательностью | 433 |
| **Всего** | | **1 299** |

---
## Этапы работы над проектом

* 1.Исследовательский анализ
* 2.Сбор и генерация изображений для формирования датасета
* 3.Тестирование базовой модели SIDA - 7B
* 4.Дообучение модели на целевом домене
* 5.Реализация программного интерфейса Streamlit
* 6.Анализ результатов

## Результаты

| Метрика | Базовая SIDA-7B | Дообученная SIDA - 7B |
|--------|-----------------|------------------------|
| Overall Accuracy | 44.0% | **94.67%** |
| Macro F1 | 0.363 | **0.946** |
| REAL accuracy | 91% | 94% |
| FAKE accuracy | 5% | **100%** |
| TAMPERED accuracy | 5% | 90% |

Лучшая эпоха: **11** · Ранняя остановка на 18 эпохе · Seed: 42

---

## Структура репозитория

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
│   ├── cls_head_new.pth         # Fine-tuned classification head weights (33 MB)
│   ├── sam_vit_h_4b8939.pth     # SAM ViT-H checkpoint (30 MB)
│   └── SIDA-7B/                 #  Download separately (see Setup)
│
├── model/                       # SIDA model architecture (original authors)
├── utils/                       # Utility functions (original authors)
│
├── environment.yml              # Conda environment (Python 3.10, CUDA 12.6)
├── requirements.txt             # pip dependencies
└── .gitignore
```

> **Streamlit demo** — будет добавлено

---

## Установка и использование

### 1. Скопируйте репозиторий и создаюте виртуальную среду

```bash
git clone https://github.com/YOUR_USERNAME/sida-tourist-deepfake.git
cd sida-tourist-deepfake

conda env create -f environment.yml
conda activate sida_modern
```

### 2. Загрузка базовой модели

```bash
pip install huggingface_hub
huggingface-cli download Peterande/SIDA-7B --local-dir ck/SIDA-7B
```

### 3. Требования по железу

| Component | Minimum | Used in this work |
|-----------|---------|-------------------|
| GPU | 24 GB VRAM | NVIDIA RTX 5090 (32 GB) |
| RAM | 32 GB | 64 GB |
| Storage | 20 GB | SSD |

---

## Создание датасета

```bash
export RUGPT_API_KEY="your_key_here"
```

### Полностью синтетические изображения

```bash
cd dataset_generation/full_synthetic

python gen_flux2pro.py    # Flux 2 Pro   → dataset/full_synthetic/flux2pro/
python gen_seedream45.py  # Seedream 4.5 → dataset/full_synthetic/seedream45/
python gen_zimage.py      # Z-Image      → dataset/full_synthetic/zimage/
python gen_imagen4.py     # Imagen 4     → dataset/full_synthetic/imagen4/
```

> Each script generates a test image first and asks for confirmation
> before running the full batch.

### Частично - синтетические изображения

```bash
cd dataset_generation/tempered

# Source: your real photographs must be in dataset/real/
python gen_nano_banana.py    # → dataset/tempered/nano_banana/
python gen_flux2pro_edit.py  # → dataset/tempered/flux2pro_edit/
```

### Разделение на тренировочную и тестовую выборки

```bash
cd dataset_generation
python split_dataset.py
# Output: photo/train/{real,full_synt,tempered}/
#         photo/test/ {real,full_synt,tempered}/
```

---

## Дообучение

```bash
cd detection
python finetune.py
```

**Этап 1** (run once, ~20 min): extracts and caches [CLS] feature vectors
from all images using the frozen SIDA-7B backbone.

**Этап 2** (seconds per epoch): trains the classification head on cached features.

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

### Архитектура

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



## Инференс

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



## Цитирование

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
