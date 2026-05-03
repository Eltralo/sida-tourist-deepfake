#!/usr/bin/env python3
"""
streamlit_app/sida_inference.py
────────────────────────────────
Адаптер инференса для Streamlit-приложения.

Класс SIDAInference инкапсулирует загрузку дообученной модели SIDA-7B и
выполнение полного цикла предсказания для одного PIL-изображения. Модуль
повторяет 1-в-1 логику ``detection/inference.py``, но возвращает структуру
данных, удобную для отображения в UI (а не пишет файлы на диск).

Внутренняя последовательность обработки изображения:
    1. Извлечение вектора скрытого состояния [CLS] и классификация через
       cls_head → REAL (0) / FAKE (1) / TAMPERED (2).
    2. Для предсказаний класса TAMPERED — генерация бинарной маски
       сегментации через декодер SAM.

Возвращаемая структура (PredictionResult):
    label         — канонический класс ("real" / "fake" / "tampered")
    confidence    — уверенность в выбранном классе [0, 1]
    logits        — вероятности по всем трём классам
    mask          — бинарная маска (uint8 0/255) или None
    latency_s     — время инференса, секунды

Использование
─────
    from streamlit_app.sida_inference import SIDAInference

    loader = SIDAInference()                    # пути по умолчанию
    result = loader.predict(pil_image)
    print(result.label, result.confidence)
"""

import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

warnings.filterwarnings("ignore")

from transformers import AutoTokenizer, CLIPImageProcessor

from model.SIDA import SIDAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
)

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL_DIR = Path("../ck/SIDA-7B")
DEFAULT_HEAD_PATH = Path("../ck/cls_head_new.pth")

LABEL_NAMES   = ["real", "fake", "tampered"]
LABEL_REAL    = "real"
LABEL_FAKE    = "fake"
LABEL_TAMPERED = "tampered"

LABEL_RU = {
    "real":     "Подлинное",
    "fake":     "Полностью синтетическое",
    "tampered": "Частично синтетическое",
}

VERDICT_RU = {
    "real":     "Данное фото является подлинным",
    "fake":     "Данное фото является полностью синтетическим",
    "tampered": "Данное фото является частично синтетическим",
}

LABELS = LABEL_NAMES  # alias для совместимости с UI-кодом

SAM_IMG_SIZE = 1024
SAM_PIXEL_MEAN = torch.Tensor([123.675, 116.28,  103.53]).view(-1, 1, 1)
SAM_PIXEL_STD  = torch.Tensor([ 58.395,  57.12,   57.375]).view(-1, 1, 1)


def preprocess_sam(x: torch.Tensor, img_size: int = SAM_IMG_SIZE) -> torch.Tensor:
    x = (x - SAM_PIXEL_MEAN) / SAM_PIXEL_STD
    h, w = x.shape[-2:]
    return F.pad(x, (0, img_size - w, 0, img_size - h))


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    label: str
    confidence: float
    logits: dict = field(default_factory=dict)
    mask: Optional[np.ndarray] = None
    latency_s: float = 0.0
    raw_text: str = ""

    @property
    def label_ru(self) -> str:
        return LABEL_RU.get(self.label, self.label)

    @property
    def verdict_ru(self) -> str:
        return VERDICT_RU.get(self.label, self.label)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "label_ru": self.label_ru,
            "verdict_ru": self.verdict_ru,
            "confidence": float(self.confidence),
            "logits": {k: float(v) for k, v in self.logits.items()},
            "has_mask": self.mask is not None,
            "latency_s": float(self.latency_s),
        }


# ── Inference class ───────────────────────────────────────────────────────────

class SIDAInference:
    """Загружает SIDA-7B + дообученную голову и выполняет inference для UI."""

    PROMPT_TEXT = (
        "Please answer begin with [CLS] for classification, "
        "if the image is tampered, output mask the tampered region."
    )

    def __init__(
        self,
        weights_path: Path | str = DEFAULT_MODEL_DIR,
        cls_head_path: Path | str = DEFAULT_HEAD_PATH,
        project_root: Optional[Path | str] = None,
        precision: str = "bf16",
    ):
        self.weights_path  = Path(weights_path)
        self.cls_head_path = Path(cls_head_path)
        self.precision     = precision

        if project_root is not None:
            project_root = str(Path(project_root).resolve())
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA недоступна. SIDA-7B требует GPU c минимум 24 GB VRAM. "
                "Проверьте `nvidia-smi` и установку драйверов CUDA 12.6."
            )

        self._dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }.get(precision, torch.bfloat16)

        self._load_model_and_head()
        self._input_ids = self._build_input_ids()

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model_and_head(self):
        """Полностью повторяет load_model_and_head() из detection/inference.py."""
        print("[1/3] Loading SIDA-7B...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.weights_path), model_max_length=512,
            padding_side="right", use_fast=False,
        )
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.seg_idx = self.tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
        self.cls_idx = self.tokenizer("[CLS]", add_special_tokens=False).input_ids[0]

        self.model = SIDAForCausalLM.from_pretrained(
            str(self.weights_path),
            low_cpu_mem_usage=True,
            vision_tower="openai/clip-vit-large-patch14",
            seg_token_idx=self.seg_idx,
            cls_token_idx=self.cls_idx,
            torch_dtype=self._dtype,
        )
        self.model.config.eos_token_id = self.tokenizer.eos_token_id

        if self.precision == "bf16":
            self.model = self.model.bfloat16().cuda()
        elif self.precision == "fp16":
            self.model = self.model.half().cuda()
        else:
            self.model = self.model.float().cuda()

        vt = self.model.get_model().get_vision_tower()
        if not vt.is_loaded:
            vt.load_model()
        if self.precision == "bf16":
            vt.bfloat16().cuda()
        elif self.precision == "fp16":
            vt.half().cuda()
        else:
            vt.float().cuda()

        self.clip_proc = CLIPImageProcessor.from_pretrained(self.model.config.vision_tower)
        self.transform = ResizeLongestSide(SAM_IMG_SIZE)

        print(f"[2/3] Loading cls_head: {self.cls_head_path}")
        state = torch.load(str(self.cls_head_path), map_location="cpu")
        head = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(2048, 3),
        )
        head.load_state_dict(state)
        head = head.bfloat16().cuda() if self.precision == "bf16" else head.cuda()

        # Replace the model's classification head in-place
        self.model.get_model().cls_head[0] = head
        self.model.eval()
        print("    cls_head loaded  ✓")

    def _build_input_ids(self) -> torch.Tensor:
        """Совпадает с build_input_ids() из detection/inference.py."""
        conv = conversation_lib.conv_templates["llava_v1"].copy()
        conv.messages = []
        full = (
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            + "\n" + self.PROMPT_TEXT
        )
        conv.append_message(conv.roles[0], full)
        conv.append_message(conv.roles[1], "")
        return tokenizer_image_token(
            conv.get_prompt(), self.tokenizer, return_tensors="pt"
        ).unsqueeze(0).cuda()

    # ── Per-image inference ───────────────────────────────────────────────────

    def _classify(self, img_rgb: np.ndarray):
        """Аналог predict_one() из detection/inference.py."""
        img_clip = (
            self.clip_proc.preprocess(img_rgb, return_tensors="pt")["pixel_values"][0]
            .unsqueeze(0).cuda()
        )
        if self.precision == "bf16":
            img_clip = img_clip.bfloat16()
        elif self.precision == "fp16":
            img_clip = img_clip.half()

        with torch.no_grad():
            attn = torch.ones_like(self._input_ids, device="cuda")
            fw = LlavaLlamaForCausalLM.forward(
                self.model,
                images=img_clip,
                attention_mask=attn,
                input_ids=self._input_ids,
                output_hidden_states=True,
                use_cache=False,
            )
            hs = fw.hidden_states[-1]
            if hs.dim() == 2:
                hs = hs.unsqueeze(0)

            # [CLS] position with 255-token visual shift
            cls_mask = (self._input_ids[:, 1:] == self.cls_idx)
            cls_mask = torch.cat([
                torch.zeros((cls_mask.shape[0], 255), dtype=torch.bool, device="cuda"),
                cls_mask,
                torch.zeros((cls_mask.shape[0],   1), dtype=torch.bool, device="cuda"),
            ], dim=1)
            if cls_mask.size(1) > hs.size(1):
                cls_mask = cls_mask[:, : hs.size(1)]
            elif cls_mask.size(1) < hs.size(1):
                pad = torch.zeros(
                    (cls_mask.size(0), hs.size(1) - cls_mask.size(1)),
                    dtype=torch.bool, device="cuda",
                )
                cls_mask = torch.cat([cls_mask, pad], dim=1)

            logits   = self.model.get_model().cls_head[0](hs)
            selected = logits[cls_mask]
            cls_logits = selected[0] if selected.numel() > 0 else logits[0, -1, :]

            probs = torch.softmax(cls_logits.float(), dim=-1)
            pred  = int(probs.argmax().item())

        return pred, probs.cpu().numpy()

    def _generate_mask(self, img_bgr: np.ndarray, img_rgb: np.ndarray) -> Optional[np.ndarray]:
        """Аналог generate_mask() из detection/inference.py.

        Возвращает бинарную маску (uint8 0/255), приведённую к размеру
        исходного изображения.
        """
        img_clip = (
            self.clip_proc.preprocess(img_rgb, return_tensors="pt")["pixel_values"][0]
            .unsqueeze(0).cuda()
        )
        if self.precision == "bf16":
            img_clip = img_clip.bfloat16()
        elif self.precision == "fp16":
            img_clip = img_clip.half()

        img_sam_np = self.transform.apply_image(img_rgb)
        img_sam_t  = preprocess_sam(
            torch.from_numpy(img_sam_np).permute(2, 0, 1).contiguous()
        ).unsqueeze(0).cuda()
        if self.precision == "bf16":
            img_sam_t = img_sam_t.bfloat16()
        elif self.precision == "fp16":
            img_sam_t = img_sam_t.half()

        with torch.no_grad():
            result = self.model.evaluate(
                img_clip, img_sam_t, self._input_ids,
                [img_sam_np.shape[:2]], [list(img_rgb.shape[:2])],
                max_new_tokens=512, tokenizer=self.tokenizer,
            )
        pred_masks = result[1] if isinstance(result, tuple) and len(result) >= 2 else []
        if pred_masks and pred_masks[0] is not None:
            m = pred_masks[0].squeeze().cpu().numpy()
            mask_bin = (m > 0.5).astype(np.uint8)
            h, w = img_bgr.shape[:2]
            return cv2.resize(mask_bin * 255, (w, h),
                              interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        return None

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(self, image: Image.Image) -> PredictionResult:
        """Полный цикл инференса для одного PIL-изображения."""
        t0  = time.time()
        rgb = np.array(image.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        pred, probs = self._classify(rgb)
        label  = LABEL_NAMES[pred]
        logits = {LABEL_NAMES[i]: float(probs[i]) for i in range(len(LABEL_NAMES))}

        mask = None
        if label == LABEL_TAMPERED:
            try:
                mask = self._generate_mask(bgr, rgb)
            except Exception as exc:
                print(f"  [MASK ERROR] {exc}")

        return PredictionResult(
            label=label,
            confidence=float(probs[pred]),
            logits=logits,
            mask=mask,
            latency_s=time.time() - t0,
        )


# ── UI helpers ────────────────────────────────────────────────────────────────

def overlay_mask(image: Image.Image, mask: np.ndarray,
                 color=(255, 60, 60), alpha: float = 0.45) -> Image.Image:
    """Накладывает полупрозрачную маску на изображение для визуализации."""
    if mask is None:
        return image
    img = image.convert("RGBA")
    if mask.shape[:2] != img.size[::-1]:
        mask_img = Image.fromarray(mask).resize(img.size, Image.NEAREST)
        mask = np.array(mask_img)
    overlay = Image.new("RGBA", img.size, color + (0,))
    overlay_arr = np.array(overlay)
    overlay_arr[..., 3] = (mask > 0).astype(np.uint8) * int(255 * alpha)
    overlay = Image.fromarray(overlay_arr)
    return Image.alpha_composite(img, overlay).convert("RGB")


def mask_area_fraction(mask: Optional[np.ndarray]) -> float:
    """Доля пикселей маски от площади изображения, [0, 1]."""
    if mask is None:
        return 0.0
    return float((mask > 0).mean())


# ── Demo stub (для проверки UI без модели) ────────────────────────────────────

class DummyInference:
    """Возвращает детерминированный по хешу результат — для отладки UI."""

    def __init__(self, seed: int = 42):
        self._rng = np.random.default_rng(seed)

    def predict(self, image: Image.Image) -> PredictionResult:
        t0 = time.time()
        h  = hash(image.tobytes()) & 0xFFFFFFFF
        rng = np.random.default_rng(h)
        probs = rng.dirichlet([2, 1.5, 1.5])
        idx   = int(np.argmax(probs))
        label = LABEL_NAMES[idx]
        logits = dict(zip(LABEL_NAMES, probs.tolist()))

        mask = None
        if label == LABEL_TAMPERED:
            w, h_ = image.size
            mask = np.zeros((h_, w), dtype=np.uint8)
            cy = int(rng.uniform(h_ * 0.2, h_ * 0.8))
            cx = int(rng.uniform(w  * 0.2, w  * 0.8))
            r  = int(min(h_, w) * rng.uniform(0.08, 0.22))
            yy, xx = np.ogrid[:h_, :w]
            mask[(yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2] = 255

        return PredictionResult(
            label=label,
            confidence=float(probs[idx]),
            logits=logits,
            mask=mask,
            latency_s=time.time() - t0 + rng.uniform(0.3, 1.2),
        )
