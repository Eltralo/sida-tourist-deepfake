#!/usr/bin/env python3
"""
detection/baseline_eval.py
───────────────────────────
Evaluates the **unmodified** SIDA-7B model (no fine-tuning) on the
tourist deepfake dataset.  Parses text output to determine class.

This script produces the baseline numbers reported in the thesis:
    Overall Accuracy : 44.0%
    Macro F1         : 0.363

Usage
─────
    conda activate sida_modern
    cd detection/
    python baseline_eval.py

    python baseline_eval.py --test-dir ../photo/test --output-dir ../results
"""

import argparse
import json
import os
import time
import warnings
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

warnings.filterwarnings("ignore")

from transformers import AutoTokenizer, CLIPImageProcessor

from model.SIDA import SIDAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)

DEFAULT_MODEL_DIR  = Path("./ck/SIDA-7B")
DEFAULT_TEST_DIR   = Path("../photo/test")
DEFAULT_OUTPUT_DIR = Path("./results")

# Maps folder name → true class label (used for file organisation only)
CLASS_MAP = {"full_synt": "FAKE", "tempered": "TAMPERED", "real": "REAL"}
ALL_CLASSES = ["FAKE", "TAMPERED", "REAL"]
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SAM_PIXEL_MEAN = torch.Tensor([123.675, 116.28,  103.53]).view(-1, 1, 1)
SAM_PIXEL_STD  = torch.Tensor([ 58.395,  57.12,   57.375]).view(-1, 1, 1)


def preprocess_sam(x: torch.Tensor, img_size: int = 1024) -> torch.Tensor:
    x = (x - SAM_PIXEL_MEAN) / SAM_PIXEL_STD
    h, w = x.shape[-2:]
    return F.pad(x, (0, img_size - w, 0, img_size - h))


def classify_response(text: str) -> str:
    """Parse SIDA text output to determine predicted class."""
    t = text.lower()
    if any(k in t for k in ["fully synthetic", "fully generated",
                             "completely synthetic", "completely generated", "synthetic"]):
        return "FAKE"
    if any(k in t for k in ["tampered", "altered", "manipulated", "modified"]):
        return "TAMPERED"
    if any(k in t for k in ["real", "authentic", "genuine", "original"]):
        return "REAL"
    return "UNKNOWN"


def load_model(model_dir: Path):
    print(f"Loading {model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), model_max_length=512, padding_side="right", use_fast=False
    )
    tokenizer.pad_token = tokenizer.unk_token
    seg_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    cls_idx = tokenizer("[CLS]", add_special_tokens=False).input_ids[0]

    model = SIDAForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        vision_tower="openai/clip-vit-large-patch14",
        seg_token_idx=seg_idx,
        cls_token_idx=cls_idx,
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vt = model.get_model().get_vision_tower()
    if not vt.is_loaded:
        vt.load_model()
    vt.to(dtype=torch.bfloat16)
    model = model.bfloat16().cuda()
    model.eval()

    clip_proc = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform  = ResizeLongestSide(1024)
    print("Model ready.\n")
    return model, tokenizer, clip_proc, transform


PROMPT = (
    "Please answer begin with [CLS] for classification, "
    "if the image is tampered, output mask the tampered region."
)


def infer_one(model, tokenizer, clip_proc, transform, img_path: str) -> str:
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_clip = (
        clip_proc.preprocess(img, return_tensors="pt")["pixel_values"][0]
        .unsqueeze(0).cuda().bfloat16()
    )
    img_sam = transform.apply_image(img)
    img_sam_t = preprocess_sam(
        torch.from_numpy(img_sam).permute(2, 0, 1).contiguous()
    ).unsqueeze(0).cuda().bfloat16()

    conv = conversation_lib.conv_templates["llava_v1"].copy()
    conv.messages = []
    conv.append_message(
        conv.roles[0],
        DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        + "\n" + PROMPT,
    )
    conv.append_message(conv.roles[1], "")
    input_ids = tokenizer_image_token(
        conv.get_prompt(), tokenizer, return_tensors="pt"
    ).unsqueeze(0).cuda()

    with torch.no_grad():
        out_ids, _ = model.evaluate(
            img_clip, img_sam_t, input_ids,
            [img_sam.shape[:2]], [img.shape[:2]],
            max_new_tokens=512, tokenizer=tokenizer,
        )
    out_ids = out_ids[0][out_ids[0] != IMAGE_TOKEN_INDEX]
    return tokenizer.decode(out_ids, skip_special_tokens=False).replace("\n", " ").strip()


def parse_args():
    p = argparse.ArgumentParser(description="Baseline SIDA-7B evaluation (no fine-tuning).")
    p.add_argument("--model-dir",  type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--test-dir",   type=Path, default=DEFAULT_TEST_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, clip_proc, transform = load_model(args.model_dir)

    confusion   = defaultdict(lambda: defaultdict(int))
    sub_stats   = defaultdict(lambda: defaultdict(int))
    results     = {}
    wrong       = []
    correct = total = errors = 0

    t0 = time.time()

    for folder, true_cls in CLASS_MAP.items():
        fpath = args.test_dir / folder
        if not fpath.exists():
            print(f"[SKIP] {fpath}")
            continue
        files = sorted(
            f for f in os.listdir(fpath)
            if Path(f).suffix.lower() in IMG_EXTENSIONS
        )
        print(f"\n{'=' * 55}\n  {true_cls} ({folder}) — {len(files)} files\n{'=' * 55}")
        results[folder] = {}

        for fn in tqdm(files, desc=true_cls):
            try:
                resp = infer_one(model, tokenizer, clip_proc, transform, str(fpath / fn))
                pred = classify_response(resp)
            except Exception as exc:
                errors += 1
                pred = "ERROR"
                resp = str(exc)

            results[folder][fn] = {
                "true": true_cls, "pred": pred, "resp": resp[:200]
            }
            confusion[true_cls][pred] += 1
            sub_stats[folder][pred]   += 1
            total += 1
            if pred == true_cls:
                correct += 1
            else:
                wrong.append((true_cls, pred, folder, fn, resp[:150]))

        n = sum(confusion[true_cls].values())
        c = confusion[true_cls][true_cls]
        print(f"  → {true_cls}: {c}/{n} = {c/n*100:.1f}%")

    elapsed = time.time() - t0
    acc = correct / total * 100 if total else 0

    # Compute macro F1
    f1s, metrics = [], {}
    pred_labels = ALL_CLASSES + (
        ["UNKNOWN"] if any(confusion[t].get("UNKNOWN", 0) > 0 for t in ALL_CLASSES) else []
    )
    for cls in ALL_CLASSES:
        tot = sum(confusion[cls].values())
        cor = confusion[cls][cls]
        ps  = sum(confusion[c][cls] for c in ALL_CLASSES)
        pr  = cor / ps if ps else 0
        rc  = cor / tot if tot else 0
        f1  = 2 * pr * rc / (pr + rc) if (pr + rc) else 0
        f1s.append(f1)
        metrics[cls] = (pr, rc, f1)

    macro_f1 = np.mean(f1s)

    lines = []
    def w(s=""): lines.append(s)
    SEP = "=" * 60

    w(SEP); w("  BASELINE SIDA-7B — tourist deepfake dataset"); w(SEP)
    w(f"  Time: {elapsed:.1f}s  ({total/elapsed:.2f} img/s)")
    w(f"  Processed: {total}  |  Errors: {errors}")
    w(f"\n  OVERALL ACCURACY: {correct}/{total} = {acc:.2f}%\n")
    w(f"  {'Class':<12}{'Correct':>8}{'Total':>8}{'Accuracy':>11}")
    w("  " + "─" * 42)
    for cls in ALL_CLASSES:
        tot = sum(confusion[cls].values())
        cor = confusion[cls][cls]
        w(f"  {cls:<12}{cor:>8}{tot:>8}{cor/tot*100 if tot else 0:>10.2f}%")
    w(f"\n  Macro F1: {macro_f1:.4f}\n")

    w("  Confusion matrix (rows=true, cols=pred):")
    w(f"  {'true/pred':<12}" + "".join(f"{p:>12}" for p in pred_labels))
    w("  " + "─" * (12 + 12 * len(pred_labels)))
    for tl in ALL_CLASSES:
        w(f"  {tl:<12}" + "".join(f"{confusion[tl][pl]:>12}" for pl in pred_labels))

    w()
    w(f"\n  {'Class':<12}{'Precision':>11}{'Recall':>11}{'F1':>11}")
    w("  " + "─" * 47)
    for cls in ALL_CLASSES:
        pr, rc, f1 = metrics[cls]
        w(f"  {cls:<12}{pr:>11.4f}{rc:>11.4f}{f1:>11.4f}")

    if wrong:
        w()
        w(SEP); w(f"  WRONG PREDICTIONS — {len(wrong)}"); w(SEP)
        for i, (tc, pc, sf, fn, tx) in enumerate(wrong[:60], 1):
            w(f"  {i:>4}.  true={tc:<10} pred={pc:<10} [{sf}]  {fn}")
            w(f"         {tx}")

    report = "\n".join(lines)
    print("\n" + report)

    rp = args.output_dir / "baseline_report.txt"
    rp.write_text(report, encoding="utf-8")
    jp = args.output_dir / "baseline_results.json"
    jp.write_text(
        json.dumps({
            "accuracy": acc, "macro_f1": macro_f1,
            "confusion": {tl: dict(confusion[tl]) for tl in ALL_CLASSES},
            "per_image": results,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[OK] {rp}\n[OK] {jp}")


if __name__ == "__main__":
    main()
