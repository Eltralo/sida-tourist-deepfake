#!/usr/bin/env python3
"""
detection/inference.py
───────────────────────
Full inference pipeline for the fine-tuned SIDA-7B model.

For each test image:
    1. Extracts [CLS] hidden-state vector and classifies via cls_head
       → REAL (0) / FAKE (1) / TAMPERED (2)
    2. For images predicted as TAMPERED: generates a binary segmentation
       mask via the SAM decoder and saves a visual overlay.

Output (--output-dir):
    report.txt     — human-readable per-class and per-tool breakdown
    results.json   — per-image predictions, probabilities, mask coverage
    masks/         — binary PNG masks for TAMPERED predictions
    overlays/      — original + red-channel overlay with coverage label

Usage
─────
    conda activate sida_modern
    cd detection/
    python inference.py

    python inference.py --test-dir ../photo/test --output-dir ../results_final
"""

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

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
    IMAGE_TOKEN_INDEX,
)

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL_DIR  = Path("./ck/SIDA-7B")
DEFAULT_HEAD_PATH  = Path("./ck/cls_head_new.pth")
DEFAULT_TEST_DIR   = Path("../photo/test")
DEFAULT_OUTPUT_DIR = Path("./results_final")

FOLDER_TO_LABEL = {"real": 0, "full_synt": 1, "tempered": 2}
LABEL_NAMES     = ["real", "fake", "tampered"]
IMG_EXTENSIONS  = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

SAM_IMG_SIZE = 1024
SAM_PIXEL_MEAN = torch.Tensor([123.675, 116.28,  103.53]).view(-1, 1, 1)
SAM_PIXEL_STD  = torch.Tensor([ 58.395,  57.12,   57.375]).view(-1, 1, 1)


def preprocess_sam(x: torch.Tensor, img_size: int = SAM_IMG_SIZE) -> torch.Tensor:
    x = (x - SAM_PIXEL_MEAN) / SAM_PIXEL_STD
    h, w = x.shape[-2:]
    return F.pad(x, (0, img_size - w, 0, img_size - h))


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_and_head(model_dir: Path, head_path: Path):
    print("[1/3] Loading SIDA-7B...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), model_max_length=512, padding_side="right", use_fast=False
    )
    tokenizer.pad_token = tokenizer.unk_token
    seg_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    cls_idx = tokenizer("[CLS]", add_special_tokens=False).input_ids[0]

    model = SIDAForCausalLM.from_pretrained(
        str(model_dir),
        low_cpu_mem_usage=True,
        vision_tower="openai/clip-vit-large-patch14",
        seg_token_idx=seg_idx,
        cls_token_idx=cls_idx,
        torch_dtype=torch.bfloat16,
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model = model.bfloat16().cuda()

    vt = model.get_model().get_vision_tower()
    if not vt.is_loaded:
        vt.load_model()
    vt.bfloat16().cuda()

    clip_proc = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform  = ResizeLongestSide(SAM_IMG_SIZE)

    print(f"[2/3] Loading cls_head: {head_path}")
    state = torch.load(str(head_path), map_location="cpu")
    head = nn.Sequential(
        nn.Linear(4096, 2048),
        nn.ReLU(),
        nn.Dropout(p=0.1),
        nn.Linear(2048, 3),
    )
    head.load_state_dict(state)
    head = head.bfloat16().cuda()

    # Replace the model's classification head in-place
    model.get_model().cls_head[0] = head
    model.eval()
    print("    cls_head loaded  ✓")
    return model, tokenizer, clip_proc, transform, cls_idx


def build_input_ids(tokenizer) -> torch.Tensor:
    prompt = (
        "Please answer begin with [CLS] for classification, "
        "if the image is tampered, output mask the tampered region."
    )
    conv = conversation_lib.conv_templates["llava_v1"].copy()
    conv.messages = []
    full = (
        DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        + "\n" + prompt
    )
    conv.append_message(conv.roles[0], full)
    conv.append_message(conv.roles[1], "")
    return tokenizer_image_token(
        conv.get_prompt(), tokenizer, return_tensors="pt"
    ).unsqueeze(0).cuda()


# ── Per-image inference ───────────────────────────────────────────────────────

def predict_one(model, clip_proc, input_ids, cls_idx, fpath: str):
    img_bgr = cv2.imread(fpath)
    if img_bgr is None:
        raise ValueError(f"Cannot read: {fpath}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img_clip = (
        clip_proc.preprocess(img_rgb, return_tensors="pt")["pixel_values"][0]
        .unsqueeze(0).cuda().bfloat16()
    )

    with torch.no_grad():
        attn = torch.ones_like(input_ids, device="cuda")
        fw = LlavaLlamaForCausalLM.forward(
            model,
            images=img_clip,
            attention_mask=attn,
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False,
        )
        hs = fw.hidden_states[-1]
        if hs.dim() == 2:
            hs = hs.unsqueeze(0)

        # [CLS] position with 255-token visual shift
        cls_mask = (input_ids[:, 1:] == cls_idx)
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

        logits   = model.get_model().cls_head[0](hs)
        selected = logits[cls_mask]
        cls_logits = selected[0] if selected.numel() > 0 else logits[0, -1, :]

        probs = torch.softmax(cls_logits.float(), dim=-1)
        pred  = int(probs.argmax().item())

    return pred, probs.cpu(), img_bgr, img_rgb


def generate_mask(model, tokenizer, clip_proc, transform, input_ids, img_bgr, img_rgb):
    img_clip = (
        clip_proc.preprocess(img_rgb, return_tensors="pt")["pixel_values"][0]
        .unsqueeze(0).cuda().bfloat16()
    )
    img_sam_np = transform.apply_image(img_rgb)
    img_sam_t  = preprocess_sam(
        torch.from_numpy(img_sam_np).permute(2, 0, 1).contiguous()
    ).unsqueeze(0).cuda().bfloat16()

    with torch.no_grad():
        result = model.evaluate(
            img_clip, img_sam_t, input_ids,
            [img_sam_np.shape[:2]], [list(img_rgb.shape[:2])],
            max_new_tokens=512, tokenizer=tokenizer,
        )
    pred_masks = result[1] if isinstance(result, tuple) and len(result) >= 2 else []
    if pred_masks and pred_masks[0] is not None:
        m = pred_masks[0].squeeze().cpu().numpy()
        return (m > 0.5).astype(np.uint8)
    return None


def save_mask_overlay(mask_np, img_bgr, stem, mask_dir, overlay_dir):
    h, w = img_bgr.shape[:2]
    mask_res = cv2.resize((mask_np * 255).astype(np.uint8), (w, h))
    cv2.imwrite(str(mask_dir / f"{stem}_mask.png"), mask_res)

    overlay = img_bgr.copy()
    overlay[mask_res > 127] = [0, 0, 255]
    blended = cv2.addWeighted(img_bgr, 0.55, overlay, 0.45, 0)
    coverage = mask_res.astype(bool).sum() / mask_res.size * 100
    cv2.putText(blended, "TAMPERED REGION", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(blended, f"Coverage: {coverage:.1f}%", (12, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(overlay_dir / f"{stem}_overlay.png"), blended)
    return coverage


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Inference with fine-tuned SIDA-7B cls_head.")
    p.add_argument("--model-dir",   type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--head-path",   type=Path, default=DEFAULT_HEAD_PATH)
    p.add_argument("--test-dir",    type=Path, default=DEFAULT_TEST_DIR)
    p.add_argument("--output-dir",  type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--no-masks",    action="store_true", help="Skip SAM mask generation")
    return p.parse_args()


def main():
    args = parse_args()
    (args.output_dir / "masks").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "overlays").mkdir(parents=True, exist_ok=True)

    model, tokenizer, clip_proc, transform, cls_idx = load_model_and_head(
        args.model_dir, args.head_path
    )
    input_ids = build_input_ids(tokenizer)

    # Collect test files
    all_files = []
    for folder, label in FOLDER_TO_LABEL.items():
        d = args.test_dir / folder
        if not d.exists():
            continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in IMG_EXTENSIONS:
                all_files.append((str(f), label, folder))

    print(f"\n[3/3] Running inference on {len(all_files)} images...\n")

    results, confusion, mask_stats, errors = [], [[0]*3 for _ in range(3)], [], []
    t0 = time.time()

    for fpath, true_label, folder in tqdm(all_files, desc="Inference"):
        fname = Path(fpath).name
        stem  = Path(fpath).stem
        try:
            pred, probs, img_bgr, img_rgb = predict_one(
                model, clip_proc, input_ids, cls_idx, fpath
            )
            confusion[true_label][pred] += 1

            mask_cov = None
            if pred == 2 and not args.no_masks:
                try:
                    mask_np = generate_mask(
                        model, tokenizer, clip_proc, transform,
                        input_ids, img_bgr, img_rgb
                    )
                    if mask_np is not None:
                        mask_cov = save_mask_overlay(
                            mask_np, img_bgr, stem,
                            args.output_dir / "masks",
                            args.output_dir / "overlays",
                        )
                        mask_stats.append({
                            "file": fname, "folder": folder,
                            "coverage": round(mask_cov, 2),
                        })
                except Exception as me:
                    print(f"\n  [MASK ERROR] {fname}: {me}")

            results.append({
                "file":       fname,
                "folder":     folder,
                "true":       LABEL_NAMES[true_label],
                "pred":       LABEL_NAMES[pred],
                "correct":    pred == true_label,
                "probs":      {n: round(probs[i].item() * 100, 1)
                               for i, n in enumerate(LABEL_NAMES)},
                "mask_cov_%": mask_cov,
            })
        except Exception as exc:
            errors.append({"file": fname, "error": str(exc)})

    elapsed  = time.time() - t0
    total    = len(results)
    correct  = sum(1 for r in results if r["correct"])

    # ── Build report ──────────────────────────────────────────────────────────
    lines = []
    def w(s=""): lines.append(s)

    w("=" * 65)
    w("  RESULTS — fine-tuned SIDA-7B (cls_head_new.pth)")
    w("=" * 65)
    w(f"  Time: {elapsed:.1f}s  ({len(results)/elapsed:.2f} img/s)")
    w(f"  Processed: {total}  |  Errors: {len(errors)}")
    w()
    w(f"  OVERALL ACCURACY: {correct}/{total} = {correct/total*100:.2f}%")
    w()
    w(f"  {'Class':<12}{'Correct':>8}{'Total':>8}{'Accuracy':>11}")
    w("  " + "─" * 40)
    for i, name in enumerate(LABEL_NAMES):
        tot = sum(r["true"] == name for r in results)
        cor = sum(r["true"] == name and r["correct"] for r in results)
        w(f"  {name:<12}{cor:>8}{tot:>8}{cor/tot*100 if tot else 0:>10.2f}%")
    w()
    w("  Confusion matrix (rows=true, cols=pred):")
    w(f"  {'':>12}" + "".join(f"{'→'+n:>12}" for n in LABEL_NAMES))
    w("  " + "─" * (12 + 12 * 3))
    for i, name in enumerate(LABEL_NAMES):
        w(f"  {name:>12}" + "".join(f"{confusion[i][j]:>12}" for j in range(3)))
    w()

    # Per-tool breakdown
    tools = sorted(set(r["folder"] for r in results))
    w("=" * 65)
    w("  BREAKDOWN BY SOURCE")
    w("=" * 65)
    for sf in tools:
        sub = [r for r in results if r["folder"] == sf]
        tot = len(sub); cor = sum(1 for r in sub if r["correct"])
        preds = {n: sum(1 for r in sub if r["pred"] == n) for n in LABEL_NAMES}
        detail = "  ".join(f"{n[0]}:{preds[n]:>3}" for n in LABEL_NAMES)
        w(f"  {sf:<22}{tot:>7}{cor:>8}{cor/tot*100 if tot else 0:>7.1f}%  [{detail}]")

    # Wrong predictions
    wrong = [r for r in results if not r["correct"]]
    if wrong:
        w()
        w("=" * 65)
        w(f"  WRONG PREDICTIONS ({len(wrong)})")
        w("=" * 65)
        for r in wrong:
            p = r["probs"]
            w(f"  ✗  true={r['true']:<10}  pred={r['pred']:<10}  "
              f"[r={p['real']:5.1f}% f={p['fake']:5.1f}% t={p['tampered']:5.1f}%]  "
              f"[{r['folder']}]  {r['file']}")

    report = "\n".join(lines)
    print("\n" + report)

    (args.output_dir / "report.txt").write_text(report, encoding="utf-8")
    (args.output_dir / "results.json").write_text(
        json.dumps({
            "summary": {"accuracy": correct / total * 100, "correct": correct, "total": total},
            "confusion": confusion,
            "per_image": results,
            "masks": mask_stats,
            "errors": errors,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n[OK] {args.output_dir}/report.txt")
    print(f"[OK] {args.output_dir}/results.json")
    if mask_stats:
        avg_cov = sum(m["coverage"] for m in mask_stats) / len(mask_stats)
        print(f"[OK] {len(mask_stats)} masks  |  avg coverage: {avg_cov:.1f}%")


if __name__ == "__main__":
    main()
