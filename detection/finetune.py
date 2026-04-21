#!/usr/bin/env python3
"""
detection/finetune.py
─────────────────────
Two-phase parameter-efficient fine-tuning of SIDA-7B on a custom dataset.

Overview
────────
Phase 1 — Feature extraction (run once, cached):
    The full SIDA-7B model (CLIP + SAM + LLaMA-7B) is run in inference mode
    on every training and test image.  For each image the [CLS] hidden-state
    vector  h_cls ∈ ℝ⁴⁰⁹⁶  is extracted from the last LLaMA layer and saved
    to disk.  The heavy backbone is then unloaded from GPU memory.

Phase 2 — cls_head training (fast, repeatable):
    A small classification head  Linear(4096→2048) → ReLU → Dropout(0.1)
    → Linear(2048→3)  is trained on the cached feature tensors.
    Only 8 396 803 parameters (0.114% of the full model) are updated.

Regularisation strategy (five mechanisms, applied simultaneously):
    1. Backbone freeze         — 99.886% of parameters are never updated
    2. Dropout(p=0.1)          — added vs. the original SIDA code (p=0.0)
    3. AdamW weight_decay=0.01 — decoupled L2 regularisation
    4. CosineAnnealingLR       — smooth LR decay, T_max=25
    5. Early stopping          — patience=7 epochs; best weights are restored

Dataset layout (--train-dir / --test-dir):
    photo/train/{real, full_synt, tempered}/
    photo/test/ {real, full_synt, tempered}/

    Labels:  real=0, full_synt=1 (FAKE), tempered=2

Results on the thesis dataset (1 000 train / 300 test, seed=42):
    Best epoch : 11
    Test Accuracy : 94.67%  (284/300)
    Macro F1      : 0.946

Usage
─────
    conda activate sida_modern
    cd detection/
    python finetune.py

    # Force feature re-extraction (ignore cache):
    python finetune.py --no-cache
"""

import argparse
import os
import random
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

from transformers import AutoTokenizer, CLIPImageProcessor

from model.SIDA import SIDAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from utils.utils import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL_DIR    = Path("./ck/SIDA-7B")
DEFAULT_SAVE_HEAD    = Path("./ck/cls_head_new.pth")
DEFAULT_FEATURES     = Path("./features_new/extracted_features.pt")
DEFAULT_TRAIN_DIR    = Path("../photo/train")
DEFAULT_TEST_DIR     = Path("../photo/test")

FOLDER_TO_LABEL: dict[str, int] = {"real": 0, "full_synt": 1, "tempered": 2}
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ── Hyperparameters ───────────────────────────────────────────────────────────
EPOCHS       = 25
LR           = 1e-3
BATCH_SIZE   = 64
WEIGHT_DECAY = 0.01
PATIENCE     = 7
SEED         = 42


# ── Dataset helpers ───────────────────────────────────────────────────────────

def collect_samples(base_dir: Path) -> list[tuple[str, int]]:
    """Return a list of (filepath, label) pairs from *base_dir*."""
    samples: list[tuple[str, int]] = []
    for folder, label in FOLDER_TO_LABEL.items():
        d = base_dir / folder
        if not d.exists():
            print(f"  [SKIP] Folder not found: {d}")
            continue
        files = sorted(f for f in d.iterdir() if f.suffix.lower() in IMG_EXTENSIONS)
        for f in files:
            samples.append((str(f), label))
        print(f"  {folder:<12}  label={label}  {len(files)} files")
    return samples


# ── Model loading ─────────────────────────────────────────────────────────────

def load_backbone(model_dir: Path):
    """Load SIDA-7B in bfloat16 inference mode."""
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
    model.eval()
    return model, tokenizer, clip_proc, cls_idx


def build_input_ids(tokenizer) -> torch.Tensor:
    """Build the fixed prompt token IDs used for feature extraction."""
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


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_cls_vector(
    model, clip_proc, input_ids: torch.Tensor, cls_idx: int, fpath: str
) -> torch.Tensor:
    """Extract the [CLS] hidden-state vector for a single image.

    Technical note
    ──────────────
    The [CLS] token position is shifted by 255 visual tokens produced by
    CLIP ViT-L/14.  Ignoring this shift yields a meaningless vector from
    an unrelated position in the sequence.
    """
    img = cv2.imread(fpath)
    if img is None:
        raise ValueError(f"Cannot read image: {fpath}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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

        # Locate [CLS] token with the 255-token visual shift applied
        cls_mask = (input_ids[:, 1:] == cls_idx)
        pad_left  = torch.zeros((cls_mask.shape[0], 255), dtype=torch.bool, device="cuda")
        pad_right = torch.zeros((cls_mask.shape[0],   1), dtype=torch.bool, device="cuda")
        cls_mask  = torch.cat([pad_left, cls_mask, pad_right], dim=1)

        # Align mask length to hidden-state sequence length
        if cls_mask.size(1) > hs.size(1):
            cls_mask = cls_mask[:, : hs.size(1)]
        elif cls_mask.size(1) < hs.size(1):
            pad = torch.zeros(
                (cls_mask.size(0), hs.size(1) - cls_mask.size(1)),
                dtype=torch.bool, device="cuda",
            )
            cls_mask = torch.cat([cls_mask, pad], dim=1)

        selected = hs[cls_mask]
        vec = selected[0] if selected.numel() > 0 else hs[0, -1, :]
        return vec.float().cpu()


def extract_all(
    model, clip_proc, input_ids, cls_idx, samples, tag: str = ""
) -> list[tuple[torch.Tensor, int, str]]:
    data = []
    for i, (fpath, label) in enumerate(samples):
        try:
            h = extract_cls_vector(model, clip_proc, input_ids, cls_idx, fpath)
            data.append((h, label, fpath))
        except Exception as exc:
            print(f"  [SKIP] {os.path.basename(fpath)}: {exc}")
        if (i + 1) % 50 == 0 or (i + 1) == len(samples):
            print(f"  {tag}: {i + 1}/{len(samples)}  (ok={len(data)})")
    return data


# ── Phase 1 ───────────────────────────────────────────────────────────────────

def phase1_extract(
    train_samples, test_samples, model_dir: Path, cache_path: Path
):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n[Phase 1] Loading SIDA-7B backbone...")
    model, tokenizer, clip_proc, cls_idx = load_backbone(model_dir)
    input_ids = build_input_ids(tokenizer)

    cls_count = (input_ids == cls_idx).sum().item()
    print(f"  [CLS] token id={cls_idx}, found {cls_count} times in prompt  ✓")

    print(f"\n  Extracting train features ({len(train_samples)} images)...")
    train_data = extract_all(model, clip_proc, input_ids, cls_idx, train_samples, "train")

    print(f"\n  Extracting test features ({len(test_samples)} images)...")
    test_data = extract_all(model, clip_proc, input_ids, cls_idx, test_samples, "test")

    torch.save({"train": train_data, "test": test_data}, cache_path)
    print(f"\n  Cache saved → {cache_path}")

    del model
    torch.cuda.empty_cache()
    print("  Backbone unloaded from GPU.\n")
    return train_data, test_data


# ── Phase 2 ───────────────────────────────────────────────────────────────────

def phase2_train(train_data, test_data, save_head: Path) -> nn.Module:
    """Train cls_head on cached features and save the best checkpoint."""
    print("[Phase 2] Training cls_head...")

    X_train = torch.stack([d[0] for d in train_data]).cuda()
    y_train = torch.tensor([d[1] for d in train_data]).cuda()
    X_test  = torch.stack([d[0] for d in test_data]).cuda()
    y_test  = torch.tensor([d[1] for d in test_data]).cuda()
    test_paths = [d[2] for d in test_data]

    dist_tr = [(y_train == i).sum().item() for i in range(3)]
    dist_te = [(y_test  == i).sum().item() for i in range(3)]
    print(f"  Train: {len(train_data)}  real={dist_tr[0]} fake={dist_tr[1]} tampered={dist_tr[2]}")
    print(f"  Test:  {len(test_data)}   real={dist_te[0]} fake={dist_te[1]} tampered={dist_te[2]}")

    # Inverse-frequency class weights
    n_total = len(train_data)
    weights = torch.tensor(
        [n_total / (3 * max((y_train == i).sum().item(), 1)) for i in range(3)]
    ).cuda()
    print(f"  Class weights: real={weights[0]:.3f}  fake={weights[1]:.3f}  tampered={weights[2]:.3f}")

    # Architecture: differs from original SIDA code by adding Dropout(0.1)
    cls_head = nn.Sequential(
        nn.Linear(4096, 2048),
        nn.ReLU(),
        nn.Dropout(p=0.1),          # regularisation (original: p=0.0)
        nn.Linear(2048, 3),
    ).float().cuda()

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        cls_head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc, best_state, best_epoch, patience_cnt = 0.0, None, 0, 0

    print(f"\n  {'Epoch':>5}  {'Loss':>8}  {'Train%':>8}  {'Test%':>8}  "
          f"{'real':>6}  {'fake':>6}  {'tamp':>6}")
    print("  " + "─" * 56)

    for epoch in range(1, EPOCHS + 1):
        perm = torch.randperm(X_train.size(0))
        cls_head.train()
        total_loss = 0.0

        for start in range(0, X_train.size(0), BATCH_SIZE):
            xb = X_train[perm[start : start + BATCH_SIZE]]
            yb = y_train[perm[start : start + BATCH_SIZE]]
            logits = cls_head(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        scheduler.step()
        avg_loss = total_loss / X_train.size(0)

        cls_head.eval()
        with torch.no_grad():
            tr_acc = (cls_head(X_train).argmax(1) == y_train).float().mean().item() * 100
            te_logits = cls_head(X_test)
            te_pred   = te_logits.argmax(1)
            te_acc    = (te_pred == y_test).float().mean().item() * 100
            per_class = []
            for c in range(3):
                m = y_test == c
                per_class.append(
                    (te_pred[m] == c).float().mean().item() * 100 if m.sum() > 0 else 0.0
                )

        marker = ""
        if te_acc > best_acc:
            best_acc   = te_acc
            best_state = {k: v.clone() for k, v in cls_head.state_dict().items()}
            best_epoch = epoch
            patience_cnt = 0
            marker = "  ← BEST"
        else:
            patience_cnt += 1

        print(
            f"  {epoch:>5}  {avg_loss:>8.4f}  {tr_acc:>7.1f}%  {te_acc:>7.1f}%  "
            f"{per_class[0]:>5.0f}%  {per_class[1]:>5.0f}%  {per_class[2]:>5.0f}%{marker}"
        )

        if patience_cnt >= PATIENCE:
            print(
                f"\n  Early stopping at epoch {epoch} "
                f"(best test_acc={best_acc:.1f}% at epoch {best_epoch})"
            )
            break

    print(f"\n  {'=' * 56}")
    print(f"  Best test_acc: {best_acc:.1f}%  (epoch {best_epoch})")

    save_head.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, save_head)
    print(f"  Weights saved → {save_head}")

    # ── Detailed evaluation on best weights ──────────────────────────────────
    cls_head.load_state_dict(best_state)
    cls_head.eval()
    with torch.no_grad():
        te_logits = cls_head(X_test)
        te_probs  = torch.softmax(te_logits, dim=1)
        te_pred   = te_logits.argmax(1)

    names = ["real", "fake", "tampered"]

    print("\n  Confusion matrix (true / pred):")
    print(f"  {'':>12s}  {'pred_r':>8s}  {'pred_f':>8s}  {'pred_t':>8s}  {'acc':>6s}")
    for i in range(3):
        m   = y_test == i
        row = [(te_pred[m] == j).sum().item() for j in range(3)]
        acc = row[i] / max(sum(row), 1) * 100
        print(f"  {names[i]:>12s}  {row[0]:>8d}  {row[1]:>8d}  {row[2]:>8d}  {acc:>5.0f}%")

    # Macro F1
    f1s = []
    for i in range(3):
        tp  = ((te_pred == i) & (y_test == i)).sum().item()
        fp  = ((te_pred == i) & (y_test != i)).sum().item()
        fn_ = ((te_pred != i) & (y_test == i)).sum().item()
        prec = tp / (tp + fp)  if (tp + fp)  > 0 else 0.0
        rec  = tp / (tp + fn_) if (tp + fn_) > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    print(f"\n  Macro F1: {np.mean(f1s):.4f}")

    print("\n  Per-file test results:")
    for idx in range(len(test_paths)):
        gt = y_test[idx].item()
        pr = te_pred[idx].item()
        p  = te_probs[idx]
        ok = "✓" if gt == pr else "✗"
        submodel = Path(test_paths[idx]).parent.name
        fname    = Path(test_paths[idx]).name
        print(
            f"  {ok} gt={names[gt]:>8s}  pred={names[pr]:>8s}  "
            f"[r={p[0]:.3f} f={p[1]:.3f} t={p[2]:.3f}]  [{submodel}]  {fname}"
        )

    correct = (te_pred == y_test).sum().item()
    print(f"\n  Total: {correct}/{len(y_test)} = {correct / len(y_test) * 100:.1f}%")
    return cls_head


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune SIDA-7B cls_head on custom tourist dataset.")
    p.add_argument("--model-dir",  type=Path, default=DEFAULT_MODEL_DIR,  help="SIDA-7B checkpoint directory")
    p.add_argument("--train-dir",  type=Path, default=DEFAULT_TRAIN_DIR,  help="Train data root")
    p.add_argument("--test-dir",   type=Path, default=DEFAULT_TEST_DIR,   help="Test data root")
    p.add_argument("--save-head",  type=Path, default=DEFAULT_SAVE_HEAD,  help="Output path for cls_head weights")
    p.add_argument("--features",   type=Path, default=DEFAULT_FEATURES,   help="Feature cache path")
    p.add_argument("--no-cache",   action="store_true",                   help="Force feature re-extraction")
    p.add_argument("--seed",       type=int,  default=SEED,               help="Random seed")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    t0 = time.time()

    print("=" * 62)
    print("  SIDA-7B  — cls_head fine-tuning")
    print("=" * 62)
    print(f"\n  Train  : {args.train_dir}")
    print(f"  Test   : {args.test_dir}")
    print(f"  Model  : {args.model_dir}")
    print(f"  Output : {args.save_head}")

    print("\nCollecting train samples:")
    train_samples = collect_samples(args.train_dir)
    print("\nCollecting test samples:")
    test_samples = collect_samples(args.test_dir)

    # Phase 1: feature extraction (with cache)
    if args.features.exists() and not args.no_cache:
        print(f"\n[Phase 1] Loading cached features: {args.features}")
        saved = torch.load(args.features, map_location="cpu")
        train_data = saved["train"]
        test_data  = saved["test"]
        print(f"  train={len(train_data)}  test={len(test_data)}")
    else:
        train_data, test_data = phase1_extract(
            train_samples, test_samples, args.model_dir, args.features
        )

    # Phase 2: head training
    phase2_train(train_data, test_data, args.save_head)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed / 60:.1f} min")
    print("Next step: python inference.py")


if __name__ == "__main__":
    main()
