#!/usr/bin/env python3
"""
detection/eval_metrics.py
──────────────────────────
Computes and prints evaluation metrics from a results.json file
produced by inference.py or baseline_eval.py.

Metrics reported:
    - Overall accuracy
    - Per-class accuracy
    - Confusion matrix
    - Precision, Recall, F1 (per class and macro-averaged)

Usage
─────
    python eval_metrics.py
    python eval_metrics.py --results ../results_final/results.json
    python eval_metrics.py --results ../results/baseline_results.json
"""

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_RESULTS = Path("./results_final/results.json")
LABEL_NAMES     = ["real", "fake", "tampered"]


def load_results(path: Path) -> list[dict]:
    """Load per-image predictions from a results.json file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if "per_image" in data:
        per_image = data["per_image"]
        # per_image may be a dict keyed by folder or a flat list
        if isinstance(per_image, dict):
            flat = []
            for folder, items in per_image.items():
                if isinstance(items, dict):
                    for fname, rec in items.items():
                        flat.append(rec)
                elif isinstance(items, list):
                    flat.extend(items)
            return flat
        return per_image
    raise ValueError(f"Cannot find 'per_image' key in {path}")


def normalise_label(label: str) -> str:
    """Normalise label strings to lowercase SIDA convention."""
    t = label.lower()
    if t in ("fake", "full_synt", "synthetic", "fully synthetic"):
        return "fake"
    if t in ("tampered", "altered", "manipulated"):
        return "tampered"
    if t in ("real", "authentic", "genuine"):
        return "real"
    return t


def compute_metrics(records: list[dict]) -> dict:
    valid = [r for r in records if "true" in r and "pred" in r]
    labels = LABEL_NAMES

    # Confusion matrix
    cm: dict[str, dict[str, int]] = {l: {l2: 0 for l2 in labels + ["unknown"]} for l in labels}
    for r in valid:
        t = normalise_label(r["true"])
        p = normalise_label(r["pred"])
        if t in labels:
            cm[t][p if p in labels else "unknown"] += 1

    results_per_label = {}
    f1s = []
    for label in labels:
        tot = sum(cm[label].values())
        tp  = cm[label][label]
        fp  = sum(cm[other][label] for other in labels if other != label)
        fn  = tot - tp
        acc = tp / tot if tot else 0
        pr  = tp / (tp + fp)  if (tp + fp)  else 0
        rec = tp / (tp + fn)  if (tp + fn)  else 0
        f1  = 2 * pr * rec / (pr + rec) if (pr + rec) else 0
        f1s.append(f1)
        results_per_label[label] = {
            "total": tot, "correct": tp,
            "accuracy": acc, "precision": pr, "recall": rec, "f1": f1,
        }

    total   = len(valid)
    correct = sum(1 for r in valid if normalise_label(r.get("true","")) == normalise_label(r.get("pred","")))
    return {
        "total": total,
        "correct": correct,
        "overall_accuracy": correct / total if total else 0,
        "macro_f1": float(np.mean(f1s)),
        "per_class": results_per_label,
        "confusion": cm,
    }


def print_report(m: dict, source_path: Path) -> None:
    SEP = "=" * 60
    print(SEP)
    print(f"  Metrics — {source_path.name}")
    print(SEP)
    print(f"\n  Overall accuracy : {m['correct']}/{m['total']} = "
          f"{m['overall_accuracy']*100:.2f}%")
    print(f"  Macro F1         : {m['macro_f1']:.4f}\n")

    print(f"  {'Class':<12}{'Total':>7}{'Correct':>9}{'Acc%':>7}  "
          f"{'Prec':>7}  {'Rec':>7}  {'F1':>7}")
    print("  " + "─" * 62)
    for label in LABEL_NAMES:
        pc = m["per_class"][label]
        print(f"  {label:<12}{pc['total']:>7}{pc['correct']:>9}"
              f"{pc['accuracy']*100:>6.1f}%  "
              f"{pc['precision']:>6.4f}  {pc['recall']:>6.4f}  {pc['f1']:>6.4f}")

    print(f"\n  Confusion matrix (rows=true, cols=pred):")
    all_cols = LABEL_NAMES + ["unknown"]
    print(f"  {'true/pred':<12}" + "".join(f"{c:>10}" for c in all_cols))
    print("  " + "─" * (12 + 10 * len(all_cols)))
    for label in LABEL_NAMES:
        row = m["confusion"][label]
        print(f"  {label:<12}" + "".join(f"{row.get(c,0):>10}" for c in all_cols))
    print()
    print(SEP)


def parse_args():
    p = argparse.ArgumentParser(description="Compute metrics from a results.json file.")
    p.add_argument("--results", type=Path, default=DEFAULT_RESULTS,
                   help="Path to results.json")
    p.add_argument("--json", action="store_true",
                   help="Also print metrics as JSON")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.results.exists():
        print(f"[ERROR] File not found: {args.results}")
        raise SystemExit(1)

    records = load_results(args.results)
    metrics = compute_metrics(records)
    print_report(metrics, args.results)

    if args.json:
        print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
