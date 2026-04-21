#!/usr/bin/env python3
"""
dataset_generation/split_dataset.py
─────────────────────────────────────
Splits the assembled raw dataset into train and test subsets
with stratification by class and sub-model source.

Input layout (--source-dir):
    dataset/
        real/           *.jpg / *.png          (438 images in the thesis)
        full_synthetic/ flux2pro/  *.png        (152)
                        seedream45/ *.png       (123)
                        zimage/     *.png       (122)
                        imagen4/    *.png       ( 31)
        tempered/       nano_banana/ *.png      (241)
                        flux2pro_edit/ *.png    (192)

Output layout (--output-dir):
    photo/
        train/
            real/          338 images
            full_synt/     329 images
            tempered/      333 images
        test/
            real/          100 images
            full_synt/     100 images
            tempered/      100 images

Split ratio: 1000 train / 300 test  (seed=42)

Usage
─────
    python split_dataset.py
    python split_dataset.py --source-dir /path/to/dataset --output-dir /path/to/photo --seed 42
"""

import argparse
import random
import shutil
import json
from pathlib import Path


# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_SOURCE = Path(__file__).parent.parent / "dataset"
DEFAULT_OUTPUT = Path(__file__).parent.parent / "photo"
DEFAULT_SEED   = 42

# Per-class test counts matching the thesis split
TEST_COUNT = {"real": 100, "full_synt": 100, "tempered": 100}

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def collect_images(source_dir: Path) -> dict[str, list[Path]]:
    """Return a dict mapping class name to sorted list of image paths."""
    classes = {}

    # ── real ──────────────────────────────────────────────────────────────────
    real_dir = source_dir / "real"
    if real_dir.exists():
        classes["real"] = sorted(
            f for f in real_dir.rglob("*") if f.suffix.lower() in IMG_EXTENSIONS
        )
    else:
        print(f"  WARN: {real_dir} not found — skipping 'real' class.")
        classes["real"] = []

    # ── full_synthetic ────────────────────────────────────────────────────────
    fs_dir = source_dir / "full_synthetic"
    if fs_dir.exists():
        classes["full_synt"] = sorted(
            f for f in fs_dir.rglob("*") if f.suffix.lower() in IMG_EXTENSIONS
        )
    else:
        print(f"  WARN: {fs_dir} not found — skipping 'full_synt' class.")
        classes["full_synt"] = []

    # ── tempered ──────────────────────────────────────────────────────────────
    t_dir = source_dir / "tempered"
    if t_dir.exists():
        classes["tempered"] = sorted(
            f for f in t_dir.rglob("*") if f.suffix.lower() in IMG_EXTENSIONS
        )
    else:
        print(f"  WARN: {t_dir} not found — skipping 'tempered' class.")
        classes["tempered"] = []

    return classes


def split_and_copy(classes: dict, output_dir: Path, seed: int) -> dict:
    """Perform stratified split and copy files to output_dir/train and test."""
    rng = random.Random(seed)
    report = {}

    for cls_name, all_files in classes.items():
        if not all_files:
            print(f"  WARN: No files for class '{cls_name}' — skipping.")
            continue

        shuffled = all_files[:]
        rng.shuffle(shuffled)

        n_test  = TEST_COUNT.get(cls_name, 100)
        n_train = len(shuffled) - n_test

        if n_train < 1:
            print(f"  WARN: Not enough files for class '{cls_name}' "
                  f"(have {len(shuffled)}, need {n_test} for test).")
            n_test  = len(shuffled)
            n_train = 0

        test_files  = shuffled[:n_test]
        train_files = shuffled[n_test:]

        for split, files in (("train", train_files), ("test", test_files)):
            dest = output_dir / split / cls_name
            dest.mkdir(parents=True, exist_ok=True)
            for src in files:
                shutil.copy2(src, dest / src.name)

        report[cls_name] = {"train": len(train_files), "test": len(test_files)}
        print(f"  {cls_name:12s}  total={len(all_files):4d}  "
              f"train={len(train_files):4d}  test={len(test_files):4d}")

    return report


def parse_args():
    p = argparse.ArgumentParser(description="Split raw dataset into train/test subsets.")
    p.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE,
                   help="Root directory containing raw class folders")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                   help="Output directory for train/ and test/ subsets")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help="Random seed for reproducible split")
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 55)
    print("  Dataset split")
    print(f"  Source : {args.source_dir}")
    print(f"  Output : {args.output_dir}")
    print(f"  Seed   : {args.seed}")
    print("=" * 55)

    classes = collect_images(args.source_dir)

    print("\nCollected images:")
    for cls, files in classes.items():
        print(f"  {cls:12s}  {len(files)} files")

    print("\nSplitting and copying...")
    report = split_and_copy(classes, args.output_dir, args.seed)

    # Save split manifest
    manifest = {
        "seed": args.seed,
        "source_dir": str(args.source_dir),
        "output_dir": str(args.output_dir),
        "split": report,
    }
    manifest_path = args.output_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nDone!")
    print(f"  Manifest saved → {manifest_path}")
    total_train = sum(v["train"] for v in report.values())
    total_test  = sum(v["test"]  for v in report.values())
    print(f"  Total: {total_train} train  /  {total_test} test")
    print("=" * 55)


if __name__ == "__main__":
    main()
