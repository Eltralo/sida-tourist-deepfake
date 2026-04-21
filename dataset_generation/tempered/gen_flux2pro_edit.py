#!/usr/bin/env python3
"""
dataset_generation/tempered/gen_flux2pro_edit.py
─────────────────────────────────────────────────
Generates partially synthetic (tempered) tourist photographs via the ruGPT.io API.
Method : image-to-image background replacement at 2K resolution
         A real photograph of a person is uploaded; Flux 2 Pro (edit mode) replaces
         only the background with a Russian landmark while keeping the foreground
         person completely unchanged.
Model  : flux-2/pro  (Black Forest Labs, 32B parameters) in image-editing mode
Output : dataset/tempered/flux2pro_edit/
Total generated for the thesis dataset: 192 images

Usage
─────
    export RUGPT_API_KEY="your_key_here"
    python gen_flux2pro_edit.py

    python gen_flux2pro_edit.py --num 50 --real-dir /path/to/real --output /path/to/output
"""

import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from _api_client import get_api_key, upload_file, submit_img2img, poll_result, download_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

MODEL          = "flux-2/pro"
ASPECT_RATIO   = "3:2"
RESOLUTION     = "2K"
ENHANCE_PROMPT = False
DEFAULT_SEED   = 4031
DEFAULT_N      = 192
DEFAULT_REAL   = Path(__file__).parent.parent.parent / "dataset" / "real"
DEFAULT_OUTPUT = Path(__file__).parent.parent.parent / "dataset" / "tempered" / "flux2pro_edit"
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

_PRESERVE_PERSON = (
    "Keep the person in the foreground completely unchanged — "
    "do not alter their face, hair, clothing, pose, or any body part. "
    "Only replace the background behind them with "
)

BACKGROUNDS = [
    _PRESERVE_PERSON + "Red Square in Moscow, Russia. Saint Basil Cathedral with colorful onion domes, red Kremlin wall, cobblestone pavement.",
    _PRESERVE_PERSON + "the State Hermitage Museum and Winter Palace in Saint Petersburg, Russia. Grand baroque facade in pale green and white, Palace Square with Alexander Column.",
    _PRESERVE_PERSON + "Peterhof Grand Cascade near Saint Petersburg, Russia. Golden statues, powerful fountains, grand palace on the hill.",
    _PRESERVE_PERSON + "the Church of the Savior on Spilled Blood in Saint Petersburg, Russia. Elaborate multicolored onion domes, Griboedov Canal.",
    _PRESERVE_PERSON + "Bolshoi Theatre in Moscow, Russia. Grand neoclassical facade with columns, Theatre Square with fountain.",
    _PRESERVE_PERSON + "Zaryadye Park floating bridge over Moskva River, Kremlin towers visible behind. Moscow, Russia.",
    _PRESERVE_PERSON + "VDNKh exhibition center in Moscow, Russia. Golden Friendship of Nations fountain, Soviet-era pavilions.",
    _PRESERVE_PERSON + "Moscow State University Stalinist tower on Sparrow Hills, Russia. Symmetrical park and fountain in front.",
    _PRESERVE_PERSON + "Nevsky Prospect avenue in Saint Petersburg, Russia. Elegant 19th century facades, trolleybus wires, cafe terraces.",
    _PRESERVE_PERSON + "Kazan Cathedral colonnade on Nevsky Prospect, Saint Petersburg, Russia. Massive semicircular columns.",
    _PRESERVE_PERSON + "Novodevichy Convent in Moscow, Russia. White fortress walls, golden bell tower reflected in monastery pond.",
    _PRESERVE_PERSON + "the Bronze Horseman monument on Senate Square, Saint Petersburg, Russia. Saint Isaac Cathedral dome visible behind.",
    _PRESERVE_PERSON + "Old Arbat pedestrian street in Moscow, Russia. Historic 19th century buildings, street musicians, cafes.",
    _PRESERVE_PERSON + "Tsaritsyno Palace in Moscow, Russia. Ornate red brick pseudo-gothic palace, beautiful park with ponds.",
    _PRESERVE_PERSON + "Kolomenskoye park in Moscow, Russia. White Church of the Ascension, apple orchards, Moskva River view.",
    _PRESERVE_PERSON + "Lakhta Center skyscraper in Saint Petersburg, Russia. Twisted glass supertall tower, Gulf of Finland embankment.",
    _PRESERVE_PERSON + "GUM department store on Red Square, Moscow. Ornate 19th century facade with glass-roofed galleries.",
    _PRESERVE_PERSON + "Palace Bridge over Neva River in Saint Petersburg. Vasilievsky Island Strelka with Rostral Columns.",
    _PRESERVE_PERSON + "Moscow City skyscrapers, Federation Tower, view from Moskva River embankment.",
    _PRESERVE_PERSON + "Izmailovsky Kremlin in Moscow, Russia. Colorful fairytale wooden towers, traditional Russian architecture.",
    _PRESERVE_PERSON + "Red Square in Moscow in winter. Snow-covered cobblestones, Saint Basil Cathedral dusted with snow.",
    _PRESERVE_PERSON + "Peterhof in autumn. Golden foliage around the fountains, Saint Petersburg, Russia.",
    _PRESERVE_PERSON + "Nevsky Prospect in Saint Petersburg in winter evening. Street lamps on, snow on sidewalks, deep blue twilight sky.",
    _PRESERVE_PERSON + "Zaryadye Park in Moscow in autumn. Golden and crimson foliage, floating bridge, Kremlin towers.",
    _PRESERVE_PERSON + "Hermitage Museum in Saint Petersburg in summer. Bright blue sky, tourists on Palace Square.",
]


def collect_real_photos(real_dir: Path) -> list[Path]:
    photos = [f for f in real_dir.iterdir() if f.suffix.lower() in IMG_EXTENSIONS]
    if not photos:
        raise FileNotFoundError(f"No images found in {real_dir}.")
    log.info("  Found %d real photos in %s", len(photos), real_dir)
    return photos


def save_log(combos: list, log_path: Path) -> None:
    data = {
        "metadata": {
            "model": MODEL,
            "resolution": RESOLUTION,
            "method": "image-to-image background replacement",
            "class": "tempered",
            "total_generated": sum(1 for c in combos if c.get("filename")),
            "total_planned": len(combos),
            "timestamp": datetime.now().isoformat(),
        },
        "images": combos,
    }
    log_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Log saved → %s", log_path)


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate tempered images (background replacement) via Flux 2 Pro edit mode.")
    p.add_argument("--num",      type=int,  default=DEFAULT_N,       help="Number of images to generate")
    p.add_argument("--real-dir", type=Path, default=DEFAULT_REAL,    help="Directory with real source photos")
    p.add_argument("--output",   type=Path, default=DEFAULT_OUTPUT,   help="Output directory")
    p.add_argument("--seed",     type=int,  default=DEFAULT_SEED,     help="Random seed")
    p.add_argument("--yes",      action="store_true",                 help="Skip confirmation prompt")
    return p.parse_args()


def main():
    args = parse_args()
    api_key = get_api_key()
    args.output.mkdir(parents=True, exist_ok=True)
    log_path = args.output / "generation_log.json"
    random.seed(args.seed)

    all_photos = collect_real_photos(args.real_dir)

    photo_cycle = []
    while len(photo_cycle) < args.num:
        batch = all_photos[:]
        random.shuffle(batch)
        photo_cycle.extend(batch)
    photo_cycle = photo_cycle[:args.num]

    bg_cycle = []
    while len(bg_cycle) < args.num:
        batch = BACKGROUNDS[:]
        random.shuffle(batch)
        bg_cycle.extend(batch)
    bg_cycle = bg_cycle[:args.num]

    combos = [
        {"index": i, "source_photo": str(photo_cycle[i]), "background_prompt": bg_cycle[i]}
        for i in range(args.num)
    ]

    log.info("=" * 60)
    log.info("  Flux 2 Pro edit (tempered) — %d images  |  output: %s", args.num, args.output)
    log.info("=" * 60)

    # Test first image
    test = combos[0]
    file_uuid = upload_file(Path(test["source_photo"]), api_key)
    if not file_uuid:
        log.error("Upload failed. Exiting."); return

    job_uuid, _ = submit_img2img(test["background_prompt"], file_uuid, api_key, MODEL,
                                  ASPECT_RATIO, RESOLUTION, ENHANCE_PROMPT)
    if not job_uuid:
        log.error("Cannot submit. Exiting."); return

    result = poll_result(job_uuid, api_key)
    if result["status"] != "completed":
        log.error("Test failed: %s", result); return

    test_price = result.get("price") or 0
    log.info("  Price: %d coins | Estimated total: %d coins", test_price, test_price * args.num)
    urls = result.get("urls") or []
    if isinstance(urls, str):
        urls = [urls]
    if urls:
        fn = f"flux_edit_{test['index']:04d}.png"
        sz = download_image(urls[0], args.output / fn)
        test.update({"filename": fn, "price": test_price, "job_uuid": job_uuid, "status": "completed"})
        log.info("  Saved: %s  (%d KB)", fn, sz // 1024)
    save_log(combos, log_path)

    if not args.yes:
        answer = input(f"\nContinue generating remaining {len(combos) - 1} images? [y/N] ").strip().lower()
        if answer != "y":
            log.info("Aborted."); return

    total_cost = test_price
    for combo in combos[1:]:
        idx = combo["index"]
        log.info("[%3d/%d]  %s | bg: %s...",
                 idx + 1, args.num,
                 Path(combo["source_photo"]).name,
                 combo["background_prompt"][len(_PRESERVE_PERSON):60])

        file_uuid = upload_file(Path(combo["source_photo"]), api_key)
        if not file_uuid:
            combo["status"] = "upload_failed"; continue
        combo["file_uuid"] = file_uuid

        job_uuid, _ = submit_img2img(combo["background_prompt"], file_uuid, api_key, MODEL,
                                      ASPECT_RATIO, RESOLUTION, ENHANCE_PROMPT)
        if not job_uuid:
            combo["status"] = "submit_failed"; continue

        result = poll_result(job_uuid, api_key)
        combo["job_uuid"] = job_uuid
        if result["status"] != "completed":
            combo.update({"status": result["status"], "error": result.get("error", "")}); continue

        price = result.get("price") or 0
        total_cost += price
        combo["price"] = price
        urls = result.get("urls") or []
        if isinstance(urls, str):
            urls = [urls]
        if urls:
            fn = f"flux_edit_{idx:04d}.png"
            sz = download_image(urls[0], args.output / fn)
            combo.update({"filename": fn, "status": "completed"})
            log.info("  OK  %s  (%d KB)  cost:%d  total:%d", fn, sz // 1024, price, total_cost)
        else:
            combo["status"] = "no_urls"
        save_log(combos, log_path)
        time.sleep(1)

    success = sum(1 for c in combos if c.get("filename"))
    log.info("DONE  %d/%d  |  total cost: %d coins", success, args.num, total_cost)
    save_log(combos, log_path)


if __name__ == "__main__":
    main()
