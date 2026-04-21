#!/usr/bin/env python3
"""
dataset_generation/full_synthetic/gen_seedream45.py
────────────────────────────────────────────────────
Generates fully synthetic tourist photographs via the ruGPT.io API.
Model  : seedream/4.5-text-to-image  (ByteDance, scalable diffusion transformer)
Domain : Moscow & Saint Petersburg landmarks, diverse Russian pedestrians
Output : dataset/full_synthetic/seedream45/
Total generated for the thesis dataset: 123 images

Note on prompt length
─────────────────────
Seedream 4.5 accepts prompts up to 3 000 characters.
The builder in this script stays safely within 2 900 characters.

Usage
─────
    export RUGPT_API_KEY="your_key_here"
    python gen_seedream45.py

    # Optional overrides:
    python gen_seedream45.py --num 50 --output /path/to/output --seed 42
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
from _api_client import get_api_key, submit_text2img, poll_result, download_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Generation defaults ───────────────────────────────────────────────────────
MODEL         = "seedream/4.5-text-to-image"
ASPECT_RATIO  = "3:2"
ENHANCE_PROMPT = False
DEFAULT_SEED   = 3015
DEFAULT_N      = 123
DEFAULT_OUTPUT = Path(__file__).parent.parent.parent / "dataset" / "full_synthetic" / "seedream45"
MAX_PROMPT_LEN = 2900

# ── Prompt components (shorter descriptions to fit the 3 000-char limit) ──────
LANDMARKS = [
    {"name": "Red Square, Moscow",
     "desc": "Red Square (Krasnaya Ploshchad) in Moscow, Russia. Saint Basil Cathedral with colorful onion domes. Red Kremlin wall. Cobblestone pavement. Russian flags."},
    {"name": "Hermitage, Saint Petersburg",
     "desc": "State Hermitage Museum and Winter Palace in Saint Petersburg, Russia. Grand baroque facade in pale green and white with gold ornaments. Palace Square with Alexander Column."},
    {"name": "Moscow City skyscrapers",
     "desc": "Moscow International Business Center in Moscow, Russia. Modern glass skyscrapers including Federation Tower. View from Moskva River embankment."},
    {"name": "Church of Savior on Spilled Blood, Saint Petersburg",
     "desc": "Church of the Savior on Spilled Blood in Saint Petersburg, Russia. Multicolored onion domes with mosaics. Griboedov Canal in foreground."},
    {"name": "Bolshoi Theatre, Moscow",
     "desc": "Bolshoi Theatre in Moscow, Russia. Grand neoclassical facade with Ionic columns. Bronze quadriga of Apollo. Theatre Square with fountain."},
    {"name": "Zaryadye Park, Moscow",
     "desc": "Zaryadye Park in Moscow, Russia. Floating bridge over Moskva River. Kremlin towers in background. Modern landscape with wild Russian plants."},
    {"name": "Kazan Cathedral, Saint Petersburg",
     "desc": "Kazan Cathedral on Nevsky Prospect in Saint Petersburg, Russia. Massive semicircular colonnade of 96 columns. Fountain in front."},
    {"name": "Peterhof Palace, near Saint Petersburg",
     "desc": "Peterhof Palace near Saint Petersburg, Russia. Grand Cascade with golden statues and powerful fountains. Grand Palace on the hill."},
    {"name": "Tsaritsyno Palace, Moscow",
     "desc": "Tsaritsyno Museum-Reserve in Moscow, Russia. Ornate red brick pseudo-gothic palace. Beautiful park with ponds."},
    {"name": "VDNKh, Moscow",
     "desc": "VDNKh exhibition center in Moscow, Russia. Golden Friendship of Nations fountain with 16 gilded figures. Grand Soviet-era pavilions."},
    {"name": "Nevsky Prospect, Saint Petersburg",
     "desc": "Nevsky Prospect main avenue in Saint Petersburg, Russia. Elegant 18th–19th century facades. Singer House with glass dome. Trolleybuses, cafe terraces."},
    {"name": "Moscow State University",
     "desc": "Moscow State University on Sparrow Hills, Russia. Imposing Stalinist skyscraper with spire and star. Symmetrical park and fountain. Students walking."},
    {"name": "Novodevichy Convent, Moscow",
     "desc": "Novodevichy Convent in Moscow, Russia. White fortress walls with red elements. Golden bell tower reflected in monastery pond."},
    {"name": "Bronze Horseman, Saint Petersburg",
     "desc": "Bronze Horseman monument to Peter the Great on Senate Square, Saint Petersburg. Equestrian statue on Thunder Stone. Saint Isaac Cathedral dome in background."},
    {"name": "Old Arbat Street, Moscow",
     "desc": "Old Arbat pedestrian street in Moscow, Russia. Historic 19th century buildings with cafes and souvenir shops. Street artists and musicians."},
]

PEOPLE = [
    {"short": "young woman 22",
     "desc": "a young Russian woman about 22 with long straight blonde hair, wearing a white summer dress, white sneakers, small crossbody bag. Smiling warmly at camera."},
    {"short": "man 35 beard",
     "desc": "a Russian man about 35 with short dark brown hair and trimmed beard, wearing charcoal grey henley shirt, dark jeans, brown leather boots."},
    {"short": "couple 28",
     "desc": "a young Russian couple about 28. Woman has auburn wavy hair, sage green dress. Man has sandy blond hair, navy polo. His arm around her waist, both smiling naturally."},
    {"short": "elderly woman 65",
     "desc": "an elegant elderly Russian woman about 65 with styled silver-grey hair, wearing burgundy wool coat with silk scarf. Kind eyes with smile wrinkles."},
    {"short": "teen boy 16",
     "desc": "a Russian teenage boy about 16 with messy light brown hair, wearing black graphic hoodie, cargo pants, chunky white sneakers. Wireless earbuds. Relaxed grin."},
    {"short": "woman 30 candid",
     "desc": "a Russian woman about 30 with chestnut hair in loose bun, beige linen blazer. Standing in the left third of frame, leaning casually against a railing, looking away. Candid moment."},
    {"short": "man 45 bench",
     "desc": "a Russian man about 45 with greying temples, navy windbreaker, hiking pants. Sitting on a bench with legs crossed, holding a Russian newspaper, glancing up with slight smile."},
    {"short": "girl 19 walking",
     "desc": "a Russian girl about 19 with long dark hair and bangs, oversized vintage denim jacket, black leggings, Dr Martens boots. Walking mid-stride, not looking at camera."},
    {"short": "family of four",
     "desc": "a Russian family: father 38 in plaid shirt, mother 36 in light cardigan, daughter 8 in pink jacket, son 5 in dinosaur hoodie. Children running slightly ahead, parents laughing."},
    {"short": "two girlfriends 24",
     "desc": "two Russian girlfriends about 24. One tall with black hair in red coat, other shorter with curly blonde hair in mustard sweater. Sitting on stone steps, one pointing, other laughing."},
    {"short": "grandpa grandson",
     "desc": "a Russian grandfather about 70 with white beard and wool flat cap, warm brown jacket, walking slowly with grandson about 6 in bright blue puffer jacket. Boy pointing excitedly."},
    {"short": "sporty man 28",
     "desc": "a fit Russian man about 28 with buzz cut, black running jacket, compression tights, running shoes. Stopped jogging, stretching against railing, not looking at camera."},
]

TIMES_OF_DAY = [
    {"name": "early_morning", "desc": "Early morning around 7 AM. Soft pink and golden sunrise light. Long gentle shadows. Very few people."},
    {"name": "midday",        "desc": "Bright midday around 1 PM. Strong overhead sunlight. Blue sky with white cumulus clouds."},
    {"name": "golden_hour",   "desc": "Late afternoon around 6 PM, warm low sunlight, long shadows on the ground."},
    {"name": "evening_blue",  "desc": "Evening blue hour around 9 PM. Deep blue twilight sky, buildings lit by warm yellow lights. Street lamps on."},
    {"name": "overcast",      "desc": "Overcast daytime. Soft diffused light, no harsh shadows. Grey sky. Colors muted and moody."},
]

SEASONS = [
    {"name": "winter", "desc": "Russian winter. Fresh snow on ground, rooftops, tree branches. Heavy coats, fur hats. Breath visible."},
    {"name": "spring", "desc": "Russian spring, late April. Fresh green buds on birch trees. Light jackets. Early flowers."},
    {"name": "summer", "desc": "Russian summer, July. Lush deep green trees. People in light summer clothes. Deep blue sky."},
    {"name": "autumn", "desc": "Russian autumn, October. Vibrant golden and crimson foliage. Fallen leaves. Scarves and coats."},
]

REALISM_DETAILS = [
    "Russian-made cars (Lada) nearby, public bus with Russian route number, pigeons on pavement.",
    "Kiosk selling Russian newspapers, people with shopping bags, man walking dog.",
    "Trolleybus wires overhead, souvenir stall with matryoshka dolls, street musician with accordion.",
    "People on phones, mother with stroller, cyclist riding past.",
    "Worn cobblestone pavement, cast-iron bollards, old Russian street lamps.",
    "Yellow marshrutka minibus passing, stray cat on a windowsill, flower vendor with tulips.",
    "Birch trees along the sidewalk, city bench with old man feeding pigeons, overhead power lines.",
]

COMPOSITIONS = [
    "Subject in left third of frame, landmark filling the right side. Rule of thirds.",
    "Subject slightly off-center right, landmark prominent in background.",
    "Wide shot with subject small in the frame, walking toward the landmark.",
    "Subject in center but shot from behind at 3/4 angle, looking up at the landmark.",
    "Classic tourist photo: subject centered smiling at camera with landmark directly behind.",
    "Candid street photography: subject caught mid-motion, landmark sharp in background.",
]


def build_prompt(landmark, person, season, tod, realism, composition):
    prompt = (
        "Documentary photograph, shot on Canon EOS R5 with Canon RF 35mm f/1.8 lens. "
        "ISO 400, f/8, 1/250s. Slightly underexposed -0.3 EV. "
        "Deep focus, everything sharp front to back, no bokeh, no background blur. "
        "Natural sensor noise, subtle luminance grain, slight vignetting at corners. "
        "Lateral chromatic aberration at edges. "
        f"{person['desc']} "
        f"Location: {landmark['desc']} "
        f"{season['desc']} "
        f"{tod['desc']} "
        f"{composition} "
        f"{realism} "
        "Realistic human skin with visible pores, micro-texture. Fabric weave visible on clothing. "
        "No text, no letters, no words, no signs, no writing of any kind anywhere in the image. "
        "No AI smoothing, no beauty filter, no HDR. Not CGI, not illustration, not painting."
    )
    return prompt[:MAX_PROMPT_LEN]


def save_log(combos, log_path):
    data = {
        "metadata": {
            "model": MODEL, "aspect_ratio": ASPECT_RATIO,
            "class": "full_synthetic",
            "total_generated": sum(1 for c in combos if c.get("filename")),
            "total_planned": len(combos),
            "timestamp": datetime.now().isoformat(),
        },
        "images": combos,
    }
    log_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Log saved → %s", log_path)


def parse_args():
    p = argparse.ArgumentParser(description="Generate fully synthetic tourist photos via Seedream 4.5.")
    p.add_argument("--num",    type=int,  default=DEFAULT_N,      help="Number of images to generate")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,  help="Output directory")
    p.add_argument("--seed",   type=int,  default=DEFAULT_SEED,    help="Random seed")
    p.add_argument("--yes",    action="store_true",                help="Skip confirmation prompt")
    return p.parse_args()


def main():
    args = parse_args()
    api_key = get_api_key()
    args.output.mkdir(parents=True, exist_ok=True)
    log_path = args.output / "generation_log.json"

    random.seed(args.seed)

    # Build generation plan with even person distribution
    people_cycle = []
    while len(people_cycle) < args.num:
        batch = PEOPLE[:]
        random.shuffle(batch)
        people_cycle.extend(batch)

    combos = []
    for i in range(args.num):
        landmark    = random.choice(LANDMARKS)
        person      = people_cycle[i]
        season      = random.choice(SEASONS)
        tod         = random.choice(TIMES_OF_DAY)
        realism     = random.choice(REALISM_DETAILS)
        composition = random.choice(COMPOSITIONS)
        prompt      = build_prompt(landmark, person, season, tod, realism, composition)
        combos.append({
            "index":    i,
            "landmark": landmark["name"],
            "person":   person["short"],
            "season":   season["name"],
            "time":     tod["name"],
            "prompt":   prompt,
            "prompt_len": len(prompt),
        })

    log.info("=" * 60)
    log.info("  Seedream 4.5 — %d images  |  output: %s", args.num, args.output)
    log.info("=" * 60)
    for c in combos:
        log.info("  [%3d] %-38s | %-20s | %-7s | %s",
                 c["index"] + 1, c["landmark"], c["person"], c["season"], c["time"])

    # Test first image
    log.info("-" * 60)
    log.info("TEST: Generating first image to verify quality and price...")
    test = combos[0]
    log.info("  Prompt (%d chars): %s...", test["prompt_len"], test["prompt"][:120])

    job_uuid, _ = submit_text2img(test["prompt"], api_key, MODEL, ASPECT_RATIO,
                                   enhance_prompt=ENHANCE_PROMPT)
    if not job_uuid:
        log.error("Cannot submit test job. Exiting.")
        return

    result = poll_result(job_uuid, api_key)
    if result["status"] != "completed":
        log.error("Test job failed: %s", result)
        return

    test_price = result.get("price") or 0
    log.info("  Price per image : %d coins", test_price)
    log.info("  Estimated total : %d coins for %d images", test_price * args.num, args.num)

    urls = result.get("urls") or []
    if isinstance(urls, str):
        urls = [urls]
    if urls:
        fn = f"seedream45_{test['index']:04d}_{test['season']}_{test['time']}.png"
        sz = download_image(urls[0], args.output / fn)
        test.update({"filename": fn, "price": test_price, "job_uuid": job_uuid, "status": "completed"})
        log.info("  Saved: %s  (%d KB)", fn, sz // 1024)

    save_log(combos, log_path)

    if not args.yes:
        answer = input(f"\nContinue generating remaining {len(combos) - 1} images? [y/N] ").strip().lower()
        if answer != "y":
            log.info("Aborted by user.")
            return

    total_cost = test_price
    for combo in combos[1:]:
        idx = combo["index"]
        log.info("[%3d/%d]  %s | %s | %s | %s",
                 idx + 1, args.num, combo["landmark"], combo["person"], combo["season"], combo["time"])

        job_uuid, _ = submit_text2img(combo["prompt"], api_key, MODEL, ASPECT_RATIO,
                                       enhance_prompt=ENHANCE_PROMPT)
        if not job_uuid:
            combo["status"] = "submit_failed"
            continue

        result = poll_result(job_uuid, api_key)
        combo["job_uuid"] = job_uuid

        if result["status"] != "completed":
            combo.update({"status": result["status"], "error": result.get("error", "")})
            log.warning("  FAILED: %s", result.get("error", result["status"]))
            continue

        price = result.get("price") or 0
        total_cost += price
        combo["price"] = price

        urls = result.get("urls") or []
        if isinstance(urls, str):
            urls = [urls]
        if urls:
            fn = f"seedream45_{idx:04d}_{combo['season']}_{combo['time']}.png"
            sz = download_image(urls[0], args.output / fn)
            combo.update({"filename": fn, "status": "completed"})
            log.info("  OK  %s  (%d KB)  cost:%d  total:%d", fn, sz // 1024, price, total_cost)
        else:
            combo["status"] = "no_urls"

        save_log(combos, log_path)
        time.sleep(1)

    success = sum(1 for c in combos if c.get("filename"))
    log.info("=" * 60)
    log.info("  DONE  %d/%d generated  |  total cost: %d coins", success, args.num, total_cost)
    log.info("  Output: %s", args.output)
    log.info("=" * 60)
    save_log(combos, log_path)


if __name__ == "__main__":
    main()
