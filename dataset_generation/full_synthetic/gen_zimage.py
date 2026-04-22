#!/usr/bin/env python3
"""
dataset_generation/full_synthetic/gen_zimage.py
────────────────────────────────────────────────
Генерация полностью синтетических туристических фотографий через API ruGPT.io.
Модель  : z-image  (Alibaba Tongyi MAI, S³-DiT, 6B параметров, arXiv:2511.22699)
Домен   : Достопримечательности Москвы и Санкт-Петербурга, разнообразные российские пешеходы
Выход   : dataset/full_synthetic/zimage/
Всего сгенерировано для датасета ВКР: 122 изображения

Примечание о длине промпта
─────────────────────
Z-Image поддерживает промпты длиной до 1 000 символов.
Сборщик промптов в этом скрипте безопасно ограничивает длину на уровне 950 символов.

Использование
─────
    export RUGPT_API_KEY="your_key_here"
    python gen_zimage.py

    python gen_zimage.py --num 50 --output /path/to/output --seed 42
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

MODEL          = "z-image"
ASPECT_RATIO   = "4:3"
ENHANCE_PROMPT = False
DEFAULT_SEED   = 3012
DEFAULT_N      = 122
DEFAULT_OUTPUT = Path(__file__).parent.parent.parent / "dataset" / "full_synthetic" / "zimage"
MAX_PROMPT_LEN = 950

# Compact landmark descriptions to stay within the 1 000-char prompt limit
LANDMARKS = [
    {"name": "Red Square, Moscow",                       "short": "Red Square, Saint Basil Cathedral, Kremlin wall, Moscow"},
    {"name": "Hermitage, Saint Petersburg",              "short": "Hermitage Museum, Winter Palace, Palace Square, Saint Petersburg"},
    {"name": "Moscow City skyscrapers",                  "short": "Moscow City skyscrapers, Federation Tower, Moskva River"},
    {"name": "Church of Savior on Spilled Blood, SPb",   "short": "Church of Savior on Spilled Blood, Griboedov Canal, Saint Petersburg"},
    {"name": "Bolshoi Theatre, Moscow",                  "short": "Bolshoi Theatre, neoclassical columns, Theatre Square, Moscow"},
    {"name": "Zaryadye Park, Moscow",                    "short": "Zaryadye Park floating bridge, Kremlin towers, Moscow"},
    {"name": "Kazan Cathedral, Saint Petersburg",        "short": "Kazan Cathedral colonnade, Nevsky Prospect, Saint Petersburg"},
    {"name": "Peterhof Palace, near Saint Petersburg",   "short": "Peterhof Grand Cascade, golden statues, fountains, Saint Petersburg"},
    {"name": "Tsaritsyno Palace, Moscow",                "short": "Tsaritsyno Palace, red brick gothic, park, Moscow"},
    {"name": "VDNKh, Moscow",                            "short": "VDNKh Friendship of Nations fountain, Soviet pavilions, Moscow"},
    {"name": "Nevsky Prospect, Saint Petersburg",        "short": "Nevsky Prospect avenue, 19th century facades, Saint Petersburg"},
    {"name": "Moscow State University",                  "short": "Moscow State University Stalinist tower, Sparrow Hills"},
    {"name": "Novodevichy Convent, Moscow",              "short": "Novodevichy Convent, white fortress walls, Moscow"},
    {"name": "Bronze Horseman, Saint Petersburg",        "short": "Bronze Horseman monument, Senate Square, Saint Petersburg"},
    {"name": "Old Arbat Street, Moscow",                 "short": "Old Arbat pedestrian street, historic buildings, Moscow"},
]

PEOPLE = [
    {"short": "young woman 22",   "desc": "young Russian woman 22, blonde, white floral dress, sneakers, crossbody bag, smiling"},
    {"short": "man 35 beard",     "desc": "Russian man 35, dark hair, trimmed beard, grey henley, dark jeans, brown boots"},
    {"short": "couple 28",        "desc": "Russian couple 28, woman auburn hair sage dress, man navy polo, arm around her waist"},
    {"short": "elderly woman 65", "desc": "elegant elderly Russian woman 65, silver hair, burgundy wool coat, silk scarf"},
    {"short": "teen boy 16",      "desc": "Russian teen boy 16, messy brown hair, black hoodie, cargo pants, wireless earbuds"},
    {"short": "woman 30 candid",  "desc": "Russian woman 30, bun, beige blazer, leaning on railing, looking away, candid"},
    {"short": "man 45 bench",     "desc": "Russian man 45, grey temples, navy windbreaker, sitting on bench, holding Russian newspaper"},
    {"short": "girl 19 walking",  "desc": "Russian girl 19, dark hair bangs, denim jacket, walking mid-stride, not looking at camera"},
    {"short": "family of four",   "desc": "Russian family: father 38, mother 36, daughter 8, son 5, children running ahead, parents laughing"},
    {"short": "two girlfriends",  "desc": "two Russian girlfriends 24, sitting on steps, one pointing at landmark, other laughing"},
    {"short": "grandpa grandson", "desc": "Russian grandfather 70, white beard, flat cap, walking with grandson 6, boy pointing excitedly"},
    {"short": "sporty man 28",    "desc": "fit Russian man 28, buzz cut, black running jacket, stretching against railing, not looking at camera"},
]

TIMES_OF_DAY = [
    {"name": "early_morning", "desc": "early morning 7 AM, soft golden sunrise light, long shadows, few people"},
    {"name": "midday",        "desc": "bright midday 1 PM, strong sunlight, blue sky with white clouds"},
    {"name": "golden_hour",   "desc": "late afternoon 6 PM, warm low sunlight, long shadows"},
    {"name": "evening_blue",  "desc": "evening blue hour 9 PM, deep blue sky, warm building lights, street lamps on"},
    {"name": "overcast",      "desc": "overcast day, soft diffused light, grey sky, muted moody colors"},
]

SEASONS = [
    {"name": "winter", "desc": "Russian winter, snow on ground and rooftops, heavy coats, fur hats, breath visible"},
    {"name": "spring", "desc": "Russian spring April, fresh green buds, light jackets, early flowers"},
    {"name": "summer", "desc": "Russian summer July, lush green trees, people in light clothes, warm atmosphere"},
    {"name": "autumn", "desc": "Russian autumn October, golden orange foliage, fallen leaves, scarves and coats"},
]


def build_prompt(landmark, person, season, tod):
    prompt = (
        "Real documentary travel photo, 35mm lens, f/8. "
        "Deep focus, everything sharp front to back, no bokeh, no background blur. "
        f"{person['desc']} at {landmark['short']}. "
        f"{season['desc']}. {tod['desc']}. "
        "Natural skin texture, subtle grain, slight vignetting. "
        "No text, no letters, no words, no signs anywhere in the image. "
        "No retouching, no HDR, not illustration, not CGI."
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
    p = argparse.ArgumentParser(description="Generate fully synthetic tourist photos via Z-Image.")
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

    people_cycle = []
    while len(people_cycle) < args.num:
        batch = PEOPLE[:]
        random.shuffle(batch)
        people_cycle.extend(batch)

    combos = []
    for i in range(args.num):
        lm     = random.choice(LANDMARKS)
        person = people_cycle[i]
        season = random.choice(SEASONS)
        tod    = random.choice(TIMES_OF_DAY)
        prompt = build_prompt(lm, person, season, tod)
        combos.append({
            "index": i, "landmark": lm["name"], "person": person["short"],
            "season": season["name"], "time": tod["name"],
            "prompt": prompt, "prompt_len": len(prompt),
        })

    log.info("=" * 60)
    log.info("  Z-Image — %d images  |  output: %s", args.num, args.output)
    log.info("=" * 60)

    # Test first image
    test = combos[0]
    job_uuid, _ = submit_text2img(test["prompt"], api_key, MODEL, ASPECT_RATIO,
                                   enhance_prompt=ENHANCE_PROMPT)
    if not job_uuid:
        log.error("Cannot submit. Exiting.")
        return
    result = poll_result(job_uuid, api_key)
    if result["status"] != "completed":
        log.error("Test failed: %s", result)
        return
    test_price = result.get("price") or 0
    log.info("  Price: %d coins | Estimated total: %d coins", test_price, test_price * args.num)

    urls = result.get("urls") or []
    if isinstance(urls, str):
        urls = [urls]
    if urls:
        fn = f"zimage_{test['index']:04d}_{test['season']}_{test['time']}.png"
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
            combo["status"] = "submit_failed"; continue
        result = poll_result(job_uuid, api_key)
        combo["job_uuid"] = job_uuid
        if result["status"] != "completed":
            combo.update({"status": result["status"], "error": result.get("error", "")})
            log.warning("  FAILED: %s", result.get("error", result["status"])); continue
        price = result.get("price") or 0
        total_cost += price
        combo["price"] = price
        urls = result.get("urls") or []
        if isinstance(urls, str):
            urls = [urls]
        if urls:
            fn = f"zimage_{idx:04d}_{combo['season']}_{combo['time']}.png"
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
