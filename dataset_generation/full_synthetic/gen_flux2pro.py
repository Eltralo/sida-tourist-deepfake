#!/usr/bin/env python3
"""
dataset_generation/full_synthetic/gen_flux2pro.py
──────────────────────────────────────────────────
Генерация полностью синтетических туристических фотографий через API ruGPT.io.
Модель  : flux-2/pro  (Black Forest Labs, диффузионная трансформерная архитектура, 32B параметров)
Домен   : Достопримечательности Москвы и Санкт-Петербурга, различные люди
Выход   : dataset/full_synthetic/flux2pro/
Всего сгенерировано для датасета ВКР: 152 изображения

Использование
─────
    export RUGPT_API_KEY="your_key_here"
    python gen_flux2pro.py

    # Optional overrides:
    python gen_flux2pro.py --num 50 --output /path/to/output --seed 42

Notes
─────
- Скрипт сначала генерирует одно тестовое изображение и запрашивает подтверждение
  перед запуском полного пакета генерации.
- Прогресс сохраняется инкрементально в JSON-лог, что позволяет возобновить работу
  после прерывания.
- API-ключ должен быть предоставлен через переменную окружения RUGPT_API_KEY.
"""

import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path

# Allow imports from the parent package (dataset_generation/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _api_client import get_api_key, submit_text2img, poll_result, download_image

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Generation defaults ───────────────────────────────────────────────────────
MODEL         = "flux-2/pro"
ASPECT_RATIO  = "3:2"
RESOLUTION    = "2K"
ENHANCE_PROMPT = False
DEFAULT_SEED   = 2025
DEFAULT_N      = 152
DEFAULT_OUTPUT = Path(__file__).parent.parent.parent / "dataset" / "full_synthetic" / "flux2pro"

# ── Prompt components ─────────────────────────────────────────────────────────
LANDMARKS = [
    {"name": "Red Square, Moscow",
     "desc": ("Red Square (Krasnaya Ploshchad) in Moscow, Russia. "
               "Saint Basil Cathedral with iconic colorful onion domes. "
               "Red Kremlin wall. Cobblestone pavement. "
               "All signs and banners in Russian Cyrillic. Russian flags on buildings.")},
    {"name": "Hermitage, Saint Petersburg",
     "desc": ("State Hermitage Museum and Winter Palace in Saint Petersburg, Russia. "
               "Grand baroque facade in pale green and white with gold ornaments. "
               "Palace Square with Alexander Column visible. All signage in Russian Cyrillic.")},
    {"name": "Moscow City skyscrapers",
     "desc": ("Moscow International Business Center in Moscow, Russia. "
               "Modern glass skyscrapers including Federation Tower. "
               "View from Moskva River embankment. "
               "Russian-language billboards, road signs in Cyrillic, Russian car plates.")},
    {"name": "Church of Savior on Spilled Blood, Saint Petersburg",
     "desc": ("Church of the Savior on Spilled Blood in Saint Petersburg, Russia. "
               "Elaborate multicolored onion domes with mosaics. "
               "Griboedov Canal in foreground. Russian Orthodox crosses. "
               "Street signs in Russian Cyrillic.")},
    {"name": "Bolshoi Theatre, Moscow",
     "desc": ("Bolshoi Theatre in Moscow, Russia. "
               "Grand neoclassical facade with eight Ionic columns. "
               "Bronze quadriga of Apollo on pediment. "
               "Theatre Square with fountain. Posters in Russian.")},
    {"name": "Zaryadye Park, Moscow",
     "desc": ("Zaryadye Park in Moscow, Russia. "
               "Famous floating bridge extending over Moskva River. "
               "Kremlin towers in background. "
               "Modern landscape architecture with wild Russian plants. "
               "All signs in Russian Cyrillic.")},
    {"name": "Kazan Cathedral, Saint Petersburg",
     "desc": ("Kazan Cathedral on Nevsky Prospect in Saint Petersburg, Russia. "
               "Massive semicircular colonnade of 96 columns. "
               "Fountain in front park. Russian tourists on steps.")},
    {"name": "Peterhof Palace, near Saint Petersburg",
     "desc": ("Peterhof Palace near Saint Petersburg, Russia. "
               "Grand Cascade with golden statues and powerful fountains. "
               "Grand Palace on the hill. Canal leading to Gulf of Finland.")},
    {"name": "Tsaritsyno Palace, Moscow",
     "desc": ("Tsaritsyno Museum-Reserve in Moscow, Russia. "
               "Ornate red brick pseudo-gothic palace. "
               "Beautiful park with ponds and bridges. Signs in Russian Cyrillic.")},
    {"name": "VDNKh, Moscow",
     "desc": ("VDNKh exhibition center in Moscow, Russia. "
               "Golden Friendship of Nations fountain with 16 gilded figures. "
               "Grand Soviet-era pavilions. Vostok rocket monument in distance. "
               "All pavilion names in Russian Cyrillic.")},
    {"name": "Nevsky Prospect, Saint Petersburg",
     "desc": ("Nevsky Prospect main avenue in Saint Petersburg, Russia. "
               "Elegant 18th–19th century facades. Singer House with glass dome. "
               "Trolleybuses, Russian shop signs, cafe terraces.")},
    {"name": "Moscow State University",
     "desc": ("Moscow State University main building on Sparrow Hills, Russia. "
               "Imposing Stalinist skyscraper with spire and star. "
               "Symmetrical park and fountain in front. Students walking. "
               "All signs in Russian.")},
    {"name": "Novodevichy Convent, Moscow",
     "desc": ("Novodevichy Convent in Moscow, Russia. "
               "White fortress walls with red elements. "
               "Golden bell tower reflected in monastery pond. UNESCO World Heritage Site.")},
    {"name": "Bronze Horseman, Saint Petersburg",
     "desc": ("Bronze Horseman monument to Peter the Great on Senate Square, Saint Petersburg, Russia. "
               "Dramatic equestrian statue on Thunder Stone. "
               "Saint Isaac Cathedral dome in background.")},
    {"name": "Old Arbat Street, Moscow",
     "desc": ("Old Arbat pedestrian street in Moscow, Russia. "
               "Historic 19th century buildings with cafes and souvenir shops. "
               "Street artists and musicians. Okudzhava monument. "
               "All shop signs in Russian Cyrillic.")},
    {"name": "GUM, Moscow",
     "desc": ("GUM department store on Red Square in Moscow, Russia. "
               "Ornate 19th century facade with glass-roofed galleries. "
               "Kremlin wall opposite. All text in Russian.")},
    {"name": "Kolomenskoye Park, Moscow",
     "desc": ("Kolomenskoye Museum-Reserve in Moscow, Russia. "
               "White Church of the Ascension (UNESCO). "
               "Apple orchards and Moskva River view. Signs in Russian.")},
    {"name": "Lakhta Center, Saint Petersburg",
     "desc": ("Lakhta Center supertall skyscraper in Saint Petersburg, Russia. "
               "Tallest building in Europe, twisted glass tower. "
               "View from Gulf of Finland embankment. Signs in Cyrillic.")},
    {"name": "Palace Bridge, Saint Petersburg",
     "desc": ("Palace Bridge over Neva River in Saint Petersburg, Russia. "
               "Vasilievsky Island Strelka with Rostral Columns. "
               "Hermitage visible on right bank.")},
    {"name": "Izmailovsky Kremlin, Moscow",
     "desc": ("Izmailovsky Kremlin in Moscow, Russia. "
               "Colorful fairytale wooden towers. "
               "Russian souvenir market with matryoshka dolls. Signs in Cyrillic.")},
]

PEOPLE = [
    {"short": "young woman 22",
     "desc": ("a young Russian woman about 22 with long straight blonde hair, "
               "wearing a white summer dress with floral print, white sneakers, small crossbody bag. "
               "Smiling warmly at camera, standing slightly turned to the side.")},
    {"short": "man 35 beard",
     "desc": ("a Russian man about 35 with short dark brown hair and trimmed beard, "
               "wearing charcoal grey henley shirt with rolled sleeves, dark jeans, brown leather boots. "
               "Standing confidently with hand in pocket, calm smile.")},
    {"short": "couple 28",
     "desc": ("a young Russian couple about 28. "
               "Woman has auburn wavy hair, sage green linen dress. "
               "Man has sandy blond hair, navy polo and khaki chinos. "
               "His arm around her waist, both smiling naturally.")},
    {"short": "elderly woman 65",
     "desc": ("an elegant elderly Russian woman about 65 with styled silver-grey hair, "
               "wearing burgundy wool coat with silk scarf, leather shoes. "
               "Kind eyes with smile wrinkles, gentle dignified expression.")},
    {"short": "teen boy 16",
     "desc": ("a Russian teenage boy about 16 with messy light brown hair, "
               "wearing black graphic hoodie, cargo pants, chunky white sneakers. "
               "Wireless earbuds. Casual relaxed grin.")},
    {"short": "woman 30 candid",
     "desc": ("a Russian woman about 30 with chestnut hair in loose bun, beige linen blazer. "
               "Standing in the left third of frame, leaning casually against a railing, "
               "looking away from camera. Candid unposed moment.")},
    {"short": "man 45 bench",
     "desc": ("a Russian man about 45 with greying temples, navy windbreaker, hiking pants. "
               "Sitting on a bench with legs crossed, holding a Russian newspaper, "
               "glancing up with slight smile.")},
    {"short": "girl 19 walking",
     "desc": ("a Russian girl about 19 with long dark hair and bangs, "
               "oversized vintage denim jacket, black leggings, Dr Martens boots. "
               "Walking through the frame mid-stride, not looking at camera.")},
    {"short": "family of four",
     "desc": ("a Russian family: father 38 in plaid shirt, mother 36 in light cardigan, "
               "daughter 8 in pink jacket, son 5 in dinosaur hoodie. "
               "Children running slightly ahead, parents laughing behind.")},
    {"short": "man 25 photographer",
     "desc": ("a young Russian man about 25 with short curly dark hair and stubble, "
               "olive field jacket and black jeans, canvas backpack. "
               "Crouching in right of frame, taking a photo with his camera, back partially to viewer.")},
    {"short": "two girlfriends 24",
     "desc": ("two Russian girlfriends about 24. "
               "One tall with black hair in red coat, other shorter with curly blonde hair in mustard sweater. "
               "Sitting on stone steps, one pointing at landmark, other laughing head tilted back.")},
    {"short": "grandpa grandson",
     "desc": ("a Russian grandfather about 70 with white beard and wool flat cap, warm brown jacket, "
               "walking slowly with grandson about 6 in bright blue puffer jacket. "
               "Boy holding grandpa's hand, pointing excitedly at the landmark.")},
    {"short": "woman 40 phone",
     "desc": ("a Russian woman about 40 with shoulder-length auburn hair and glasses, "
               "tailored dark blue coat, silk scarf. "
               "Slightly off-center, checking her phone, other hand holding takeaway coffee.")},
    {"short": "sporty man 28",
     "desc": ("a fit Russian man about 28 with buzz cut, black running jacket, "
               "compression tights, running shoes, wireless earbuds. "
               "Stopped jogging, stretching against railing, not looking at camera.")},
    {"short": "artist woman 50",
     "desc": ("a creative Russian woman about 50 with short silver pixie cut and red lipstick, "
               "long olive trench coat, colorful hand-knitted scarf. "
               "Sitting on folding stool, sketching landmark in notebook.")},
    {"short": "three students",
     "desc": ("three Russian college students around 20: "
               "guy with glasses in university hoodie, girl with braids carrying textbooks, "
               "another guy in bomber jacket. Sitting on low wall, chatting and laughing.")},
    {"short": "backpacker 21",
     "desc": ("a young Russian backpacker about 21 with shaggy blonde hair under beanie, "
               "faded flannel shirt, ripped jeans, worn boots. "
               "Large backpack on ground. Sitting cross-legged, eating pirozhok from paper bag.")},
    {"short": "mother daughter",
     "desc": ("a Russian mother about 35 with long braid, denim jacket, "
               "with daughter about 10 in yellow raincoat. "
               "Walking hand in hand, daughter skipping, both looking toward the landmark.")},
]

TIMES_OF_DAY = [
    {"name": "early_morning",
     "desc": "Early morning around 7 AM. Soft pink and golden sunrise light. Long gentle shadows. Very few people."},
    {"name": "midday",
     "desc": "Bright midday around 1 PM. Strong overhead sunlight. Blue sky with white cumulus clouds. Tourists in background."},
    {"name": "golden_hour",
     "desc": "Late afternoon around 6 PM, natural warm sunlight from low angle, long shadows on the ground."},
    {"name": "evening_blue",
     "desc": "Evening blue hour after sunset around 9 PM. Deep blue twilight sky, buildings lit by warm yellow lights. Street lamps on."},
    {"name": "overcast",
     "desc": "Overcast daytime. Soft diffused light, no harsh shadows. Grey sky. Colors muted and moody. Typical Russian weather."},
]

SEASONS = [
    {"name": "winter",
     "desc": "Russian winter. Fresh snow on ground, rooftops, tree branches. Heavy coats, fur hats, scarves. Breath visible."},
    {"name": "spring",
     "desc": "Russian spring, late April. Fresh green buds on birch trees. Light jackets. Early flowers."},
    {"name": "summer",
     "desc": "Russian summer, July. Lush deep green trees. People in light summer clothes. Deep blue sky, warm atmosphere."},
    {"name": "autumn",
     "desc": "Russian autumn, October. Vibrant golden, orange and crimson foliage. Fallen leaves on pavement. Scarves and coats."},
]

REALISM_DETAILS = [
    "Russian-made cars (Lada) nearby, public bus with Russian route number, pigeons on pavement.",
    "Kiosk selling Russian newspapers, people with shopping bags, man walking dog.",
    "Trolleybus wires overhead, souvenir stall with matryoshka dolls, street musician with accordion.",
    "People on phones, mother with stroller, cyclist riding past, leaves swirling in breeze.",
    "Worn cobblestone pavement, cast-iron bollards, old Russian street lamps, weathered building plaster.",
    "Yellow marshrutka minibus passing, stray cat on a windowsill, flower vendor with tulips.",
    "Birch trees along the sidewalk, city bench with old man feeding pigeons, overhead power lines.",
]

COMPOSITIONS = [
    "Subject in left third of frame using rule of thirds, landmark filling the right side.",
    "Subject slightly off-center right, landmark prominent in the left and center background.",
    "Wide shot with subject small in the frame, walking toward the landmark. Environmental portrait.",
    "Subject in center but shot from behind at 3/4 angle, looking up at the landmark.",
    "Classic tourist photo: subject centered smiling at camera with landmark directly behind.",
    "Candid street photography: subject caught mid-motion, landmark sharp in background.",
    "Subject in lower-left corner, shot from slightly above, landmark towering in background.",
]


def build_prompt(landmark: dict, person: dict, season: dict, tod: dict,
                 realism: str, composition: str) -> str:
    """Assemble the full generation prompt from individual components."""
    return (
        "Documentary photograph, shot on Canon EOS R5 with Canon RF 35mm f/1.8 IS STM lens. "
        "ISO 400, f/5.6, 1/250s. Slightly underexposed by -0.3 EV. "
        "Natural sensor noise, subtle luminance grain, slight vignetting at corners. "
        "Authentic JPEG straight from camera with standard picture profile. "
        "Slight barrel distortion from wide lens, lateral chromatic aberration at edges. "
        f"{person['desc']} "
        f"Location: {landmark['desc']} "
        f"{season['desc']} "
        f"{tod['desc']} "
        f"{composition} "
        f"{realism} "
        "Realistic human skin with visible pores, fine hairs, micro-texture. "
        "Fabric weave texture visible on clothing. Worn surfaces on pavement and walls. "
        "Authentic reflections and refractions. Accurate shadow casting on skin. "
        "All visible text, signage, labels in Russian Cyrillic script only. "
        "Subtle motion blur on background figures. "
        "Real atmospheric haze if backlit. No HDR tone mapping. "
        "No AI smoothing, no beauty filter. "
        "Uneven natural lighting, no studio fill light. "
        "Not CGI, not 3D render, not stock photo pose, not illustration, not painting."
    )


def save_log(combos: list, log_path: Path) -> None:
    """Write generation progress to a JSON log file."""
    data = {
        "metadata": {
            "model": MODEL,
            "resolution": RESOLUTION,
            "aspect_ratio": ASPECT_RATIO,
            "class": "full_synthetic",
            "total_generated": sum(1 for c in combos if c.get("filename")),
            "total_planned": len(combos),
            "timestamp": datetime.now().isoformat(),
        },
        "images": combos,
    }
    log_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Log saved → %s", log_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate fully synthetic tourist photos via Flux 2 Pro.")
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

    # Build generation plan
    combos = []
    for i in range(args.num):
        landmark    = random.choice(LANDMARKS)
        person      = random.choice(PEOPLE)
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
        })

    log.info("=" * 60)
    log.info("  Flux 2 Pro — %d images  |  output: %s", args.num, args.output)
    log.info("=" * 60)
    for c in combos:
        log.info("  [%3d] %-40s | %-20s | %-7s | %s",
                 c["index"] + 1, c["landmark"], c["person"], c["season"], c["time"])

    # ── Test first image ──
    log.info("-" * 60)
    log.info("TEST: Generating first image to verify quality and price...")
    test = combos[0]
    job_uuid, _ = submit_text2img(
        test["prompt"], api_key, model=MODEL,
        aspect_ratio=ASPECT_RATIO, resolution=RESOLUTION,
        enhance_prompt=ENHANCE_PROMPT,
    )
    if not job_uuid:
        log.error("Cannot submit test job. Exiting.")
        return

    result = poll_result(job_uuid, api_key)
    if result["status"] != "completed":
        log.error("Test job failed: %s", result)
        return

    test_price = result.get("price", 0) or 0
    log.info("  Price per image : %d coins", test_price)
    log.info("  Estimated total : %d coins for %d images", test_price * args.num, args.num)

    urls = result.get("urls") or []
    if isinstance(urls, str):
        urls = [urls]
    if urls:
        fn = f"flux2pro_{test['index']:04d}_{test['season']}_{test['time']}.png"
        sz = download_image(urls[0], args.output / fn)
        test.update({"filename": fn, "price": test_price, "job_uuid": job_uuid, "status": "completed"})
        log.info("  Saved: %s  (%d KB)", fn, sz // 1024)

    save_log(combos, log_path)

    if not args.yes:
        answer = input(f"\nContinue generating remaining {len(combos) - 1} images? [y/N] ").strip().lower()
        if answer != "y":
            log.info("Aborted by user.")
            return

    # ── Batch generation ──
    total_cost = test_price
    for combo in combos[1:]:
        idx = combo["index"]
        log.info("[%3d/%d]  %s | %s | %s | %s",
                 idx + 1, args.num, combo["landmark"], combo["person"], combo["season"], combo["time"])

        job_uuid, _ = submit_text2img(
            combo["prompt"], api_key, model=MODEL,
            aspect_ratio=ASPECT_RATIO, resolution=RESOLUTION,
            enhance_prompt=ENHANCE_PROMPT,
        )
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
            fn = f"flux2pro_{idx:04d}_{combo['season']}_{combo['time']}.png"
            sz = download_image(urls[0], args.output / fn)
            combo.update({"filename": fn, "status": "completed"})
            log.info("  OK  %s  (%d KB)  | cost: %d | total: %d", fn, sz // 1024, price, total_cost)
        else:
            combo["status"] = "no_urls"

        save_log(combos, log_path)
        time.sleep(1)

    success = sum(1 for c in combos if c.get("filename"))
    log.info("=" * 60)
    log.info("  DONE  %d/%d images generated", success, args.num)
    log.info("  Total cost: %d coins", total_cost)
    log.info("  Output: %s", args.output)
    log.info("=" * 60)
    save_log(combos, log_path)


if __name__ == "__main__":
    main()
