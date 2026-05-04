#!/usr/bin/env python3
"""
dataset_generation/full_synthetic/gen_imagen4.py
─────────────────────────────────────────────────
Генерация синтетических туристических фотографий через API ruGPT.io.
Модель  : Imagen 4  (Google DeepMind, каскадная архитектура латентной диффузии)
Домен   : Достопримечательности Москвы и Санкт-Петербурга
Выход   : dataset/full_synthetic/imagen4/
Всего сгенерировано: 31 изображение
 
Использование
─────
    export RUGPT_API_KEY="your_key_here"
    python gen_imagen4.py
 
    # Optional overrides:
    python gen_imagen4.py --num 50 --output /path/to/output --seed 42
 
Notes
─────
- Скрипт генерирует одно тестовое изображение и запрашивает подтверждение
  перед запуском полного пакета генерации.
- Прогресс сохраняется  в JSON-лог, что позволяет возобновить работу
  после прерывания.
- API-ключ должен быть предоставлен через переменную окружения RUGPT_API_KEY.
- Промпты намеренно короткие и без технических фототерминов: Imagen 4 при
  перегрузке промпта склонна рисовать буквы и текст прямо на изображении.
- Все изображения, сгенерированные Imagen 4, содержат невидимый водяной знак
  SynthID в соответствии с политикой развёртывания Google DeepMind
  (arXiv:2510.09263).
"""
 
import argparse
import json
import logging
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
MODEL          = "imagen-4.0-generate-001"
ASPECT_RATIO   = "4:3"
ENHANCE_PROMPT = False
DEFAULT_SEED   = 5001
DEFAULT_N      = 31
DEFAULT_OUTPUT = Path(__file__).parent.parent.parent / "dataset" / "full_synthetic" / "imagen4"
 
# ── Промпты ──────────────────────────────────
# Imagen 4 чувствительна к длине и сложности промпта (лимит ≈1000 символов;
# при перегрузке техническими терминами модель рисует буквы на фото). Эти
# 35 SCENES были отобраны эмпирически и дали стабильный результат.
SCENES = [
    "A tourist snapshot of a young woman with dark hair and beige jacket standing at Peterhof Grand Cascade in Saint Petersburg. Golden statues and fountains behind her. Overcast summer day. Photorealistic, no text.",
    "A tourist photo of a man with beard in grey jacket at Red Square in Moscow. Saint Basil Cathedral with colorful domes behind him. Sunny autumn day, fallen leaves. Photorealistic, no text.",
    "A tourist snapshot of an elderly woman in burgundy coat at the Hermitage Museum in Saint Petersburg. Grand baroque facade behind her. Spring day, blue sky. Photorealistic, no text.",
    "A tourist photo of a young couple at Bolshoi Theatre in Moscow. Neoclassical columns and fountain visible behind them. Summer evening, warm lights. Photorealistic, no text.",
    "A tourist snapshot of a teenage boy in black hoodie at VDNKh in Moscow. Golden Friendship of Nations fountain and Soviet pavilions behind him. Summer day. Photorealistic, no text.",
    "A tourist photo of a woman in green dress at Church of the Savior on Spilled Blood in Saint Petersburg. Colorful onion domes reflected in canal. Summer day. Photorealistic, no text.",
    "A tourist snapshot of a grandfather and grandson at Zaryadye Park in Moscow. Floating bridge and Kremlin towers visible behind them. Autumn, golden foliage. Photorealistic, no text.",
    "A tourist photo of a sporty man in running jacket at Nevsky Prospect in Saint Petersburg. Elegant 19th century facades behind him. Overcast morning. Photorealistic, no text.",
    "A tourist snapshot of two girlfriends at Kazan Cathedral in Saint Petersburg. Massive colonnade behind them. Spring day, fresh green trees. Photorealistic, no text.",
    "A tourist photo of a family at Moscow State University on Sparrow Hills. Stalinist tower with spire visible behind them. Autumn day, colorful foliage. Photorealistic, no text.",
    "A tourist snapshot of a woman in red coat at Novodevichy Convent in Moscow. White fortress walls and golden bell tower behind her. Winter, light snow. Photorealistic, no text.",
    "A tourist photo of a young man at Bronze Horseman monument in Saint Petersburg. Equestrian statue and Saint Isaac Cathedral dome behind him. Overcast day. Photorealistic, no text.",
    "A tourist snapshot of a mother and daughter at Old Arbat Street in Moscow. Historic buildings and street musicians visible behind them. Summer evening. Photorealistic, no text.",
    "A tourist photo of an elderly couple at Tsaritsyno Palace in Moscow. Red brick gothic palace and park behind them. Autumn, golden leaves. Photorealistic, no text.",
    "A tourist snapshot of a young woman in yellow jacket at Kolomenskoye park in Moscow. White Church of the Ascension behind her. Spring day, early flowers. Photorealistic, no text.",
    "A tourist photo of a man in navy coat at Lakhta Center in Saint Petersburg. Twisted glass skyscraper behind him. Overcast winter day. Photorealistic, no text.",
    "A tourist snapshot of a teenage girl at GUM department store on Red Square in Moscow. Ornate facade with glass galleries behind her. Winter evening, illuminated windows. Photorealistic, no text.",
    "A tourist photo of a young couple at Palace Bridge in Saint Petersburg. Neva River and Rostral Columns visible behind them. Summer evening, golden light. Photorealistic, no text.",
    "A tourist snapshot of a woman in white dress at Izmailovsky Kremlin in Moscow. Colorful wooden towers behind her. Summer day, bright sunlight. Photorealistic, no text.",
    "A tourist photo of a man with camera at Peterhof in winter. Snow-covered fountains and golden statues behind him. Grey winter sky. Photorealistic, no text.",
    "A tourist snapshot of three students at Red Square in Moscow. Saint Basil Cathedral behind them. Summer midday, tourists in background. Photorealistic, no text.",
    "A tourist photo of a woman in blue coat at Hermitage Museum in winter. Snow on Palace Square, Alexander Column behind her. Cold grey day. Photorealistic, no text.",
    "A tourist snapshot of a young man at VDNKh in Moscow at evening. Golden fountain lit by warm lights behind him. Blue hour sky. Photorealistic, no text.",
    "A tourist photo of a girl in denim jacket walking along Nevsky Prospect. Singer House with glass dome visible behind her. Autumn day. Photorealistic, no text.",
    "A tourist snapshot of an elderly woman at Bolshoi Theatre in Moscow. Grand facade with columns illuminated at night behind her. Evening lights. Photorealistic, no text.",
    "A tourist photo of a couple at Kazan Cathedral in autumn. Golden foliage around the colonnade. Warm afternoon light. Photorealistic, no text.",
    "A tourist snapshot of a young woman at Moscow City skyscrapers. Federation Tower and glass buildings behind her. Sunny summer day. Photorealistic, no text.",
    "A tourist photo of a backpacker at Bronze Horseman in spring. Fresh green trees around Senate Square. Bright spring day. Photorealistic, no text.",
    "A tourist snapshot of a family at Zaryadye Park in summer. Floating bridge and Kremlin in background. Sunny midday, tourists around. Photorealistic, no text.",
    "A tourist photo of a sporty woman at Sparrow Hills lookout in Moscow. Moscow State University tower behind her. Autumn sunset. Photorealistic, no text.",
    "A tourist snapshot of an elderly man at Church of the Savior on Spilled Blood in winter. Snow on colorful domes. Cold grey day. Photorealistic, no text.",
    "A tourist photo of two friends at Arbat Street in Moscow in winter. Snow on historic buildings behind them. Evening, street lamps on. Photorealistic, no text.",
    "A tourist snapshot of a woman in orange scarf at Novodevichy Convent in autumn. Golden foliage reflected in monastery pond. Warm afternoon light. Photorealistic, no text.",
    "A tourist photo of a teenage boy at Peterhof in spring. Fresh green park and fountains behind him. Sunny spring morning. Photorealistic, no text.",
    "A tourist snapshot of a couple at Red Square in Moscow in winter evening. Saint Basil Cathedral lit by lights, snow on cobblestones. Festive atmosphere. Photorealistic, no text.",
]
 
 
def save_log(combos: list, log_path: Path) -> None:
    """Write generation progress to a JSON log file."""
    data = {
        "metadata": {
            "model": MODEL,
            "aspect_ratio": ASPECT_RATIO,
            "class": "full_synthetic",
            "note": ("All images carry an invisible SynthID watermark "
                     "(Google DeepMind, arXiv:2510.09263)."),
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
    p = argparse.ArgumentParser(description="Generate fully synthetic tourist photos via Imagen 4.")
    p.add_argument("--num",    type=int,  default=DEFAULT_N,      help="Number of images to generate")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,  help="Output directory")
    p.add_argument("--seed",   type=int,  default=DEFAULT_SEED,    help="Random seed (для совместимости)")
    p.add_argument("--yes",    action="store_true",                help="Skip confirmation prompt")
    return p.parse_args()
 
 
def main():
    args = parse_args()
    api_key = get_api_key()
    args.output.mkdir(parents=True, exist_ok=True)
    log_path = args.output / "generation_log.json"
 
    # Build generation plan: циклически перебираем готовые SCENES.
    # Это гарантирует тот же набор сюжетов, что использовался при сборке датасета.
    combos = []
    for i in range(args.num):
        prompt = SCENES[i % len(SCENES)]
        combos.append({
            "index":      i,
            "prompt":     prompt,
            "prompt_len": len(prompt),
        })
 
    log.info("=" * 60)
    log.info("  Imagen 4 — %d images  |  output: %s", args.num, args.output)
    log.info("=" * 60)
    for c in combos:
        log.info("  [%3d] %s...", c["index"] + 1, c["prompt"][:70])
 
    # ── Test first image ──
    log.info("-" * 60)
    log.info("TEST: Generating first image to verify quality and price...")
    test = combos[0]
    log.info("  Prompt (%d chars): %s...", test["prompt_len"], test["prompt"][:100])
    job_uuid, _ = submit_text2img(
        test["prompt"], api_key, model=MODEL,
        aspect_ratio=ASPECT_RATIO,
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
        fn = f"imagen4_{test['index']:03d}.png"
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
        log.info("[%3d/%d]  %s...", idx + 1, args.num, combo["prompt"][:70])
 
        job_uuid, _ = submit_text2img(
            combo["prompt"], api_key, model=MODEL,
            aspect_ratio=ASPECT_RATIO,
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
            fn = f"imagen4_{idx:03d}.png"
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
