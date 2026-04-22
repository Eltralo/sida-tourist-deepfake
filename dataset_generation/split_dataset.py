#!/usr/bin/env python3
"""
dataset_generation/split_dataset.py
─────────────────────────────────────
Разбивка собранного датасета на обучающую и тестовую выборки
со стратификацией по классу и источнику генерации.

Входная структура (--source-dir):
    dataset/
        real/           *.jpg / *.png          (438 изображений в диссертации)
        full_synthetic/ flux2pro/  *.png        (152)
                        seedream45/ *.png       (123)
                        zimage/     *.png       (122)
                        imagen4/    *.png       ( 31)
        tempered/       nano_banana/ *.png      (241)
                        flux2pro_edit/ *.png    (192)

Выходная структура (--output-dir):
    photo/
        train/
            real/          338 изображений
            full_synt/     329 изображений
            tempered/      333 изображений
        test/
            real/          100 изображений
            full_synt/     100 изображений
            tempered/      100 изображений

Соотношение: 999 обучающих / 300 тестовых  (seed=42)

Использование
─────────────
    python split_dataset.py
    python split_dataset.py --source-dir /path/to/dataset --output-dir /path/to/photo --seed 42
"""

import argparse
import random
import shutil
import json
from pathlib import Path


# ── Параметры по умолчанию ────────────────────────────────────────────────────
DEFAULT_SOURCE = Path(__file__).parent.parent / "dataset"
DEFAULT_OUTPUT = Path(__file__).parent.parent / "photo"
DEFAULT_SEED   = 42

# Количество тестовых образцов по классам согласно диссертации
TEST_COUNT = {"real": 100, "full_synt": 100, "tempered": 100}

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def collect_images(source_dir: Path) -> dict[str, list[Path]]:
    """
    Сбор путей к изображениям по классам из исходной директории.

    Параметры
    ─────────
    source_dir : Path
        Корневая директория с исходными данными, содержащая поддиректории
        real/, full_synthetic/, tempered/.

    Возвращает
    ──────────
    dict[str, list[Path]]
        Словарь, где ключ — имя класса ("real", "full_synt", "tempered"),
        значение — отсортированный список путей к файлам изображений.

    Примечание
    ──────────
    Функция рекурсивно обходит поддиректории каждого класса, собирая все
    файлы с расширениями изображений. Это позволяет сохранить информацию
    о подклассах (модели генерации) для дальнейшего анализа.
    """
    classes = {}

    # ── real (реальные фотографии) ────────────────────────────────────────────
    real_dir = source_dir / "real"
    if real_dir.exists():
        classes["real"] = sorted(
            f for f in real_dir.rglob("*") if f.suffix.lower() in IMG_EXTENSIONS
        )
    else:
        print(f"  ПРЕДУПРЕЖДЕНИЕ: {real_dir} не найдена — пропуск класса 'real'.")
        classes["real"] = []

    # ── full_synthetic (полностью синтетические) ──────────────────────────────
    fs_dir = source_dir / "full_synthetic"
    if fs_dir.exists():
        classes["full_synt"] = sorted(
            f for f in fs_dir.rglob("*") if f.suffix.lower() in IMG_EXTENSIONS
        )
    else:
        print(f"  ПРЕДУПРЕЖДЕНИЕ: {fs_dir} не найдена — пропуск класса 'full_synt'.")
        classes["full_synt"] = []

    # ── tempered (подвергнутые манипуляциям) ──────────────────────────────────
    t_dir = source_dir / "tempered"
    if t_dir.exists():
        classes["tempered"] = sorted(
            f for f in t_dir.rglob("*") if f.suffix.lower() in IMG_EXTENSIONS
        )
    else:
        print(f"  ПРЕДУПРЕЖДЕНИЕ: {t_dir} не найдена — пропуск класса 'tempered'.")
        classes["tempered"] = []

    return classes


def split_and_copy(classes: dict, output_dir: Path, seed: int) -> dict:
    """
    Выполнение стратифицированного разбиения и копирование файлов в train/test.

    Параметры
    ─────────
    classes : dict[str, list[Path]]
        Словарь с путями к изображениям по классам (результат collect_images).
    output_dir : Path
        Выходная директория, в которой будут созданы поддиректории train/ и test/.
    seed : int
        Seed для генератора случайных чисел (обеспечивает воспроизводимость разбиения).

    Возвращает
    ──────────
    dict
        Отчёт о разбиении: {класс: {"train": количество, "test": количество}}.

    Описание алгоритма
    ──────────────────
    1. Для каждого класса перемешивает список файлов с фиксированным seed
    2. Выделяет TEST_COUNT[класс] образцов в тестовую выборку
    3. Остальные образцы отправляет в обучающую выборку
    4. Копирует файлы в соответствующие поддиректории output_dir/train/класс
       и output_dir/test/класс
    5. Сохраняет исходные имена файлов для трассируемости
    """
    rng = random.Random(seed)
    report = {}

    for cls_name, all_files in classes.items():
        if not all_files:
            print(f"  ПРЕДУПРЕЖДЕНИЕ: Нет файлов для класса '{cls_name}' — пропуск.")
            continue

        shuffled = all_files[:]
        rng.shuffle(shuffled)

        n_test  = TEST_COUNT.get(cls_name, 100)
        n_train = len(shuffled) - n_test

        if n_train < 1:
            print(f"  ПРЕДУПРЕЖДЕНИЕ: Недостаточно файлов для класса '{cls_name}' "
                  f"(имеется {len(shuffled)}, требуется {n_test} для теста).")
            n_test  = len(shuffled)
            n_train = 0

        test_files  = shuffled[:n_test]
        train_files = shuffled[n_test:]

        # Копирование файлов в train и test
        for split, files in (("train", train_files), ("test", test_files)):
            dest = output_dir / split / cls_name
            dest.mkdir(parents=True, exist_ok=True)
            for src in files:
                shutil.copy2(src, dest / src.name)

        report[cls_name] = {"train": len(train_files), "test": len(test_files)}
        print(f"  {cls_name:12s}  всего={len(all_files):4d}  "
              f"train={len(train_files):4d}  test={len(test_files):4d}")

    return report


def parse_args():
    """
    Парсинг аргументов командной строки.

    Возвращает
    ──────────
    argparse.Namespace
        Объект с параметрами:
        - source_dir: путь к исходному датасету
        - output_dir: путь к выходной директории
        - seed: seed для воспроизводимости разбиения
    """
    p = argparse.ArgumentParser(
        description="Разбивка датасета на обучающую и тестовую выборки."
    )
    p.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE,
                   help="Корневая директория с исходными данными по классам")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                   help="Выходная директория для train/ и test/")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help="Seed генератора случайных чисел для воспроизводимого разбиения")
    return p.parse_args()


def main():
    """
    Главная функция программы.

    Выполняет:
    1. Парсинг аргументов командной строки
    2. Сбор путей к изображениям по классам
    3. Стратифицированное разбиение на train/test
    4. Копирование файлов в выходные директории
    5. Сохранение манифеста разбиения (split_manifest.json)
    6. Вывод финальной статистики
    """
    args = parse_args()
    print("=" * 55)
    print("  Разбивка датасета")
    print(f"  Источник : {args.source_dir}")
    print(f"  Выход    : {args.output_dir}")
    print(f"  Seed     : {args.seed}")
    print("=" * 55)

    classes = collect_images(args.source_dir)

    print("\nСобранные изображения:")
    for cls, files in classes.items():
        print(f"  {cls:12s}  {len(files)} файлов")

    print("\nРазбиение и копирование...")
    report = split_and_copy(classes, args.output_dir, args.seed)

    # Сохранение манифеста разбиения
    manifest = {
        "seed": args.seed,
        "source_dir": str(args.source_dir),
        "output_dir": str(args.output_dir),
        "split": report,
    }
    manifest_path = args.output_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), 
                             encoding="utf-8")

    print("\nГотово!")
    print(f"  Манифест сохранён → {manifest_path}")
    total_train = sum(v["train"] for v in report.values())
    total_test  = sum(v["test"]  for v in report.values())
    print(f"  Итого: {total_train} обучающих  /  {total_test} тестовых")
    print("=" * 55)


if __name__ == "__main__":
    main()
