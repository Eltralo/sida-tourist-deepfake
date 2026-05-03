#!/usr/bin/env python3
"""
streamlit_app/metrics.py
─────────────────────────
Расчёт метрик качества для Streamlit-приложения.

Логика соответствует ``detection/eval_metrics.py``: те же три класса,
та же нормализация меток, те же формулы Precision / Recall / F1. Отличие
в том, что здесь метрики считаются прямо в памяти (не из results.json),
а на вход подаются два списка y_true / y_pred либо DataFrame с колонками
``predicted`` и ``gt``.

Дополнительно поддерживается извлечение истинной метки из имени папки —
структура каталогов в проекте: ``photo/test/{real, full_synt, tempered}/``.

Использование
─────────────
    from streamlit_app.metrics import compute_metrics, infer_label_from_filename

    report = compute_metrics(y_true, y_pred)
    print(report.accuracy, report.macro_f1)
"""

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import pandas as pd

# ── Defaults ──────────────────────────────────────────────────────────────────
LABEL_NAMES = ["real", "fake", "tampered"]
CLASSES     = LABEL_NAMES  # alias

CLASSES_RU = {
    "real":     "Подлинное",
    "fake":     "Полностью синтетическое",
    "tampered": "Частично синтетическое",
}

# Маппинг имён папок проекта → канонические классы
FOLDER_TO_LABEL = {
    "real":            "real",
    "full_synt":       "fake",
    "full_synthetic":  "fake",
    "fake":            "fake",
    "tempered":        "tampered",
    "tampered ":       "tampered",
}


def normalise_label(label: str) -> str:
    """Нормализация строковых меток к канонической форме SIDA.

    Совпадает с одноимённой функцией из ``detection/eval_metrics.py``.
    """
    t = str(label).lower().strip()
    if t in ("fake", "full_synt", "full_synthetic", "synthetic", "fully synthetic"):
        return "fake"
    if t in ("tampered", "tempered", "altered", "manipulated"):
        return "tampered"
    if t in ("real", "authentic", "genuine"):
        return "real"
    return t


def infer_label_from_filename(name: str) -> Optional[str]:
    """Извлекает истинный класс из пути к файлу.

    Работает, если файл лежит в подпапке real/, full_synt/ или tempered/.
    Возвращает None, если ничего не подошло.
    """
    n = str(name).lower().replace("\\", "/")
    parts = n.split("/")
    for p in parts:
        if p in FOLDER_TO_LABEL:
            return FOLDER_TO_LABEL[p]
    for folder, label in FOLDER_TO_LABEL.items():
        if f"/{folder}/" in n or n.startswith(f"{folder}/"):
            return label
    if "tampered" in n or "tempered" in n:
        return "tampered"
    if "full_synt" in n or "full_synthetic" in n or "/fake/" in n:
        return "fake"
    if "/real/" in n or "real" in n.split("/")[0]:
        return "real"
    return None


def load_gt(source) -> dict:
    """Загружает GT-метки из CSV / DataFrame / dict.

    CSV должен содержать колонки filename (или path/file/name) и
    label (или class/gt/y/true).
    """
    if isinstance(source, dict):
        return dict(source)
    df = source if isinstance(source, pd.DataFrame) else pd.read_csv(source)
    cols = [c.lower() for c in df.columns]
    df.columns = cols
    fcol = next((c for c in ("filename", "file", "name", "path") if c in cols), None)
    lcol = next((c for c in ("label", "class", "gt", "y", "true") if c in cols), None)
    if fcol is None or lcol is None:
        raise ValueError(
            f"GT-файл должен содержать filename/path и label. Найдено: {cols}"
        )
    return {str(row[fcol]): normalise_label(row[lcol]) for _, row in df.iterrows()}


# ── Report container ──────────────────────────────────────────────────────────

@dataclass
class MetricsReport:
    accuracy: float
    macro_f1: float
    macro_precision: float
    macro_recall: float
    weighted_f1: float
    per_class: dict = field(default_factory=dict)
    confusion_matrix: np.ndarray = field(default_factory=lambda: np.zeros((3, 3), dtype=int))
    classes: list = field(default_factory=lambda: list(LABEL_NAMES))
    n_samples: int = 0

    def to_long_df(self) -> pd.DataFrame:
        """Длинная таблица: по строке на класс, столбцы P / R / F1 / Support."""
        rows = []
        for c in self.classes:
            d = self.per_class.get(c, {})
            rows.append({
                "Класс": CLASSES_RU.get(c, c),
                "Precision": d.get("precision", 0.0),
                "Recall":    d.get("recall",    0.0),
                "F1":        d.get("f1",        0.0),
                "Support":   d.get("support",   0),
            })
        return pd.DataFrame(rows)

    def confusion_df(self) -> pd.DataFrame:
        """Матрица ошибок с русскими названиями классов."""
        labels_ru = [CLASSES_RU[c] for c in self.classes]
        return pd.DataFrame(self.confusion_matrix, index=labels_ru, columns=labels_ru)


# ── Compute metrics ───────────────────────────────────────────────────────────

def compute_metrics(y_true: Iterable[str], y_pred: Iterable[str],
                    classes: Optional[list] = None) -> MetricsReport:
    """Считает все метрики по двум спискам меток одинаковой длины."""
    classes = list(classes or LABEL_NAMES)
    y_true = [normalise_label(x) for x in y_true]
    y_pred = [normalise_label(x) for x in y_pred]
    n = len(y_true)
    if n == 0:
        return MetricsReport(0, 0, 0, 0, 0)
    if n != len(y_pred):
        raise ValueError("y_true и y_pred должны быть одной длины")

    idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t not in idx or p not in idx:
            continue
        cm[idx[t], idx[p]] += 1

    per_class = {}
    f1s, ps, rs, supports = [], [], [], []
    for i, c in enumerate(classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        support = int(cm[i, :].sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[c] = {
            "precision": precision, "recall": recall, "f1": f1, "support": support,
        }
        f1s.append(f1); ps.append(precision); rs.append(recall); supports.append(support)

    accuracy = float(np.trace(cm)) / cm.sum() if cm.sum() else 0.0
    total = sum(supports) or 1
    weighted_f1 = float(sum(f * s for f, s in zip(f1s, supports)) / total)

    return MetricsReport(
        accuracy=accuracy,
        macro_f1=float(np.mean(f1s)),
        macro_precision=float(np.mean(ps)),
        macro_recall=float(np.mean(rs)),
        weighted_f1=weighted_f1,
        per_class=per_class,
        confusion_matrix=cm,
        classes=classes,
        n_samples=n,
    )
