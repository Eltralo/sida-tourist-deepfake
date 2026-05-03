#!/usr/bin/env python3
"""
streamlit_app/app.py
─────────────────────
Streamlit-приложение «Система оценки пользовательских фотографий
в туристических сервисах».

Демонстрационный стенд дообученной мультимодальной модели SIDA-7B,
выполненный в рамках магистерской диссертации НИЯУ МИФИ. Использует
тот же пайплайн инференса, что и ``detection/inference.py``, но
оборачивает его в интерактивный веб-интерфейс.

Поддерживаемые режимы работы:
    1. Одиночный анализ изображения с визуализацией маски подделанной
       области (для класса TAMPERED).
    2. Пакетный анализ нескольких файлов или ZIP-архива.
    3. Расчёт метрик качества (Overall Accuracy, Macro F1, Precision,
       Recall, F1-score по классам, Confusion Matrix) при наличии
       ground-truth разметки.

Использование
─────
    conda activate sida_modern
    cd streamlit_app/
    streamlit run app.py

    streamlit run app.py --server.port 8502

Дизайн рассчитан на демонстрацию с ноутбука 1440×900 (и видеозапись),
палитра — корпоративные цвета НИЯУ МИФИ (тёмно-синий + оранжевый).
"""

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# Подключаем общие модули streamlit_app/ к sys.path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Корень репозитория — на уровень выше streamlit_app/.
# Используется для дефолтных путей к ck/ и для импортов model.* / utils.*
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sida_inference import (  # noqa: E402
    DummyInference, SIDAInference,
    LABEL_REAL, LABEL_FAKE, LABEL_TAMPERED, LABEL_NAMES,
    LABEL_RU, overlay_mask, mask_area_fraction,
)
from metrics import (  # noqa: E402
    compute_metrics, infer_label_from_filename, load_gt,
    LABEL_NAMES as METRICS_LABELS, CLASSES_RU,
)

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL_DIR = REPO_ROOT / "ck" / "SIDA-7B"
DEFAULT_HEAD_PATH = REPO_ROOT / "ck" / "cls_head_new.pth"
DEFAULT_PRECISION = "bf16"


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Система оценки пользовательских фотографий",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ── Styling: palette НИЯУ МИФИ + типографика для 1440×900 ─────────────────────

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=PT+Serif:wght@400;700&family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');

:root {
    /* Корпоративные цвета НИЯУ МИФИ */
    --mephi-blue:        #003D82;
    --mephi-blue-deep:   #002D66;
    --mephi-blue-soft:   #E8EFF8;
    --mephi-orange:      #F39200;
    --mephi-orange-deep: #D17A00;
    --mephi-orange-soft: #FFF1DB;

    /* Семантические цвета классов */
    --green:        #2E8B57;
    --green-deep:   #1F6B3D;
    --red:          #C81E1E;

    /* Нейтральные */
    --bg:           #ffffff;
    --bg-soft:      #F7F8FB;
    --ink:          #0E1A2B;
    --ink-soft:     #2C3A52;
    --ink-mute:     #5A6577;
    --ink-faint:    #98A1B3;
    --rule:         #D8DEE7;
    --rule-soft:    #EBEEF3;
}

html, body, [class*="css"], .stApp {
    background: var(--bg) !important;
    color: var(--ink) !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
    font-size: 16px;
}

.block-container {
    max-width: 1400px !important;
    padding-top: 1.2rem !important;
    padding-bottom: 3rem !important;
}

.stApp > header { background: transparent !important; }

section[data-testid="stSidebar"] {
    background: var(--bg-soft) !important;
    border-right: 1px solid var(--rule);
}
section[data-testid="stSidebar"] * { color: var(--ink-soft) !important; }

h1, h2, h3, h4 {
    font-family: 'PT Serif', Georgia, serif !important;
    color: var(--ink) !important;
    font-weight: 700;
    letter-spacing: -0.01em;
    line-height: 1.15;
}

/* ═══ ШАПКА ═══ */
.app-title {
    padding: 8px 0 26px;
    border-bottom: 4px solid var(--mephi-blue);
    margin-bottom: 18px;
    position: relative;
}
.app-title::after {
    content: "";
    position: absolute;
    bottom: -4px; left: 0;
    width: 140px; height: 4px;
    background: var(--mephi-orange);
}
.app-title .kicker {
    font-family: 'Inter', sans-serif;
    font-size: 18px;
    font-weight: 700;
    color: var(--mephi-blue);
    text-transform: uppercase;
    letter-spacing: 0.18em;
    margin-bottom: 14px;
    line-height: 1;
}
.app-title h1 {
    font-size: 56px !important;
    margin: 0 0 6px !important;
    line-height: 1.05 !important;
    color: var(--mephi-blue-deep) !important;
    font-weight: 700 !important;
}
.app-title h1 .accent { color: var(--mephi-orange); }

/* ═══ ЗАГОЛОВКИ РАЗДЕЛОВ ═══ */
.section { margin-top: 32px; margin-bottom: 16px; }
.section h2 {
    font-size: 30px !important;
    margin: 0 0 6px !important;
    padding-bottom: 10px;
    border-bottom: 2px solid var(--mephi-blue);
    color: var(--mephi-blue-deep) !important;
    display: inline-block;
    padding-right: 24px;
}
.section .desc {
    font-family: 'PT Serif', serif;
    font-size: 16px;
    color: var(--ink-mute);
    font-style: italic;
    margin-top: 10px;
}

/* ═══ КАРТОЧКА ВЕРДИКТА ═══ */
.verdict-box {
    background: linear-gradient(180deg, var(--mephi-blue) 0%, var(--mephi-blue-deep) 100%);
    border-radius: 4px;
    padding: 28px 32px;
    color: white;
    box-shadow: 0 4px 18px rgba(0, 61, 130, 0.25);
    position: relative;
    overflow: hidden;
}
.verdict-box::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; height: 4px;
    background: var(--mephi-orange);
}
.verdict-box .kicker {
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    color: var(--mephi-orange);
    margin-bottom: 18px;
}
.verdict-box .label-big {
    font-family: 'PT Serif', serif;
    font-size: 38px;
    font-weight: 700;
    line-height: 1.1;
    color: white;
    margin-bottom: 16px;
}
.verdict-box .meta-row {
    display: flex;
    gap: 22px;
    padding-top: 14px;
    border-top: 1px solid rgba(255,255,255,0.18);
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    color: rgba(255,255,255,0.85);
    flex-wrap: wrap;
}
.verdict-box .meta-row .item .k {
    text-transform: uppercase;
    font-size: 10px;
    letter-spacing: 0.15em;
    color: rgba(255,255,255,0.55);
    margin-bottom: 3px;
}
.verdict-box .meta-row .item .v {
    font-size: 16px;
    color: white;
    font-weight: 600;
}

/* ═══ РАСПРЕДЕЛЕНИЕ УВЕРЕННОСТИ ═══ */
.conf-card {
    border: 2px solid var(--mephi-blue);
    border-radius: 4px;
    padding: 22px 26px;
    margin-top: 16px;
    background: white;
}
.conf-card .title {
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    font-weight: 700;
    color: var(--mephi-blue);
    text-transform: uppercase;
    letter-spacing: 0.18em;
    margin-bottom: 16px;
}
.conf-row {
    display: grid;
    grid-template-columns: 220px 1fr 80px;
    gap: 14px;
    align-items: center;
    margin: 10px 0;
}
.conf-row .name {
    font-family: 'Inter', sans-serif;
    font-size: 16px;
    color: var(--ink-soft);
    font-weight: 500;
}
.conf-row .bar-wrap {
    background: var(--rule-soft);
    height: 14px;
    border-radius: 1px;
    overflow: hidden;
    border: 1px solid var(--rule);
}
.conf-row .bar { height: 100%; transition: width 0.4s ease; }
.conf-row .pct {
    text-align: right;
    font-family: 'JetBrains Mono', monospace;
    font-size: 18px;
    color: var(--ink);
    font-weight: 600;
}

/* ═══ KPI ═══ */
.kpi-big {
    background: white;
    border: 1px solid var(--rule);
    border-top: 4px solid var(--mephi-blue);
    padding: 18px 20px;
    border-radius: 2px;
}
.kpi-big .k {
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    font-weight: 600;
    color: var(--ink-mute);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 8px;
}
.kpi-big .v {
    font-family: 'PT Serif', serif;
    font-size: 36px;
    font-weight: 700;
    color: var(--mephi-blue-deep);
    line-height: 1;
}
.kpi-big .sub {
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    color: var(--ink-faint);
    margin-top: 6px;
}

/* ═══ ТАБЫ ═══ */
div[data-baseweb="tab-list"] {
    border-bottom: 3px solid var(--rule) !important;
    gap: 0 !important;
    margin-bottom: 24px !important;
}
button[data-baseweb="tab"] {
    font-family: 'PT Serif', serif !important;
    font-size: 26px !important;
    color: var(--ink-mute) !important;
    font-weight: 700 !important;
    background: transparent !important;
    padding: 18px 32px !important;
    border-bottom: 4px solid transparent !important;
    margin-bottom: -3px !important;
}
button[data-baseweb="tab"]:hover { color: var(--mephi-blue) !important; }
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--mephi-blue-deep) !important;
    border-bottom-color: var(--mephi-orange) !important;
    background: var(--mephi-blue-soft) !important;
}
.subtabs button[data-baseweb="tab"] {
    font-size: 18px !important;
    padding: 12px 22px !important;
}

/* ═══ ПОДПИСИ К РИСУНКАМ ═══ */
.fig-caption {
    font-family: 'PT Serif', serif;
    font-size: 14px;
    color: var(--ink-mute);
    text-align: center;
    font-style: italic;
    margin-top: 8px;
    line-height: 1.4;
}

/* ═══ КНОПКИ ═══ */
.stButton > button, .stDownloadButton > button {
    background: var(--mephi-blue) !important;
    border: 1px solid var(--mephi-blue) !important;
    color: white !important;
    border-radius: 2px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    padding: 10px 22px !important;
    letter-spacing: 0.04em;
    box-shadow: none !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: var(--mephi-orange) !important;
    border-color: var(--mephi-orange) !important;
}

/* ═══ ЗАГРУЗЧИК ФАЙЛОВ ═══ */
[data-testid="stFileUploaderDropzone"] {
    background: var(--mephi-blue-soft) !important;
    border: 2px dashed var(--mephi-blue) !important;
    border-radius: 4px !important;
    padding: 28px !important;
}
[data-testid="stFileUploaderDropzone"] section {
    color: var(--mephi-blue-deep) !important;
}

/* ═══ DataFrame ═══ */
[data-testid="stDataFrame"] {
    border: 1px solid var(--rule);
    border-radius: 2px;
    font-size: 14px;
}

/* ═══ Превью галереи ═══ */
.preview-meta {
    margin-top: 6px;
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    line-height: 1.3;
}
.preview-meta .lbl-real { color: var(--green-deep); font-weight: 700; }
.preview-meta .lbl-fake { color: var(--red); font-weight: 700; }
.preview-meta .lbl-tamp { color: var(--mephi-orange-deep); font-weight: 700; }
.preview-meta .filename {
    color: var(--ink-faint);
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
    display: block;
    margin-top: 2px;
}

/* ═══ Сноска ═══ */
.app-foot {
    margin-top: 50px;
    padding: 14px 0;
    border-top: 2px solid var(--mephi-blue);
    text-align: center;
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    color: var(--ink-mute);
    letter-spacing: 0.06em;
}
.app-foot strong { color: var(--mephi-blue-deep); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ── Model loading (cached for the session) ────────────────────────────────────

@st.cache_resource(show_spinner="Загружаю SIDA-7B и дообученную голову…")
def load_real_model(weights_path: str, cls_head_path: str,
                    project_root: str, precision: str):
    """Создаёт SIDAInference и кэширует на время сессии."""
    return SIDAInference(
        weights_path=weights_path,
        cls_head_path=cls_head_path,
        project_root=project_root,
        precision=precision,
    )


@st.cache_resource(show_spinner=False)
def load_dummy():
    """Заглушка для демо-режима без загрузки реальной модели."""
    return DummyInference()


# ── Palette and helpers ───────────────────────────────────────────────────────

PALETTE = {
    LABEL_REAL:     "#2E8B57",      # зелёный
    LABEL_FAKE:     "#C81E1E",      # красный
    LABEL_TAMPERED: "#F39200",      # оранжевый МИФИ
}
CLASS_CSS = {LABEL_REAL: "real", LABEL_FAKE: "fake", LABEL_TAMPERED: "tamp"}

LABEL_SHORT = {
    LABEL_REAL:     "Подлинное",
    LABEL_FAKE:     "Полностью синтетическое",
    LABEL_TAMPERED: "Частично синтетическое",
}


def section(title: str, desc: str = ""):
    """Заголовок раздела (без номера) с опциональным описанием."""
    desc_html = f'<div class="desc">{desc}</div>' if desc else ""
    st.markdown(f'<div class="section"><h2>{title}</h2>{desc_html}</div>',
                unsafe_allow_html=True)


def fig_caption(text: str):
    """Курсивная подпись под рисунком."""
    st.markdown(f'<div class="fig-caption">{text}</div>', unsafe_allow_html=True)


def render_verdict_box(result):
    """Большая синяя карточка-вердикт в правой колонке одиночного анализа."""
    mask_str = (f"{mask_area_fraction(result.mask)*100:.1f}%"
                if result.mask is not None else "—")
    st.markdown(
        f"""
        <div class="verdict-box">
            <div class="kicker">Предсказанный класс</div>
            <div class="label-big">{LABEL_SHORT[result.label]}</div>
            <div class="meta-row">
                <div class="item">
                    <div class="k">Уверенность</div>
                    <div class="v">{result.confidence*100:.1f}%</div>
                </div>
                <div class="item">
                    <div class="k">Время</div>
                    <div class="v">{result.latency_s:.2f} с</div>
                </div>
                <div class="item">
                    <div class="k">Площадь маски</div>
                    <div class="v">{mask_str}</div>
                </div>
                <div class="item">
                    <div class="k">Класс</div>
                    <div class="v">{result.label}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_confidence_card(logits):
    """Карточка с распределением вероятностей по трём классам."""
    rows_html = ""
    for cls in (LABEL_REAL, LABEL_FAKE, LABEL_TAMPERED):
        p = logits.get(cls, 0.0)
        rows_html += (
            f'<div class="conf-row">'
            f'<div class="name">{LABEL_RU[cls]}</div>'
            f'<div class="bar-wrap"><div class="bar" '
            f'style="width:{p*100:.1f}%; background:{PALETTE[cls]}"></div></div>'
            f'<div class="pct">{p*100:.1f}%</div>'
            f'</div>'
        )
    st.markdown(
        f'<div class="conf-card">'
        f'<div class="title">Распределение вероятностей</div>'
        f'{rows_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def kpi_big(label: str, value: str, sub: str = ""):
    """Крупная KPI-плитка с верхним синим бордером."""
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="kpi-big"><div class="k">{label}</div>'
        f'<div class="v">{value}</div>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def plotly_layout(fig, height: int = 380,
                  axis_title_size: int = 15, tick_size: int = 13):
    """Единый стиль графиков Plotly: синие оси, крупные подписи."""
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font=dict(family="Inter, sans-serif", color="#0E1A2B", size=tick_size),
        margin=dict(l=20, r=20, t=40, b=40),
        legend=dict(
            font=dict(size=14, color="#0E1A2B"),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#D8DEE7", borderwidth=1,
        ),
        title=dict(font=dict(size=16, color="#003D82", family="PT Serif"), x=0.02),
    )
    fig.update_xaxes(
        gridcolor="#EBEEF3", zerolinecolor="#003D82", linecolor="#003D82",
        linewidth=2, tickfont=dict(size=tick_size, color="#0E1A2B"),
        title_font=dict(size=axis_title_size, color="#003D82", family="Inter"),
    )
    fig.update_yaxes(
        gridcolor="#EBEEF3", zerolinecolor="#003D82", linecolor="#003D82",
        linewidth=2, tickfont=dict(size=tick_size, color="#0E1A2B"),
        title_font=dict(size=axis_title_size, color="#003D82", family="Inter"),
    )
    return fig


def collect_uploaded_images(files):
    """Принимает список UploadedFile (multi-upload + ZIP).

    Возвращает список кортежей (имя файла, PIL.Image).
    """
    out = []
    for f in files:
        name = f.name
        f.seek(0)
        data = f.read()
        if name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for inner in z.namelist():
                    if (inner.lower().endswith(
                            (".jpg", ".jpeg", ".png", ".bmp", ".webp"))
                            and not inner.startswith("__MACOSX")):
                        try:
                            img = Image.open(io.BytesIO(z.read(inner))).convert("RGB")
                            out.append((inner, img))
                        except Exception:
                            continue
        else:
            try:
                img = Image.open(io.BytesIO(data)).convert("RGB")
                out.append((name, img))
            except Exception:
                continue
    return out


def run_batch(model, images, progress_cb=None):
    """Прогоняет модель по списку изображений.

    Возвращает (DataFrame с результатами, dict {имя файла: (image, mask)}).
    """
    rows, masks = [], {}
    for i, (name, img) in enumerate(images):
        try:
            res = model.predict(img)
        except Exception as exc:
            rows.append({
                "filename": name, "predicted": "error", "predicted_ru": "Ошибка",
                "verdict_ru": str(exc)[:120],
                "confidence": 0.0, "p_real": 0.0, "p_fake": 0.0, "p_tampered": 0.0,
                "mask_area": 0.0, "latency_s": 0.0,
            })
            continue
        rows.append({
            "filename":    name,
            "predicted":   res.label,
            "predicted_ru": res.label_ru,
            "verdict_ru":  res.verdict_ru,
            "confidence":  res.confidence,
            "p_real":      res.logits.get(LABEL_REAL, 0.0),
            "p_fake":      res.logits.get(LABEL_FAKE, 0.0),
            "p_tampered":  res.logits.get(LABEL_TAMPERED, 0.0),
            "mask_area":   mask_area_fraction(res.mask),
            "latency_s":   res.latency_s,
        })
        if res.mask is not None:
            masks[name] = (img, res.mask)
        if progress_cb:
            progress_cb(i + 1, len(images))
    return pd.DataFrame(rows), masks


# ── Sidebar: configuration and demo toggle ────────────────────────────────────

with st.sidebar:
    st.markdown("### Конфигурация")
    demo = st.toggle(
        "Демо-режим",
        value=False,
        help="Заглушка без загрузки модели — для проверки UI",
    )
    st.markdown("---")
    st.markdown("**Параметры модели**")
    project_root  = st.text_input("Корень проекта", value=str(REPO_ROOT))
    weights_path  = st.text_input("Веса SIDA-7B",   value=str(DEFAULT_MODEL_DIR))
    cls_head_path = st.text_input("Дообученная голова", value=str(DEFAULT_HEAD_PATH))
    precision     = st.selectbox("Точность", ["bf16", "fp16", "fp32"], index=0)
    st.markdown("---")
    st.caption("SIDA-7B + cls_head_new.pth")
    st.caption("Классы: real · fake · tampered")


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="app-title">
        <div class="kicker">SIDA · Detection · Localization</div>
        <h1>Система оценки пользовательских<br>фотографий <span class="accent">в туристических сервисах</span></h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# Загружаем модель (или заглушку для демо)
if demo:
    model = load_dummy()
else:
    try:
        model = load_real_model(weights_path, cls_head_path, project_root, precision)
    except Exception as exc:
        st.error(f"Не удалось загрузить модель: {exc}")
        st.caption("Проверьте пути в боковой панели или включите демо-режим.")
        st.stop()


# ── Top-level tabs ────────────────────────────────────────────────────────────

tab_analysis, tab_metrics, tab_about = st.tabs(
    ["Анализ", "Метрики качества", "О проекте"]
)


# ════════════════════════════════════════════════════════════════════
#  TAB — Анализ (одиночный + пакетный)
# ════════════════════════════════════════════════════════════════════

with tab_analysis:

    # ── Одиночный анализ ─────────────────────────────────────────────
    section("Одиночный анализ", "Загрузите одно изображение для анализа")

    up_single = st.file_uploader(
        "Изображение",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=False,
        key="single",
        label_visibility="collapsed",
    )

    if up_single is None:
        st.caption("Перетащите изображение в область выше или выберите файл.")
    else:
        image = Image.open(up_single).convert("RGB")
        col_in, col_out = st.columns([1, 1.05], gap="large")

        with col_in:
            st.image(image, use_container_width=True)
            fig_caption(f"Анализируемое изображение «{up_single.name}», "
                        f"{image.size[0]}×{image.size[1]} px")

        with col_out:
            with st.spinner("Модель работает…"):
                result = model.predict(image)
            render_verdict_box(result)
            render_confidence_card(result.logits)

        # Локализация подделки — отдельным блоком ниже
        if result.label == LABEL_TAMPERED and result.mask is not None:
            section("Локализация подделанной области")
            cm1, cm2, cm3 = st.columns(3, gap="medium")
            with cm1:
                st.image(image, use_container_width=True)
                fig_caption("Исходное изображение")
            with cm2:
                mask_vis = Image.fromarray(result.mask).convert("L")
                if mask_vis.size != image.size:
                    mask_vis = mask_vis.resize(image.size, Image.NEAREST)
                st.image(mask_vis, use_container_width=True)
                fig_caption("Сегментационная маска")
            with cm3:
                st.image(overlay_mask(image, result.mask), use_container_width=True)
                fig_caption("Наложение маски на оригинал")

    # ── Пакетный анализ ──────────────────────────────────────────────
    section("Пакетный анализ",
            "Загрузите несколько изображений или ZIP-архив с папкой фотографий.")

    up_batch = st.file_uploader(
        "Несколько файлов или ZIP",
        type=["jpg", "jpeg", "png", "bmp", "webp", "zip"],
        accept_multiple_files=True,
        key="batch",
        label_visibility="collapsed",
        help="Можно выделить несколько JPG/PNG в проводнике (Ctrl/Cmd-клик) "
             "или загрузить ZIP-архив.",
    )

    if up_batch:
        images = collect_uploaded_images(up_batch)
        if not images:
            st.warning("Не удалось прочитать ни одного изображения.")
        else:
            colL, colR = st.columns([3, 1])
            with colL:
                st.markdown(
                    f'<div style="font-family:Inter; font-size:18px; color:#003D82; '
                    f'font-weight:600; padding-top:8px">'
                    f'К обработке: {len(images)} изображений</div>',
                    unsafe_allow_html=True,
                )
            with colR:
                run = st.button("Запустить анализ", use_container_width=True)
            if run:
                pbar = st.progress(0.0, text="Обработка…")
                df, masks = run_batch(
                    model, images,
                    progress_cb=lambda i, n: pbar.progress(i / n, text=f"{i}/{n}"),
                )
                pbar.empty()
                st.session_state["batch_df"] = df
                st.session_state["batch_masks"] = masks
                st.session_state["batch_previews"] = {n: img for n, img in images}

    # ── Результаты пакета ────────────────────────────────────────────
    if "batch_df" in st.session_state:
        df       = st.session_state["batch_df"]
        masks    = st.session_state.get("batch_masks", {})
        previews = st.session_state.get("batch_previews", {})
        valid_df = df[df["predicted"] != "error"].copy()

        section("Результаты классификации",
                f"Загружено и обработано {len(valid_df)} из {len(df)} изображений.")

        # Верхние KPI
        k1, k2, k3, k4 = st.columns(4)
        with k1: kpi_big("Всего изображений", f"{len(df)}")
        with k2: kpi_big("Обработано", f"{len(valid_df)}",
                         f"ошибок: {(df['predicted']=='error').sum()}"
                         if (df['predicted']=='error').any() else "")
        with k3:
            mean_conf = valid_df["confidence"].mean() if len(valid_df) else 0.0
            kpi_big("Средняя уверенность", f"{mean_conf*100:.1f}%")
        with k4:
            n_classes = valid_df["predicted"].nunique() if len(valid_df) else 0
            kpi_big("Классов выявлено", f"{n_classes}")

        st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)
        st.markdown('<div class="subtabs">', unsafe_allow_html=True)
        sub_imgs, sub_stats, sub_cats, sub_masks, sub_log = st.tabs(
            ["Изображения", "Статистика", "Категории", "Маски", "Детальный лог"]
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Изображения ──
        with sub_imgs:
            items = list(zip(valid_df["filename"].tolist(),
                             valid_df["predicted"].tolist(),
                             valid_df["predicted_ru"].tolist(),
                             valid_df["confidence"].tolist()))
            cols_per_row = 4
            for i in range(0, len(items), cols_per_row):
                row = st.columns(cols_per_row)
                for col, (name, lbl, lbl_ru, conf) in zip(row, items[i:i + cols_per_row]):
                    with col:
                        img = previews.get(name)
                        if img is None and name in masks:
                            img = masks[name][0]
                        if img is not None:
                            st.image(img, use_container_width=True)
                        css = CLASS_CSS.get(lbl, "real")
                        st.markdown(
                            f'<div class="preview-meta">'
                            f'<span class="lbl-{css}">{lbl_ru}</span>'
                            f' &nbsp;·&nbsp; <strong>{conf*100:.0f}%</strong>'
                            f'<span class="filename">{name[:48]}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

        # ── Статистика ──
        with sub_stats:
            sc1, sc2 = st.columns(2, gap="large")

            with sc1:
                fig = px.histogram(
                    valid_df, x="confidence", color="predicted", nbins=20,
                    color_discrete_map=PALETTE,
                    labels={"confidence": "Уверенность модели",
                            "predicted": "Класс", "count": "Количество"},
                )
                fig.update_traces(marker_line_color="#003D82", marker_line_width=1)
                st.plotly_chart(plotly_layout(fig, 380), use_container_width=True)
                fig_caption("Распределение уверенности модели по классам")

            with sc2:
                fig = px.scatter(
                    valid_df, x="latency_s", y="confidence", color="predicted",
                    color_discrete_map=PALETTE, hover_data=["filename"],
                    labels={"latency_s": "Время инференса, с",
                            "confidence": "Уверенность", "predicted": "Класс"},
                )
                fig.update_traces(marker=dict(size=12, line=dict(color="#003D82", width=1.5)))
                st.plotly_chart(plotly_layout(fig, 380), use_container_width=True)
                fig_caption("Соотношение времени инференса и уверенности модели")

            sc3, sc4 = st.columns(2, gap="large")

            with sc3:
                agg = (valid_df.groupby("predicted")["confidence"]
                       .agg(["mean", "std", "count"]).reindex(LABEL_NAMES).reset_index())
                agg["Класс"] = agg["predicted"].map(CLASSES_RU)
                fig = go.Figure(go.Bar(
                    x=agg["Класс"], y=agg["mean"]*100,
                    marker=dict(color=[PALETTE.get(c, "#999") for c in agg["predicted"]],
                                line=dict(color="#003D82", width=1.5)),
                    text=[f"{v*100:.1f}%" if pd.notna(v) else "—" for v in agg["mean"]],
                    textposition="outside",
                    textfont=dict(size=15, color="#003D82", family="Inter"),
                ))
                fig.update_layout(yaxis=dict(title="Средняя уверенность, %"),
                                  xaxis=dict(title=""))
                st.plotly_chart(plotly_layout(fig, 380), use_container_width=True)
                fig_caption("Средняя уверенность модели в разрезе классов")

            with sc4:
                fig = go.Figure(go.Box(
                    y=valid_df["latency_s"], boxpoints="all", jitter=0.3, pointpos=0,
                    marker=dict(color="#F39200", size=8,
                                line=dict(color="#003D82", width=1.2)),
                    line=dict(color="#003D82", width=2),
                    fillcolor="rgba(243, 146, 0, 0.15)",
                ))
                fig.update_layout(yaxis=dict(title="Время инференса, с"),
                                  xaxis=dict(title=""))
                st.plotly_chart(plotly_layout(fig, 380), use_container_width=True)
                fig_caption("Распределение времени инференса")

        # ── Категории ──
        with sub_cats:
            counts = valid_df["predicted"].value_counts().reindex(LABEL_NAMES, fill_value=0)
            cc1, cc2 = st.columns([1.4, 1], gap="large")

            with cc1:
                fig = go.Figure(go.Bar(
                    x=[CLASSES_RU[c] for c in LABEL_NAMES],
                    y=counts.values,
                    marker=dict(color=[PALETTE[c] for c in LABEL_NAMES],
                                line=dict(color="#003D82", width=1.5)),
                    text=counts.values, textposition="outside",
                    textfont=dict(size=20, color="#003D82", family="PT Serif"),
                ))
                fig.update_layout(yaxis=dict(title="Количество изображений"),
                                  xaxis=dict(title=""))
                st.plotly_chart(plotly_layout(fig, 420, axis_title_size=15, tick_size=14),
                                use_container_width=True)
                fig_caption("Распределение предсказанных классов в загруженном корпусе")

            with cc2:
                fig = go.Figure(go.Pie(
                    labels=[CLASSES_RU[c] for c in LABEL_NAMES],
                    values=counts.values,
                    hole=0.55,
                    marker=dict(colors=[PALETTE[c] for c in LABEL_NAMES],
                                line=dict(color="white", width=3)),
                    textinfo="percent",
                    textfont=dict(family="Inter", size=16, color="white"),
                ))
                st.plotly_chart(plotly_layout(fig, 320), use_container_width=True)
                fig_caption("Доли классов в корпусе")

                pcdf = pd.DataFrame({
                    "Класс":      [CLASSES_RU[c] for c in LABEL_NAMES],
                    "Количество": counts.values,
                    "Доля":       [f"{v / max(1, counts.sum()) * 100:.1f}%"
                                   for v in counts.values],
                })
                st.dataframe(pcdf, use_container_width=True, hide_index=True)

        # ── Маски ──
        with sub_masks:
            if not masks:
                st.info("В этом корпусе нет изображений с предсказанным классом "
                        "«частично синтетическое» — масок не построено.")
            else:
                st.markdown(
                    f'<div style="font-family:Inter; font-size:16px; color:#003D82; '
                    f'font-weight:600; margin-bottom:14px">'
                    f'Сегментационные маски подделанных областей — {len(masks)} шт.</div>',
                    unsafe_allow_html=True,
                )
                items = list(masks.items())
                cols_per_row = 3
                for i in range(0, len(items), cols_per_row):
                    row = st.columns(cols_per_row)
                    for col, (name, (img, mask)) in zip(row, items[i:i + cols_per_row]):
                        with col:
                            st.image(overlay_mask(img, mask), use_container_width=True)
                            cov = (mask > 0).mean() * 100
                            st.markdown(
                                f'<div class="preview-meta">'
                                f'<span class="lbl-tamp">{cov:.1f}%</span>'
                                f' &nbsp;·&nbsp; покрытие'
                                f'<span class="filename">{name[:48]}</span>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

        # ── Детальный лог ──
        with sub_log:
            show_df = df[["filename", "predicted_ru", "confidence",
                          "p_real", "p_fake", "p_tampered",
                          "mask_area", "latency_s"]].copy()
            show_df.columns = ["Файл", "Вердикт", "Уверенность",
                               "P(real)", "P(fake)", "P(tampered)",
                               "Доля маски", "Время, с"]
            st.dataframe(
                show_df.style.format({
                    "Уверенность": "{:.3f}", "P(real)": "{:.3f}", "P(fake)": "{:.3f}",
                    "P(tampered)": "{:.3f}", "Доля маски": "{:.3f}", "Время, с": "{:.2f}",
                }),
                use_container_width=True, height=420,
            )
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Скачать как CSV", csv,
                               file_name="sida_results.csv", mime="text/csv")


# ════════════════════════════════════════════════════════════════════
#  TAB — Метрики качества
# ════════════════════════════════════════════════════════════════════

with tab_metrics:
    section("Оценка качества модели",
            "Метрики рассчитываются по результатам пакетного анализа "
            "и истинным меткам (ground truth).")

    if "batch_df" not in st.session_state:
        st.info("Сначала выполните пакетный анализ во вкладке «Анализ».")
    else:
        df = st.session_state["batch_df"]
        df = df[df["predicted"] != "error"].copy()

        st.markdown('<div style="font-family:Inter; font-size:15px; color:#003D82; '
                    'font-weight:600; margin-bottom:6px">Источник истинных меток</div>',
                    unsafe_allow_html=True)
        gt_mode = st.radio(
            "GT",
            ["Автоматически из имени файла/папки", "Загрузить CSV (filename, label)"],
            horizontal=True, label_visibility="collapsed",
        )

        gt_dict = {}
        if gt_mode.startswith("Автоматически"):
            for n in df["filename"]:
                gt = infer_label_from_filename(n)
                if gt:
                    gt_dict[n] = gt
        else:
            gt_file = st.file_uploader("CSV", type=["csv"], key="gt_csv",
                                       label_visibility="collapsed")
            if gt_file is not None:
                gt_dict = load_gt(pd.read_csv(gt_file))

        df_eval = df[df["filename"].isin(gt_dict)].copy()
        df_eval["gt"] = df_eval["filename"].map(gt_dict)
        st.caption(f"Сопоставлено с GT: {len(df_eval)} из {len(df)} изображений.")

        if len(df_eval) == 0:
            st.warning("Ни одна строка не сопоставлена с GT — проверьте формат разметки.")
        else:
            report = compute_metrics(df_eval["gt"], df_eval["predicted"])

            mk1, mk2, mk3, mk4 = st.columns(4)
            with mk1: kpi_big("Accuracy",        f"{report.accuracy*100:.2f}%")
            with mk2: kpi_big("Macro F1",        f"{report.macro_f1:.3f}")
            with mk3: kpi_big("Macro Precision", f"{report.macro_precision:.3f}")
            with mk4: kpi_big("Macro Recall",    f"{report.macro_recall:.3f}")

            section("Матрица ошибок")
            cm_df = report.confusion_df()
            fig = px.imshow(
                cm_df.values, x=cm_df.columns, y=cm_df.index,
                text_auto=True, aspect="auto",
                color_continuous_scale=[[0, "#FFFFFF"], [0.5, "#E8EFF8"], [1, "#003D82"]],
                labels=dict(x="Предсказанный класс", y="Истинный класс", color="Количество"),
            )
            fig.update_traces(textfont=dict(size=22, family="PT Serif", color="#0E1A2B"))
            st.plotly_chart(plotly_layout(fig, 480, axis_title_size=16, tick_size=14),
                            use_container_width=True)
            fig_caption("Матрица ошибок: строки — истинные классы, столбцы — предсказания модели")

            section("Метрики по классам")
            mc1, mc2 = st.columns([1.4, 1], gap="large")
            with mc1:
                pcdf = report.to_long_df().melt(
                    id_vars="Класс", value_vars=["Precision", "Recall", "F1"],
                    var_name="Метрика", value_name="Значение",
                )
                fig = px.bar(
                    pcdf, x="Класс", y="Значение", color="Метрика", barmode="group",
                    color_discrete_sequence=["#003D82", "#F39200", "#2E8B57"],
                )
                fig.update_traces(marker_line_color="#0E1A2B", marker_line_width=1)
                fig.update_yaxes(range=[0, 1.05])
                st.plotly_chart(plotly_layout(fig, 420, axis_title_size=15, tick_size=14),
                                use_container_width=True)
                fig_caption("Precision, Recall и F1-score по каждому классу")
            with mc2:
                st.dataframe(
                    report.to_long_df().style.format({
                        "Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}",
                    }),
                    use_container_width=True, hide_index=True,
                )


# ════════════════════════════════════════════════════════════════════
#  TAB — О проекте
# ════════════════════════════════════════════════════════════════════

with tab_about:
    section("О проекте")
    st.markdown(
        """
<div style="font-family: 'PT Serif', serif; font-size: 18px; line-height: 1.65;
            color: #0E1A2B; max-width: 920px;">
Данный проект выполнен в рамках магистерской диссертации
<strong>НИЯУ&nbsp;МИФИ</strong> и представляет собой реализацию системы
проверки пользовательских фотографий в туристических сервисах с применением
дообучения предобученной мультимодальной модели <strong>SIDA-7B</strong>.
</div>
""",
        unsafe_allow_html=True,
    )

    section("Архитектура решения")
    st.markdown(
        """
<div style="font-family: 'Inter', sans-serif; font-size: 16px; line-height: 1.6;
            color: #2C3A52; max-width: 920px;">
Базой служит мультимодальная модель LISA-7B (LLaVA + SAM ViT-H), дообученная
методом LoRA на датасете SID-Set. Поверх модели обучена отдельная
классификационная голова (<code style="font-family: 'JetBrains Mono';
background: #F7F8FB; padding: 2px 6px; border: 1px solid #D8DEE7;">cls_head_new.pth</code>,
многослойный перцептрон 4096&nbsp;→&nbsp;2048&nbsp;→&nbsp;3),
которая по эмбеддингу <code style="font-family: 'JetBrains Mono';
background: #F7F8FB; padding: 2px 6px; border: 1px solid #D8DEE7;">[CLS]</code>-токена
относит изображение к одному из трёх классов.
</div>
""",
        unsafe_allow_html=True,
    )

    section("Поддерживаемые классы")
    cls_cols = st.columns(3)
    descriptions = [
        ("Подлинное", "real",
         "Настоящее фото, не подвергавшееся синтезу или модификации.",
         PALETTE[LABEL_REAL]),
        ("Полностью синтетическое", "fake",
         "Изображение целиком сгенерировано моделью (FLUX, Seedream, Z-Image, Imagen и т.п.).",
         PALETTE[LABEL_FAKE]),
        ("Частично синтетическое", "tampered",
         "Реальное фото с подменённой областью. Дополнительно строится сегментационная маска.",
         PALETTE[LABEL_TAMPERED]),
    ]
    for col, (title, code, desc, color) in zip(cls_cols, descriptions):
        with col:
            st.markdown(
                f"""
                <div style="border:1px solid #D8DEE7; border-top:4px solid {color};
                            border-radius:2px; padding:18px 20px; background:#F7F8FB;
                            min-height: 180px;">
                    <div style="font-family:'PT Serif', serif; font-size:22px;
                                font-weight:700; color:{color}; margin-bottom:6px;">
                        {title}
                    </div>
                    <div style="font-family:'JetBrains Mono', monospace; font-size:11px;
                                color:#5A6577; letter-spacing:0.1em; margin-bottom:12px;
                                text-transform:uppercase;">
                        class · {code}
                    </div>
                    <div style="font-family:'PT Serif', serif; font-size:15px;
                                color:#2C3A52; line-height:1.5;">
                        {desc}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    section("Цитирование базовой модели")
    st.markdown(
        """
<div style="font-family: 'PT Serif', serif; font-size: 15px; line-height: 1.6;
            color: #2C3A52; font-style: italic; max-width: 920px;">
Huang Z. et al. SIDA: Social Media Image Deepfake Detection,
Localization and Explanation with Large Multimodal Model. CVPR, 2025.
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="app-foot"><strong>НИЯУ МИФИ</strong> · '
        'Магистерская диссертация · Система оценки пользовательских фотографий</div>',
        unsafe_allow_html=True,
    )
