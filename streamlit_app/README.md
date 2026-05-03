# streamlit_app/

Демонстрационный стенд дообученной модели SIDA-7B для проекта
*«Система оценки пользовательских фотографий в туристических сервисах»*.
Использует тот же пайплайн инференса, что и `detection/inference.py`,
но оборачивает его в интерактивный веб-интерфейс.

## Содержимое

```
streamlit_app/
├── app.py              # Главное Streamlit-приложение
├── sida_inference.py   # Адаптер инференса (повторяет detection/inference.py)
├── metrics.py          # Расчёт метрик (повторяет detection/eval_metrics.py)
├── requirements.txt    # Дополнительные зависимости поверх sida_modern
└── README.md
```

## Возможности

- **Одиночный анализ** — загрузка одного изображения, мгновенный
  вердикт (REAL / FAKE / TAMPERED), распределение вероятностей по
  классам, визуализация маски подделанной области.
- **Пакетный анализ** — multi-upload или ZIP-архив, KPI-сводка,
  графики (распределение классов, уверенности, времени), галерея
  изображений и масок, экспорт в CSV.
- **Метрики качества** — Overall Accuracy, Macro F1, Precision /
  Recall / F1 по классам, Confusion Matrix (при наличии ground-truth
  разметки).

## Использование

Из корня репозитория:

```bash
conda activate sida_modern
pip install -r streamlit_app/requirements.txt

cd streamlit_app/
streamlit run app.py
```

Приложение откроется на `http://localhost:8501`.

## Конфигурация

В боковой панели приложения (значок ↗ слева вверху) можно настроить:

- **Демо-режим** — заглушка без загрузки реальной модели для проверки UI без GPU.
- **Корень проекта** — путь к корню репозитория.
- **Веса SIDA-7B** — путь к директории `ck/SIDA-7B/`.
- **Дообученная голова** — путь к `ck/cls_head_new.pth`.
- **Точность** — `bf16` (рекомендовано), `fp16` или `fp32`.

По умолчанию используются те же пути, что и в `detection/inference.py`:
`../ck/SIDA-7B`, `../ck/cls_head_new.pth`.

## Требования

- CUDA-совместимый GPU c минимум **24 GB VRAM** (для загрузки SIDA-7B в bf16).
- Установленные пакеты `model/` и `utils/` из основного репозитория.

## Безопасный запуск

Если основная среда `sida_modern` используется для обучения, рекомендуется
работать в её клоне:

```bash
conda create --name sida_modern_app --clone sida_modern
conda activate sida_modern_app
pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/app.py
```
