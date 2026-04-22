#!/usr/bin/env python3
"""
dataset_generation/_api_client.py
──────────────────────────────────
Общие вспомогательные функции для работы с API ruGPT.io, используемые всеми скриптами генерации.
Импортируйте этот модуль вместо дублирования HTTP-логики в каждом скрипте.

Пример использования
────────────────────
    from _api_client import get_api_key, submit_text2img, submit_img2img, poll_result, download_image
"""

import json
import logging
import os
import time
from pathlib import Path

import requests

log = logging.getLogger(__name__)

# ── Константы API ─────────────────────────────────────────────────────────────
API_BASE       = "https://api.rugpt.io/api/private/b2b"
POLL_INTERVAL  = 15  # интервал опроса статуса задачи (секунды)
POLL_MAX_TRIES = 180 # максимальное количество попыток опроса


# ── Аутентификация ────────────────────────────────────────────────────────────

def get_api_key() -> str:
    """
    Чтение API-ключа из переменной окружения RUGPT_API_KEY.

    Возвращает
    ──────────
    str
        API-ключ для доступа к ruGPT.io.

    Исключения
    ──────────
    EnvironmentError
        Если переменная окружения не установлена.
    """
    key = os.environ.get("RUGPT_API_KEY", "").strip()
    if not key:
        raise EnvironmentError(
            "Переменная окружения RUGPT_API_KEY не установлена.\n"
            "Выполните:  export RUGPT_API_KEY='your_key_here'\n"
            "Получить ключ можно на: https://rugpt.io"
        )
    return key


# ── Генерация изображений из текста (text-to-image) ───────────────────────────

def submit_text2img(
    prompt: str,
    api_key: str,
    model: str,
    aspect_ratio: str = "3:2",
    resolution: str | None = None,
    enhance_prompt: bool = False,
    max_retries: int = 5,
) -> tuple[str | None, str | None]:
    """
    Отправка задачи на генерацию изображения из текстового промпта.

    Параметры
    ─────────
    prompt : str
        Текстовое описание желаемого изображения.
    api_key : str
        API-ключ для аутентификации.
    model : str
        Идентификатор модели генерации (например, "flux-2/pro", "z-image").
    aspect_ratio : str, optional
        Соотношение сторон изображения (по умолчанию "3:2").
    resolution : str | None, optional
        Разрешение изображения (например, "2K"). Если None, используется значение по умолчанию модели.
    enhance_prompt : bool, optional
        Автоматическое улучшение промпта моделью (по умолчанию False).
    max_retries : int, optional
        Максимальное количество повторных попыток при сетевых ошибках (по умолчанию 5).

    Возвращает
    ──────────
    tuple[str | None, str | None]
        (job_uuid, status) — UUID задачи и её начальный статус, или (None, None) при неудаче.
    """
    url     = f"{API_BASE}/image/generation"
    headers = {"x-rugpt-key": api_key, "Content-Type": "application/json"}
    params  = {
        "aspectRatio":   aspect_ratio,
        "attachedFiles": [],
        "enhancePrompt": enhance_prompt,
    }
    if resolution:
        params["resolution"] = resolution

    payload = {"model": model, "prompt": prompt, "params": params}
    return _post_job(url, headers, payload, max_retries)


# ── Редактирование изображений (image-to-image, tempered) ─────────────────────

def upload_file(filepath: Path, api_key: str) -> str | None:
    """
    Загрузка локального файла изображения на сервер ruGPT.io.

    Параметры
    ─────────
    filepath : Path
        Путь к локальному файлу изображения.
    api_key : str
        API-ключ для аутентификации.

    Возвращает
    ──────────
    str | None
        UUID загруженного файла для использования в вызовах img2img, или None при ошибке.
    """
    # Таблица соответствия MIME-типов
    _MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
             ".png": "image/png",  ".webp": "image/webp"}
    mime = _MIME.get(filepath.suffix.lower(), "image/jpeg")
    url  = f"{API_BASE}/file/upload"
    headers = {"x-rugpt-key": api_key}
    
    try:
        with filepath.open("rb") as fh:
            resp = requests.post(
                url, headers=headers,
                files={"file": (filepath.name, fh, mime)},
                timeout=60,
            )
        if resp.status_code not in (200, 201):
            log.error("Ошибка загрузки: HTTP %d — %s", resp.status_code, resp.text[:200])
            return None
            
        data = resp.json().get("data", {})
        uuid = data.get("uuid")
        log.info("  Загружено: %s → uuid=%s  стоимость=%s",
                 filepath.name, uuid, data.get("attachmentCost", "?"))
        return uuid
        
    except Exception as exc:
        log.error("Ошибка при загрузке: %s", exc)
        return None


def submit_img2img(
    prompt: str,
    file_uuid: str,
    api_key: str,
    model: str,
    aspect_ratio: str = "3:2",
    resolution: str | None = None,
    enhance_prompt: bool = False,
    max_retries: int = 5,
) -> tuple[str | None, str | None]:
    """
    Отправка задачи на редактирование изображения (image-to-image generation).

    Параметры
    ─────────
    prompt : str
        Текстовое описание желаемых изменений.
    file_uuid : str
        UUID файла, возвращённый функцией :func:`upload_file`.
    api_key : str
        API-ключ для аутентификации.
    model : str
        Идентификатор модели генерации.
    aspect_ratio : str, optional
        Соотношение сторон результата (по умолчанию "3:2").
    resolution : str | None, optional
        Разрешение результата. Если None, используется значение по умолчанию модели.
    enhance_prompt : bool, optional
        Автоматическое улучшение промпта (по умолчанию False).
    max_retries : int, optional
        Максимальное количество повторных попыток (по умолчанию 5).

    Возвращает
    ──────────
    tuple[str | None, str | None]
        (job_uuid, status) — UUID задачи и её начальный статус, или (None, None) при неудаче.
    """
    url     = f"{API_BASE}/image/generation"
    headers = {"x-rugpt-key": api_key, "Content-Type": "application/json"}
    params  = {
        "aspectRatio":   aspect_ratio,
        "attachedFiles": [file_uuid],
        "enhancePrompt": enhance_prompt,
    }
    if resolution:
        params["resolution"] = resolution

    payload = {"model": model, "prompt": prompt, "params": params}
    return _post_job(url, headers, payload, max_retries)


# ── Опрос статуса задачи ──────────────────────────────────────────────────────

def poll_result(job_uuid: str, api_key: str) -> dict:
    """
    Опрос статуса задачи генерации до её завершения, неудачи или таймаута.

    Параметры
    ─────────
    job_uuid : str
        UUID задачи, возвращённый функциями submit_text2img или submit_img2img.
    api_key : str
        API-ключ для аутентификации.

    Возвращает
    ──────────
    dict
        Словарь с ключами:
        - status: "completed", "failed" или "timeout"
        - urls: список URL сгенерированных изображений (или None)
        - price: стоимость генерации в коинах
        - error: сообщение об ошибке (опционально, при статусе "failed")
    """
    url     = f"{API_BASE}/image/generation/jobs/{job_uuid}"
    headers = {"x-rugpt-key": api_key}

    for attempt in range(POLL_MAX_TRIES):
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code != 200:
                time.sleep(POLL_INTERVAL)
                continue

            data   = resp.json()["data"]
            status = data["status"]

            if status == "completed":
                return {"urls": data.get("urls"), "price": data.get("price"), "status": "completed"}
            if status == "failed":
                return {"urls": None, "price": data.get("price"),
                        "status": "failed", "error": data.get("error")}

            # Адаптивный интервал ожидания
            wait = POLL_INTERVAL if attempt < 5 else min(POLL_INTERVAL * 2, 30)
            if attempt % 4 == 0:
                log.info("  статус=%s  (попытка %d, ~%dс прошло)",
                         status, attempt, attempt * POLL_INTERVAL)
            time.sleep(wait)

        except Exception as exc:
            log.warning("Ошибка опроса: %s", exc)
            time.sleep(POLL_INTERVAL)

    return {"urls": None, "status": "timeout"}


# ── Загрузка изображения ──────────────────────────────────────────────────────

def download_image(img_url: str, filepath: Path) -> int:
    """
    Загрузка изображения по URL и сохранение в указанный файл.

    Параметры
    ─────────
    img_url : str
        URL изображения для загрузки.
    filepath : Path
        Путь к файлу для сохранения.

    Возвращает
    ──────────
    int
        Размер загруженного файла в байтах.

    Исключения
    ──────────
    requests.HTTPError
        При ошибках HTTP (статус-коды 4xx, 5xx).
    """
    resp = requests.get(img_url, timeout=120)
    resp.raise_for_status()
    filepath.write_bytes(resp.content)
    return len(resp.content)


# ── Внутренние вспомогательные функции ────────────────────────────────────────

def _post_job(url, headers, payload, max_retries):
    """
    Внутренняя функция для отправки POST-запроса с повторными попытками.

    Параметры
    ─────────
    url : str
        URL эндпоинта API.
    headers : dict
        HTTP-заголовки запроса.
    payload : dict
        JSON-тело запроса.
    max_retries : int
        Максимальное количество повторных попыток.

    Возвращает
    ──────────
    tuple[str | None, str | None]
        (job_uuid, status) или (None, None) при неудаче после всех попыток.
    """
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            
            # Повторная попытка при временных серверных ошибках
            if resp.status_code in (502, 503, 504):
                wait = (attempt + 1) * 10
                log.warning("HTTP %d, повтор %d/%d через %dс", 
                           resp.status_code, attempt + 1, max_retries, wait)
                time.sleep(wait)
                continue
                
            if resp.status_code not in (200, 201):
                log.error("Ошибка отправки: HTTP %d — %s", 
                         resp.status_code, resp.text[:300])
                return None, None
                
            data = resp.json()["data"]
            return data["uuid"], data["status"]
            
        except requests.exceptions.Timeout:
            wait = (attempt + 1) * 10
            log.warning("Таймаут, повтор %d/%d через %dс", 
                       attempt + 1, max_retries, wait)
            time.sleep(wait)
            
        except Exception as exc:
            log.error("Ошибка запроса: %s", exc)
            time.sleep(10)
            
    return None, None
