#!/usr/bin/env python3
"""
dataset_generation/_api_client.py
──────────────────────────────────
Shared ruGPT.io API helpers used by all generation scripts.
Import this module instead of duplicating HTTP logic in each script.

    from _api_client import get_api_key, submit_text2img, submit_img2img, poll_result, download_image
"""

import json
import logging
import os
import time
from pathlib import Path

import requests

log = logging.getLogger(__name__)

API_BASE       = "https://api.rugpt.io/api/private/b2b"
POLL_INTERVAL  = 15
POLL_MAX_TRIES = 180


# ── Authentication ─────────────────────────────────────────────────────────────

def get_api_key() -> str:
    """Read the API key from the RUGPT_API_KEY environment variable.

    Raises
    ------
    EnvironmentError
        If the variable is not set.
    """
    key = os.environ.get("RUGPT_API_KEY", "").strip()
    if not key:
        raise EnvironmentError(
            "RUGPT_API_KEY environment variable is not set.\n"
            "Run:  export RUGPT_API_KEY='your_key_here'\n"
            "Get your key at: https://rugpt.io"
        )
    return key


# ── Text-to-image generation ───────────────────────────────────────────────────

def submit_text2img(
    prompt: str,
    api_key: str,
    model: str,
    aspect_ratio: str = "3:2",
    resolution: str | None = None,
    enhance_prompt: bool = False,
    max_retries: int = 5,
) -> tuple[str | None, str | None]:
    """Submit a text-to-image generation job.

    Returns
    -------
    (job_uuid, status) or (None, None) on failure.
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


# ── Image editing (tempered) ───────────────────────────────────────────────────

def upload_file(filepath: Path, api_key: str) -> str | None:
    """Upload a local image file and return its UUID for use in img2img calls."""
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
            log.error("Upload failed: HTTP %d — %s", resp.status_code, resp.text[:200])
            return None
        data = resp.json().get("data", {})
        uuid = data.get("uuid")
        log.info("  Uploaded: %s → uuid=%s  cost=%s",
                 filepath.name, uuid, data.get("attachmentCost", "?"))
        return uuid
    except Exception as exc:
        log.error("Upload error: %s", exc)
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
    """Submit an image-editing (image-to-image) generation job.

    Parameters
    ----------
    file_uuid:
        UUID returned by :func:`upload_file`.
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


# ── Polling ────────────────────────────────────────────────────────────────────

def poll_result(job_uuid: str, api_key: str) -> dict:
    """Poll a generation job until it completes, fails, or times out.

    Returns
    -------
    dict with keys: status, urls, price, error (optional)
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

            wait = POLL_INTERVAL if attempt < 5 else min(POLL_INTERVAL * 2, 30)
            if attempt % 4 == 0:
                log.info("  status=%s  (attempt %d, ~%ds elapsed)",
                         status, attempt, attempt * POLL_INTERVAL)
            time.sleep(wait)

        except Exception as exc:
            log.warning("Poll error: %s", exc)
            time.sleep(POLL_INTERVAL)

    return {"urls": None, "status": "timeout"}


# ── Download ───────────────────────────────────────────────────────────────────

def download_image(img_url: str, filepath: Path) -> int:
    """Download an image URL to *filepath* and return the byte size."""
    resp = requests.get(img_url, timeout=120)
    resp.raise_for_status()
    filepath.write_bytes(resp.content)
    return len(resp.content)


# ── Internal ───────────────────────────────────────────────────────────────────

def _post_job(url, headers, payload, max_retries):
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            if resp.status_code in (502, 503, 504):
                wait = (attempt + 1) * 10
                log.warning("HTTP %d, retry %d/%d in %ds", resp.status_code, attempt + 1, max_retries, wait)
                time.sleep(wait)
                continue
            if resp.status_code not in (200, 201):
                log.error("Submit failed: HTTP %d — %s", resp.status_code, resp.text[:300])
                return None, None
            data = resp.json()["data"]
            return data["uuid"], data["status"]
        except requests.exceptions.Timeout:
            wait = (attempt + 1) * 10
            log.warning("Timeout, retry %d/%d in %ds", attempt + 1, max_retries, wait)
            time.sleep(wait)
        except Exception as exc:
            log.error("Request error: %s", exc)
            time.sleep(10)
    return None, None
