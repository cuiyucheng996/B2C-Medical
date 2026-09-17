"""Docker ASR / TTS 代理。Agent 不调用本模块。"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from configs.paths_b2c import ensure_uvgraph_root

logger = logging.getLogger(__name__)

__all__ = ["synthesize_speech", "transcribe_audio", "voice_status"]


def _cfg() -> dict[str, Any]:
    ensure_uvgraph_root()
    from src.configs.config_loader import get_yaml_config

    yaml = get_yaml_config()
    raw = yaml.get("voice") if isinstance(yaml.get("voice"), dict) else {}
    asr_url = (os.getenv("ASR_BASE_URL") or str(raw.get("asr_url") or "")).strip().rstrip("/")
    tts_url = (os.getenv("TTS_BASE_URL") or str(raw.get("tts_url") or "")).strip().rstrip("/")
    return {
        "asr_url": asr_url,
        "asr_path": str(raw.get("asr_path") or "/v1/audio/transcriptions"),
        "tts_url": tts_url,
        "tts_path": str(raw.get("tts_path") or "/v1/audio/speech"),
        "tts_voice": str(raw.get("tts_voice") or "zh"),
        "timeout": float(raw.get("timeout") or 60),
    }


def voice_status() -> dict[str, Any]:
    cfg = _cfg()
    return {
        "asr": bool(cfg["asr_url"]),
        "tts": bool(cfg["tts_url"]),
        "asr_url": cfg["asr_url"],
        "tts_url": cfg["tts_url"],
    }


def transcribe_audio(data: bytes, filename: str = "audio.webm", content_type: str = "audio/webm") -> str:
    cfg = _cfg()
    if not cfg["asr_url"]:
        raise RuntimeError("未配置 ASR_BASE_URL / voice.asr_url")
    if not data:
        raise ValueError("空音频")
    name = filename or "audio.webm"
    mime = content_type or "application/octet-stream"
    openai_url = cfg["asr_url"] + (cfg["asr_path"] if str(cfg["asr_path"]).startswith("/") else "/" + cfg["asr_path"])
    errors: list[str] = []
    try:
        raw = _post_multipart(
            openai_url,
            fields={"model": "whisper-1"},
            files={"file": (name, data, mime)},
            timeout=cfg["timeout"],
        )
        text = _extract_text(raw)
        if text:
            return text
        errors.append("openai 接口无 text")
    except Exception as exc:
        errors.append(f"openai: {exc}")
    try:
        raw = _post_json(
            cfg["asr_url"],
            {
                "mode": "offline",
                "wav_name": name,
                "wav_format": name.rsplit(".", 1)[-1] if "." in name else "webm",
                "audio": __import__("base64").b64encode(data).decode("ascii"),
            },
            timeout=cfg["timeout"],
        )
        text = _extract_text(raw)
        if text:
            return text
        errors.append("funasr 接口无 text")
    except Exception as exc:
        errors.append(f"funasr: {exc}")
    raise RuntimeError("ASR 失败: " + "; ".join(errors))


def synthesize_speech(text: str) -> tuple[bytes, str]:
    cfg = _cfg()
    if not cfg["tts_url"]:
        raise RuntimeError("未配置 TTS_BASE_URL / voice.tts_url")
    spoken = (text or "").strip()
    if not spoken:
        raise ValueError("空文本")
    openai_url = cfg["tts_url"] + (cfg["tts_path"] if str(cfg["tts_path"]).startswith("/") else "/" + cfg["tts_path"])
    errors: list[str] = []
    try:
        audio = _post_bytes(
            openai_url,
            {
                "model": "tts-1",
                "input": spoken,
                "voice": cfg["tts_voice"],
            },
            timeout=cfg["timeout"],
        )
        if audio:
            return audio, "audio/mpeg"
    except Exception as exc:
        errors.append(f"openai: {exc}")
    try:
        audio = _post_bytes(
            cfg["tts_url"].rstrip("/") + "/inference_sft",
            {"tts_text": spoken, "spk_id": cfg["tts_voice"]},
            timeout=cfg["timeout"],
        )
        if audio:
            return audio, "audio/wav"
    except Exception as exc:
        errors.append(f"cosyvoice: {exc}")
    raise RuntimeError("TTS 失败: " + "; ".join(errors))


def _extract_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload.strip()
    if not isinstance(payload, dict):
        return ""
    for key in ("text", "result", "transcription"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    data = payload.get("data")
    if isinstance(data, dict):
        return _extract_text(data)
    if isinstance(data, str):
        return data.strip()
    return ""


def _post_json(url: str, payload: dict[str, Any], *, timeout: float) -> Any:
    body = json.dumps(payload).encode("utf-8")
    req = Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
    return json.loads(_read_raw(req, timeout=timeout).decode("utf-8") or "{}")


def _post_bytes(url: str, payload: dict[str, Any], *, timeout: float) -> bytes:
    body = json.dumps(payload).encode("utf-8")
    req = Request(url, data=body, method="POST", headers={"Content-Type": "application/json", "Accept": "*/*"})
    return _read_raw(req, timeout=timeout)


def _post_multipart(
    url: str,
    *,
    fields: dict[str, str],
    files: dict[str, tuple[str, bytes, str]],
    timeout: float,
) -> Any:
    boundary = uuid.uuid4().hex
    chunks: list[bytes] = []
    for key, value in fields.items():
        chunks.append(f"--{boundary}\r\n".encode())
        chunks.append(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode())
        chunks.append(str(value).encode("utf-8") + b"\r\n")
    for key, (name, data, mime) in files.items():
        chunks.append(f"--{boundary}\r\n".encode())
        chunks.append(
            f'Content-Disposition: form-data; name="{key}"; filename="{name}"\r\n'.encode()
        )
        chunks.append(f"Content-Type: {mime}\r\n\r\n".encode())
        chunks.append(data)
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode())
    blob = b"".join(chunks)
    req = Request(
        url,
        data=blob,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    raw = _read_raw(req, timeout=timeout)
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        return {"text": raw.decode("utf-8", errors="ignore")}


def _read_raw(req: Request, *, timeout: float) -> bytes:
    try:
        with urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(detail[:400] or str(exc)) from exc
    except URLError as exc:
        raise RuntimeError(f"连不上语音服务: {exc.reason}") from exc
