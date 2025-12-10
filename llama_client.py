"""
Lightweight client for calling a local Llama model served via Ollama.
"""

import os
import requests

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_URL = os.getenv("OLLAMA_URL", DEFAULT_OLLAMA_URL)
LLAMA_MODEL = "llama3.2"


def call_local_llama(prompt: str, max_tokens: int = 256) -> str:
    """
    Call local Llama via Ollama Docker and return a single response string.
    """
    payload = {
        "model": LLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens},
    }

    # Try the configured URL first; if it fails and is not localhost, fall back to localhost.
    candidates = [OLLAMA_URL]
    if OLLAMA_URL != DEFAULT_OLLAMA_URL:
        candidates.append(DEFAULT_OLLAMA_URL)

    last_err = None
    for url in candidates:
        try:
            res = requests.post(url, json=payload, timeout=120)
            res.raise_for_status()
            data = res.json()
            return data.get("response", "").strip()
        except Exception as ex:  # noqa: BLE001
            last_err = ex
    return (
        "(Could not get explanation from Llama: "
        f"{last_err}. Ensure Ollama is running and model '{LLAMA_MODEL}' is pulled.)"
    )
