"""
Lightweight client for calling a local Llama model served via Ollama.
"""

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
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
    try:
        res = requests.post(OLLAMA_URL, json=payload, timeout=120)
        res.raise_for_status()
        data = res.json()
        return data.get("response", "").strip()
    except Exception as ex:  # noqa: BLE001
        return f"(Could not get explanation from Llama: {ex})"
