"""
Generation module for M365Mind.

Uses Ollama for local LLM inference — fast, simple, model stays loaded in memory.

Setup (one time):
    1. Install Ollama: https://ollama.com
    2. ollama pull qwen2.5:0.5b
    3. ollama serve

Model: qwen2.5:0.5b — ~390 MB, 5-15 s per response on CPU, no GPU needed.
"""

from __future__ import annotations

import logging
import os
import httpx

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# Model / limits are env-configurable so you can trade quality for speed
# without touching code. For roughly 2x faster answers on CPU, set
# M365_LLM_MODEL=qwen2.5:0.5b (after `ollama pull qwen2.5:0.5b`).
MODEL_NAME      = os.getenv("M365_LLM_MODEL", "qwen2.5:1.5b")
MAX_TOKENS      = int(os.getenv("M365_MAX_TOKENS", "256"))
# Keep the model resident in Ollama between requests (default is 5 min, which
# means an idle session pays a multi-second reload on the next query).
KEEP_ALIVE      = os.getenv("M365_KEEP_ALIVE", "30m")


def generate(system_prompt: str, user_message: str) -> str:
    """
    Generate a response via Ollama's chat API.

    Returns
    -------
    Generated text string (assistant turn only).
    """
    try:
        response = httpx.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model":    MODEL_NAME,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ],
                "stream": False,
                "keep_alive": KEEP_ALIVE,
                "options": {
                    "temperature": 0,
                    "num_predict": MAX_TOKENS,
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["message"]["content"].strip()
    except httpx.ConnectError:
        return (
            "Ollama is not running. Start it with `ollama serve` in a terminal, "
            "then try again."
        )
    except Exception as exc:
        logger.error("Generation error: %s", exc)
        return f"Generation failed: {exc}"


def prime() -> None:
    """
    Load the model into Ollama's memory ahead of the first real query, so the
    user doesn't wait for a cold model load. Called at server startup.
    """
    try:
        httpx.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "ok"}],
                "stream": False,
                "keep_alive": KEEP_ALIVE,
                "options": {"num_predict": 1},
            },
            timeout=120,
        )
        logger.info("Ollama model primed: %s", MODEL_NAME)
    except Exception as exc:
        logger.warning("Ollama prime skipped (%s)", exc)
