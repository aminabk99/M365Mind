"""
Generation module for M365Mind.

Uses Qwen/Qwen2.5-1.5B-Instruct via HuggingFace transformers.
~3 GB download, fast on CPU (10-25 s per response), no GPU required.

For GPU users: swap MODEL_ID to "microsoft/Phi-3.5-mini-instruct" for
higher quality at the cost of a 7.6 GB download and GPU VRAM.

Model cache: ~/.cache/huggingface
"""

from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 512


@lru_cache(maxsize=1)
def _get_pipeline():
    import torch
    from transformers import pipeline

    logger.info("Loading generation model: %s", MODEL_ID)

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    pipe = pipeline(
        "text-generation",
        model=MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    logger.info("Generation model ready.")
    return pipe


def generate(system_prompt: str, user_message: str) -> str:
    """
    Generate a response given a system prompt and user message.

    Parameters
    ----------
    system_prompt : instruction context for the model
    user_message  : the user's query with retrieved context

    Returns
    -------
    Generated text string (assistant turn only).
    """
    pipe = _get_pipeline()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_message},
    ]

    result = pipe(
        messages,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,   # deterministic — important for governance answers
        temperature=None,
        top_p=None,
        return_full_text=False,
    )

    # transformers pipeline with return_full_text=False returns only the
    # assistant turn directly as a string
    output = result[0]["generated_text"]
    if isinstance(output, list):
        # Chat format: last message is the assistant turn
        return output[-1].get("content", "")
    return str(output)
