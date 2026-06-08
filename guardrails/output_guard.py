# guardrails/output_guard.py
# ============================================================
# GUARDRAIL 2 — LLM Output Validation
# Validates LLM responses AFTER every call.
# Catches: empty responses, hallucinated JSON, garbage text,
#          off-topic outputs, responses that are too short.
# ============================================================

import re
import json
from typing import Any, Optional, Tuple


# ── Thresholds ────────────────────────────────────────────────
MIN_TEXT_LENGTH    = 50    # minimum chars for a valid text response
MIN_JSON_ITEMS     = 1     # minimum items in a JSON list response
HALLUCINATION_MARKERS = [  # phrases that signal hallucination
    "i don't have access",
    "i cannot browse",
    "as an ai language model",
    "i am unable to access the internet",
    "i don't have real-time",
    "my training data",
]


def validate_text_response(
    response:   str,
    field_name: str = "response",
    min_length: int = MIN_TEXT_LENGTH,
) -> Tuple[bool, str]:
    """
    Validate a free-text LLM response.
    Returns (is_valid, cleaned_response_or_error_message).
    """
    if not response or not response.strip():
        return False, f"[Guardrail] LLM returned empty {field_name}."

    cleaned = response.strip()

    if len(cleaned) < min_length:
        return False, (
            f"[Guardrail] {field_name} too short ({len(cleaned)} chars). "
            f"Min {min_length} expected."
        )

    # Check for hallucination markers
    lower = cleaned.lower()
    for marker in HALLUCINATION_MARKERS:
        if marker in lower:
            return False, (
                f"[Guardrail] {field_name} contains hallucination marker: '{marker}'. "
                "LLM refused to answer based on real data."
            )

    return True, cleaned


def validate_json_list(
    response:   str,
    field_name: str = "list",
    min_items:  int = MIN_JSON_ITEMS,
) -> Tuple[bool, list]:
    """
    Extract and validate a JSON list from LLM output.
    Returns (is_valid, list_or_empty).
    """
    if not response or not response.strip():
        return False, []

    # Try to extract JSON array even if wrapped in text
    match = re.search(r'\[.*?\]', response, re.DOTALL)
    if not match:
        return False, []

    try:
        items = json.loads(match.group())
        if not isinstance(items, list):
            return False, []
        # Filter out empty strings
        items = [str(x).strip() for x in items if str(x).strip()]
        if len(items) < min_items:
            return False, []
        return True, items
    except (json.JSONDecodeError, ValueError):
        return False, []


def validate_json_object(
    response:      str,
    required_keys: list = None,
    field_name:    str  = "object",
) -> Tuple[bool, dict]:
    """
    Extract and validate a JSON object from LLM output.
    Returns (is_valid, dict_or_empty).
    """
    if not response or not response.strip():
        return False, {}

    match = re.search(r'\{.*\}', response, re.DOTALL)
    if not match:
        return False, {}

    try:
        obj = json.loads(match.group())
        if not isinstance(obj, dict):
            return False, {}

        if required_keys:
            missing = [k for k in required_keys if k not in obj]
            if missing:
                # Object exists but missing keys — return partial
                return True, obj

        return True, obj
    except (json.JSONDecodeError, ValueError):
        # Try cleaning common issues
        raw = match.group().replace("'", '"')
        try:
            obj = json.loads(raw)
            return True, obj
        except Exception:
            return False, {}


def safe_llm_response(
    response:     str,
    fallback:     str  = "",
    field_name:   str  = "response",
    min_length:   int  = MIN_TEXT_LENGTH,
) -> str:
    """
    Return the response if valid, else return fallback string.
    Use this wherever you want silent degradation instead of crash.
    """
    is_valid, result = validate_text_response(response, field_name, min_length)
    if is_valid:
        return result
    return fallback or f"[Analysis not available for {field_name}]"