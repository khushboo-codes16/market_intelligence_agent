# workflow/run_guarded.py
# ============================================================
# Guardrail-wrapped entry point for the pipeline.
# Use this instead of run_intelligence_pipeline directly.
# ============================================================

from typing import Dict, Any
from guardrails.input_guard import validate_inputs
from guardrails.pipeline_guard import check_api_key, safe_agent_run
from config.settings import GROQ_API_KEY
from utils.logger import logger


def run_pipeline_guarded(
    company_name:     str,
    company_website:  str = "",
    product_name:     str = "",
    product_scenario: str = "",
) -> Dict[str, Any]:
    """
    Full guardrail-wrapped pipeline runner.
    1. Validates API key
    2. Validates all user inputs
    3. Runs pipeline with safe error handling
    4. Returns result or clean error state
    """
    # ── GUARDRAIL 1: API Key ──────────────────────────────────
    key_valid, key_error = check_api_key(GROQ_API_KEY)
    if not key_valid:
        return _error_state(company_name, f"API Key Error: {key_error}")

    # ── GUARDRAIL 2: Input Validation ─────────────────────────
    validation = validate_inputs(
        company_name=company_name,
        company_website=company_website,
        product_name=product_name,
        product_scenario=product_scenario,
    )
    if not validation.is_valid:
        return _error_state(company_name, f"Input Error: {validation.error}")

    # Use cleaned inputs
    clean_name = validation.cleaned_name
    clean_url  = validation.cleaned_url or company_website

    logger.info(f"[GuardedPipeline] All guardrails passed. Running for: {clean_name}")

    # ── GUARDRAIL 3: Run with safe error handling ─────────────
    try:
        from workflow.graph import run_intelligence_pipeline
        result = run_intelligence_pipeline(
            company_name=clean_name,
            company_website=clean_url,
            product_name=product_name,
            product_scenario=product_scenario,
        )
        return result

    except Exception as e:
        logger.error(f"[GuardedPipeline] Unexpected failure: {e}")
        return _error_state(
            company_name,
            f"Pipeline Error: {str(e)}\n\n"
            "If this says RateLimitError → wait 60 seconds and try again.\n"
            "If this says API key → check your .env file."
        )


def _error_state(company_name: str, message: str) -> Dict[str, Any]:
    """Return a clean error state that the UI can display."""
    return {
        "input":        {"company_name": company_name},
        "status":       "failed",
        "errors":       [message],
        "agent_logs":   [f"[Guardrail] Blocked: {message}"],
        "final_report": f"# ❌ Error\n\n{message}",
        "run_duration_seconds": 0,
        "run_id": "error",
    }                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               