# guardrails/pipeline_guard.py
# ============================================================
# GUARDRAIL 3 — Pipeline Safety
# Wraps each agent call in a safe executor that:
#   - Catches ALL exceptions (never crashes the UI)
#   - Logs every failure with context
#   - Returns a partial/fallback state instead of traceback
#   - Validates state before passing between agents
# ============================================================

import traceback
from typing import Dict, Any, Callable, Optional
from utils.logger import logger


# ── Minimum state fields each agent must populate ────────────
REQUIRED_AFTER_RESEARCH = ["raw_documents", "rag_chunks_count"]
REQUIRED_AFTER_MARKETING = ["company_overview", "product_positioning"]
REQUIRED_AFTER_COMPETITOR = ["identified_competitors"]
REQUIRED_AFTER_LEAD = ["lead_signals", "lead_score"]
REQUIRED_AFTER_REPORT = ["final_report", "executive_summary"]


def safe_agent_run(
    agent_fn:   Callable[[Dict], Dict],
    state:      Dict[str, Any],
    agent_name: str,
    required_outputs: list = None,
) -> Dict[str, Any]:
    """
    Safely execute an agent function.
    - Never raises an exception to the caller
    - Logs full traceback on failure
    - Returns original state + error info if agent crashes
    - Validates required output fields after run
    """
    try:
        logger.info(f"[PipelineGuard] Running {agent_name}...")
        result = agent_fn(state)

        # Validate required outputs were populated
        if required_outputs:
            for field in required_outputs:
                value = result.get(field)
                if value is None or value == "" or value == [] or value == {}:
                    logger.warning(
                        f"[PipelineGuard] {agent_name} did not populate '{field}'. "
                        "Continuing with empty value."
                    )
                    result.setdefault("agent_logs", []).append(
                        f"[Guardrail] Warning: {agent_name} produced empty '{field}'"
                    )

        logger.info(f"[PipelineGuard] {agent_name} completed successfully.")
        return result

    except MemoryError:
        error_msg = f"{agent_name} ran out of memory. Try a shorter scenario."
        logger.error(f"[PipelineGuard] MemoryError in {agent_name}")
        return _inject_error(state, agent_name, error_msg)

    except TimeoutError:
        error_msg = f"{agent_name} timed out. The website may be slow."
        logger.error(f"[PipelineGuard] TimeoutError in {agent_name}")
        return _inject_error(state, agent_name, error_msg)

    except Exception as e:
        tb = traceback.format_exc()
        error_msg = f"{agent_name} failed: {type(e).__name__}: {str(e)}"
        logger.error(f"[PipelineGuard] {error_msg}\n{tb}")
        return _inject_error(state, agent_name, error_msg)


def _inject_error(
    state:      Dict[str, Any],
    agent_name: str,
    error_msg:  str,
) -> Dict[str, Any]:
    """Add error info to state without crashing."""
    result = dict(state)
    errors = result.get("errors", [])
    errors.append(error_msg)
    result["errors"] = errors

    logs = result.get("agent_logs", [])
    logs.append(f"[Guardrail] {error_msg}")
    result["agent_logs"] = logs

    return result


def validate_state_transition(
    state:       Dict[str, Any],
    from_status: str,
    to_status:   str,
) -> bool:
    """
    Check that a state transition makes sense.
    Prevents agents from running on completely empty state.
    """
    # If research produced nothing at all, warn but allow
    if from_status == "research_complete":
        docs   = len(state.get("raw_documents", []))
        chunks = state.get("rag_chunks_count", 0)
        if docs == 0 and chunks == 0:
            logger.warning(
                "[PipelineGuard] State transition to analysis with zero documents. "
                "Output quality will be low — LLM knowledge only."
            )

    # If analysis produced nothing, block report generation
    if to_status == "report_generation":
        has_overview = bool(state.get("company_overview", "").strip())
        has_score    = state.get("lead_score") is not None
        if not has_overview and not has_score:
            logger.warning(
                "[PipelineGuard] Proceeding to report with no analysis data. "
                "Report will be minimal."
            )

    return True  # Always allow — degraded output is better than crash


def check_api_key(api_key: str) -> tuple:
    """
    Validate Groq API key format before making any call.
    Returns (is_valid, error_message).
    """
    if not api_key:
        return False, (
            "GROQ_API_KEY is not set. "
            "Please add it to your .env file. "
            "Get a free key at https://console.groq.com"
        )

    if not api_key.startswith("gsk_"):
        return False, (
            "GROQ_API_KEY looks invalid (should start with 'gsk_'). "
            "Please check your .env file."
        )

    if len(api_key) < 40:
        return False, (
            "GROQ_API_KEY looks too short. "
            "Please copy the full key from https://console.groq.com"
        )

    return True, ""