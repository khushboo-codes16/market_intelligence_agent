# reports/report_saver.py
# ============================================================
# Save intelligence reports to disk
# ============================================================
# STEP 3: PDF export via weasyprint
# STEP 4: JSON run history for compare / load past runs
# ============================================================

import re
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from config.settings import REPORTS_DIR
from utils.logger import logger


# ── Markdown → PDF ────────────────────────────────────────────

def generate_pdf_bytes(report_md: str) -> Optional[bytes]:
    """
    Convert a markdown report to PDF bytes.
    Returns None if weasyprint or markdown are not installed.
    """
    try:
        import markdown as md_lib
        from weasyprint import HTML, CSS

        html_body = md_lib.markdown(
            report_md,
            extensions=["tables", "fenced_code", "nl2br"],
        )
        full_html = f"""
        <html>
        <head><meta charset="utf-8"></head>
        <body style="
            font-family: 'Helvetica Neue', Arial, sans-serif;
            font-size: 13px;
            line-height: 1.6;
            color: #222;
            max-width: 900px;
            margin: 0 auto;
            padding: 40px;
        ">
        {html_body}
        </body>
        </html>
        """
        css = CSS(string="""
            h1 { color: #0f3460; border-bottom: 2px solid #0f3460; padding-bottom: 6px; }
            h2 { color: #16213e; margin-top: 1.5em; }
            h3 { color: #333; }
            table { border-collapse: collapse; width: 100%; margin: 1em 0; }
            th { background: #0f3460; color: white; padding: 8px 12px; text-align: left; }
            td { padding: 6px 12px; border-bottom: 1px solid #ddd; }
            tr:nth-child(even) { background: #f5f7ff; }
            code { background: #f4f4f4; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }
            pre { background: #1e1e1e; color: #ccc; padding: 12px; border-radius: 6px; }
            blockquote { border-left: 4px solid #0f3460; margin: 0; padding-left: 16px; color: #555; }
        """)
        return HTML(string=full_html).write_pdf(stylesheets=[css])

    except ImportError:
        logger.warning("[PDF] weasyprint or markdown not installed — PDF export unavailable.")
        return None
    except Exception as e:
        logger.error(f"[PDF] Generation failed: {e}")
        return None


# ── Run History ───────────────────────────────────────────────

def _history_path() -> Path:
    return Path(REPORTS_DIR) / "run_history.json"


def load_run_history() -> List[Dict[str, Any]]:
    """Load all past run summaries from the history file."""
    path = _history_path()
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def _append_to_history(entry: Dict[str, Any]) -> None:
    """Append a run summary to the history file."""
    history = load_run_history()
    history.insert(0, entry)          # newest first
    history = history[:50]            # keep last 50 runs
    try:
        _history_path().write_text(
            json.dumps(history, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning(f"[History] Could not save run history: {e}")


# ── Main Save ─────────────────────────────────────────────────

def save_report(state: Dict[str, Any]) -> str:
    """
    Save the final report to disk:
      - <company>_<ts>.md   — full markdown report
      - run_history.json    — lightweight summary for history panel

    Returns the .md file path.
    """
    company = state.get("input", {}).get("company_name", "unknown")
    report_text = state.get("final_report", "")

    if not report_text:
        logger.warning("No report to save.")
        return ""

    safe_company = re.sub(r"[^a-z0-9_]", "_", company.lower().strip())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"intel_{safe_company}_{timestamp}.md"
    filepath = Path(REPORTS_DIR) / filename

    Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)
    filepath.write_text(report_text, encoding="utf-8")
    logger.info(f"Report saved: {filepath}")

    # ── Append lightweight summary to run history ─────────────
    lead_score = state.get("lead_score") or {}
    if isinstance(lead_score, dict):
        score_val = lead_score.get("total_score", 0)
    else:
        score_val = getattr(lead_score, "total_score", 0)

    signals = state.get("lead_signals", [])
    competitors = state.get("identified_competitors", [])

    history_entry = {
        "company": company,
        "timestamp": timestamp,
        "datetime": datetime.now().strftime("%b %d, %Y %H:%M"),
        "lead_score": score_val,
        "competitor_count": len(competitors),
        "signal_count": len(signals),
        "competitors": competitors[:5],
        "filepath": str(filepath),
        "rag_chunks": state.get("rag_chunks_count", 0),
    }
    _append_to_history(history_entry)

    return str(filepath)


def load_report(filepath: str) -> str:
    """Load a saved markdown report."""
    return Path(filepath).read_text(encoding="utf-8")
