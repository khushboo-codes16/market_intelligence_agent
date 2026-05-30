# reports/report_saver.py
# ============================================================
# Save intelligence reports to disk
# ============================================================

import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from config.settings import REPORTS_DIR
from utils.logger import logger


def save_report(state: Dict[str, Any]) -> str:
    """
    Save the final report to the reports directory.
    Returns the file path.
    """
    company = state.get("input", {}).get("company_name", "unknown")
    report_text = state.get("final_report", "")

    if not report_text:
        logger.warning("No report to save.")
        return ""

    # Build filename
    safe_company = re.sub(r"[^a-z0-9_]", "_", company.lower().strip())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"intel_{safe_company}_{timestamp}.md"
    filepath = Path(REPORTS_DIR) / filename

    filepath.write_text(report_text, encoding="utf-8")
    logger.info(f"Report saved: {filepath}")
    return str(filepath)


def load_report(filepath: str) -> str:
    """Load a saved report."""
    return Path(filepath).read_text(encoding="utf-8")
