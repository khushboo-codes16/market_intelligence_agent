# utils/confidence.py
# ============================================================
# STEP 5: Data confidence badge helpers
# Shared between agents and the UI.
# ============================================================

from typing import Dict


# Visual badge config per level
_BADGE = {
    "high":   {"icon": "🟢", "label": "High confidence",   "color": "#1a7f37"},
    "medium": {"icon": "🟡", "label": "Medium confidence",  "color": "#9a6700"},
    "low":    {"icon": "🔴", "label": "Low confidence",     "color": "#cf222e"},
}

_TOOLTIP = {
    "high":   "Backed by substantial scraped data from the company website and news.",
    "medium": "Partially backed by scraped data; some sections inferred by the LLM.",
    "low":    "Limited scraped data — output is primarily LLM knowledge, not live data.",
}


def get_badge(level: str) -> Dict[str, str]:
    """Return icon, label, color, and tooltip for a confidence level."""
    level = level.lower() if level else "low"
    # Normalise unknown levels to "low"
    if level not in _BADGE:
        level = "low"
    badge = _BADGE[level].copy()
    badge["tooltip"] = _TOOLTIP.get(level, _TOOLTIP["low"])
    badge["level"] = level
    return badge


def confidence_html(level: str) -> str:
    """Return a small inline HTML badge suitable for st.markdown(..., unsafe_allow_html=True)."""
    b = get_badge(level)
    return (
        f'<span title="{b["tooltip"]}" style="'
        f'font-size:0.75rem; font-weight:600; color:{b["color"]}; '
        f'background:{"#e6ffed" if level=="high" else "#fff8c5" if level=="medium" else "#ffebe9"}; '
        f'border:1px solid {b["color"]}33; '
        f'padding:2px 8px; border-radius:10px; margin-left:8px; vertical-align:middle;">'
        f'{b["icon"]} {b["label"].upper()}'
        f'</span>'
    )
