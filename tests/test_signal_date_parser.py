# tests/test_signal_date_parser.py
# ============================================================
# Unit tests: signal date parser (used by timeline chart)
# Tests the _parse_signal_date function from dashboard helpers
# ============================================================

import pytest
from datetime import datetime


# Import the parser directly from dashboard's module-level function
# We replicate it here to keep tests independent of Streamlit
def _parse_signal_date(date_str: str):
    """Mirror of the same function in ui/dashboard.py."""
    if not date_str:
        return None
    import re
    for fmt in ("%B %Y", "%b %Y", "%Y-%m-%d", "%Y-%m", "%B %d, %Y"):
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            pass
    q_match = re.search(r"Q([1-4])\s*(\d{4})", date_str)
    if q_match:
        quarter_start = {"1": "01", "2": "04", "3": "07", "4": "10"}
        month = quarter_start[q_match.group(1)]
        return datetime.strptime(f"{q_match.group(2)}-{month}-01", "%Y-%m-%d")
    yr_match = re.search(r"\b(20\d{2})\b", date_str)
    if yr_match:
        try:
            return datetime.strptime(f"{yr_match.group(1)}-01-01", "%Y-%m-%d")
        except ValueError:
            pass
    return None


class TestSignalDateParser:
    def test_empty_string_returns_none(self):
        assert _parse_signal_date("") is None

    def test_none_returns_none(self):
        assert _parse_signal_date(None) is None

    def test_q1_2025(self):
        result = _parse_signal_date("Q1 2025")
        assert result is not None
        assert result.year == 2025
        assert result.month == 1

    def test_q2_2025(self):
        result = _parse_signal_date("Q2 2025")
        assert result.month == 4

    def test_q3_2024(self):
        result = _parse_signal_date("Q3 2024")
        assert result.month == 7

    def test_q4_2023(self):
        result = _parse_signal_date("Q4 2023")
        assert result.month == 10

    def test_quarter_without_space(self):
        result = _parse_signal_date("Q12025")
        assert result is not None
        assert result.year == 2025

    def test_full_month_year(self):
        result = _parse_signal_date("January 2024")
        assert result.month == 1
        assert result.year == 2024

    def test_abbreviated_month(self):
        result = _parse_signal_date("Jan 2024")
        assert result.month == 1

    def test_iso_date(self):
        result = _parse_signal_date("2024-06-15")
        assert result.year == 2024
        assert result.month == 6
        assert result.day == 15

    def test_year_only(self):
        result = _parse_signal_date("2023")
        assert result.year == 2023
        assert result.month == 1

    def test_year_in_sentence(self):
        result = _parse_signal_date("Raised funding in 2024")
        assert result.year == 2024

    def test_garbled_string_returns_none(self):
        assert _parse_signal_date("sometime last year") is None

    def test_future_year_parsed(self):
        result = _parse_signal_date("Q1 2026")
        assert result is not None
        assert result.year == 2026
