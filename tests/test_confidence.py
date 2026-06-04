# tests/test_confidence.py
# ============================================================
# Unit tests: data confidence badge helpers
# ============================================================

import pytest
from utils.confidence import get_badge, confidence_html


class TestGetBadge:
    def test_high_returns_green(self):
        b = get_badge("high")
        assert b["icon"] == "🟢"
        assert "high" in b["label"].lower()
        assert b["level"] == "high"

    def test_medium_returns_yellow(self):
        b = get_badge("medium")
        assert b["icon"] == "🟡"

    def test_low_returns_red(self):
        b = get_badge("low")
        assert b["icon"] == "🔴"

    def test_unknown_level_defaults_to_low(self):
        b = get_badge("nonexistent")
        assert b["level"] == "low"

    def test_empty_string_defaults_to_low(self):
        b = get_badge("")
        assert b["level"] == "low"

    def test_none_defaults_to_low(self):
        b = get_badge(None)
        assert b["level"] == "low"

    def test_tooltip_present(self):
        for level in ("high", "medium", "low"):
            b = get_badge(level)
            assert len(b["tooltip"]) > 10


class TestConfidenceHtml:
    def test_returns_string(self):
        result = confidence_html("high")
        assert isinstance(result, str)

    def test_contains_level_text(self):
        result = confidence_html("high")
        assert "HIGH" in result or "high" in result.lower()

    def test_valid_html_span(self):
        result = confidence_html("medium")
        assert "<span" in result
        assert "</span>" in result

    def test_all_levels_produce_output(self):
        for level in ("high", "medium", "low"):
            result = confidence_html(level)
            assert len(result) > 20

    def test_high_uses_green_background(self):
        result = confidence_html("high")
        assert "#e6ffed" in result

    def test_medium_uses_yellow_background(self):
        result = confidence_html("medium")
        assert "#fff8c5" in result

    def test_low_uses_red_background(self):
        result = confidence_html("low")
        assert "#ffebe9" in result
