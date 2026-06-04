# tests/test_lead_scoring.py
# ============================================================
# Unit tests: lead scoring logic (no LLM calls)
# ============================================================

import pytest
from models.state import LeadSignal, LeadScore
from agents.lead_generation_agent import _compute_lead_score


# ── Helper ────────────────────────────────────────────────────

def make_signal(stype: str, confidence: str = "high") -> LeadSignal:
    return LeadSignal(signal_type=stype, description=f"Test {stype} signal", confidence=confidence)


# ── Score Calculation ─────────────────────────────────────────

class TestLeadScoring:

    def test_empty_signals_zero_score(self):
        score = _compute_lead_score([])
        assert score.total_score == 0

    def test_single_high_funding_signal(self):
        signals = [make_signal("funding", "high")]
        score = _compute_lead_score(signals)
        assert score.funding_score > 0
        assert score.total_score == score.funding_score

    def test_high_confidence_beats_low(self):
        high_score = _compute_lead_score([make_signal("hiring", "high")])
        low_score = _compute_lead_score([make_signal("hiring", "low")])
        assert high_score.hiring_score >= low_score.hiring_score

    def test_multiple_signal_types_accumulate(self):
        signals = [
            make_signal("funding", "high"),
            make_signal("hiring", "high"),
            make_signal("launch", "high"),
        ]
        score = _compute_lead_score(signals)
        assert score.funding_score > 0
        assert score.hiring_score > 0
        assert score.launch_score > 0

    def test_score_never_exceeds_100(self):
        signals = [make_signal(t, "high") for t in
                   ["funding"] * 10 + ["hiring"] * 10 + ["launch"] * 10]
        score = _compute_lead_score(signals)
        assert score.total_score <= 100

    def test_sub_scores_sum_equals_total(self):
        signals = [
            make_signal("funding", "high"),
            make_signal("hiring", "medium"),
            make_signal("launch", "low"),
            make_signal("expansion", "high"),
            make_signal("partnership", "medium"),
        ]
        score = _compute_lead_score(signals)
        expected = (score.hiring_score + score.funding_score + score.launch_score
                    + score.expansion_score + score.partnership_score)
        assert score.total_score == expected

    def test_unknown_signal_type_ignored(self):
        signals = [LeadSignal(signal_type="unknown_type", description="mystery", confidence="high")]
        score = _compute_lead_score(signals)
        assert score.total_score == 0

    def test_justification_mentions_active_categories(self):
        signals = [make_signal("funding", "high"), make_signal("hiring", "medium")]
        score = _compute_lead_score(signals)
        lower_just = score.justification.lower()
        assert "funding" in lower_just
        assert "hiring" in lower_just

    def test_confidence_multipliers_ordered(self):
        scores = {}
        for level in ("high", "medium", "low"):
            s = _compute_lead_score([make_signal("funding", level)])
            scores[level] = s.funding_score
        assert scores["high"] >= scores["medium"] >= scores["low"]

    def test_signals_list_attached_to_score(self):
        signals = [make_signal("launch", "high")]
        score = _compute_lead_score(signals)
        assert len(score.signals) == 1
        assert score.signals[0].signal_type == "launch"
