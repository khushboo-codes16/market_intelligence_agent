# tests/test_models.py
# ============================================================
# Unit tests: Pydantic models and state validation
# ============================================================

import pytest
from pydantic import ValidationError

from models.state import (
    ResearchInput,
    IntelligenceState,
    LeadSignal,
    LeadScore,
    CompetitorInfo,
    SWOTAnalysis,
)


# ── ResearchInput ─────────────────────────────────────────────

class TestResearchInput:
    def test_required_company_name(self):
        inp = ResearchInput(company_name="OpenAI")
        assert inp.company_name == "OpenAI"
        assert inp.company_website is None
        assert inp.product_name is None

    def test_all_fields(self):
        inp = ResearchInput(
            company_name="ElevenLabs",
            company_website="https://elevenlabs.io",
            product_name="Voice AI",
            product_scenario="AI voice agent for sales",
        )
        assert inp.company_website == "https://elevenlabs.io"

    def test_missing_company_name_raises(self):
        with pytest.raises(ValidationError):
            ResearchInput()


# ── LeadSignal ────────────────────────────────────────────────

class TestLeadSignal:
    def test_defaults(self):
        sig = LeadSignal(signal_type="hiring", description="Hiring 50 engineers")
        assert sig.confidence == "medium"
        assert sig.source == ""
        assert sig.date_mentioned == ""

    def test_all_confidence_levels(self):
        for level in ("high", "medium", "low"):
            sig = LeadSignal(signal_type="funding", description="Raised $10M", confidence=level)
            assert sig.confidence == level


# ── LeadScore ─────────────────────────────────────────────────

class TestLeadScore:
    def test_score_within_bounds(self):
        score = LeadScore(total_score=75, hiring_score=20, funding_score=25,
                         launch_score=15, expansion_score=10, partnership_score=5)
        assert 0 <= score.total_score <= 100

    def test_score_zero_default(self):
        score = LeadScore()
        assert score.total_score == 0
        assert score.hiring_score == 0

    def test_score_overflow_rejected(self):
        with pytest.raises(ValidationError):
            LeadScore(total_score=150)

    def test_sub_score_overflow_rejected(self):
        with pytest.raises(ValidationError):
            LeadScore(hiring_score=50)   # max is 25


# ── IntelligenceState ─────────────────────────────────────────

class TestIntelligenceState:
    def _make_state(self, company="TestCo"):
        return IntelligenceState(input=ResearchInput(company_name=company))

    def test_default_status(self):
        state = self._make_state()
        assert state.status == "initializing"

    def test_empty_lists_by_default(self):
        state = self._make_state()
        assert state.raw_documents == []
        assert state.lead_signals == []
        assert state.errors == []

    def test_email_draft_field_exists(self):
        state = self._make_state()
        assert hasattr(state, "outreach_email_draft")
        assert state.outreach_email_draft == ""

    def test_data_confidence_field_exists(self):
        state = self._make_state()
        assert hasattr(state, "data_confidence")
        assert isinstance(state.data_confidence, dict)

    def test_model_dump_roundtrip(self):
        state = self._make_state("Notion")
        dumped = state.model_dump()
        restored = IntelligenceState(**dumped)
        assert restored.input.company_name == "Notion"

    def test_serializes_with_competitor_info(self):
        state = self._make_state()
        state.competitor_analyses = [
            CompetitorInfo(name="Rival Co", strengths=["Fast", "Cheap"])
        ]
        dumped = state.model_dump()
        assert dumped["competitor_analyses"][0]["name"] == "Rival Co"

    def test_serializes_with_swot(self):
        state = self._make_state()
        state.swot_analysis = SWOTAnalysis(
            strengths=["Strong brand"],
            weaknesses=["High price"],
            opportunities=["New markets"],
            threats=["Competition"],
        )
        dumped = state.model_dump()
        assert "Strong brand" in dumped["swot_analysis"]["strengths"]
