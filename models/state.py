# models/state.py
# ============================================================
# Pydantic models for LangGraph state and agent outputs
# ============================================================

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Input Schema ─────────────────────────────────────────────

class ResearchInput(BaseModel):
    company_name: str = Field(..., description="Name of the target company")
    company_website: Optional[str] = Field(None, description="Company website URL (optional)")
    product_name: Optional[str] = Field(None, description="Specific product to analyze (optional)")
    product_scenario: Optional[str] = Field(None, description="Product scenario description (optional)")


# ── Intermediate Data Models ──────────────────────────────────

class ScrapedDocument(BaseModel):
    url: str
    title: str
    text: str
    meta_description: str = ""
    source_type: str = "website"  # website | news | review | careers


class CompetitorInfo(BaseModel):
    name: str
    website: str = ""
    description: str = ""
    positioning: str = ""
    value_proposition: str = ""
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    messaging_comparison: str = ""


class LeadSignal(BaseModel):
    signal_type: str  # hiring | funding | launch | expansion | partnership
    description: str
    source: str = ""
    date_mentioned: str = ""
    confidence: str = "medium"  # low | medium | high


class LeadScore(BaseModel):
    total_score: int = Field(0, ge=0, le=100)
    hiring_score: int = Field(0, ge=0, le=25)
    funding_score: int = Field(0, ge=0, le=30)
    launch_score: int = Field(0, ge=0, le=20)
    expansion_score: int = Field(0, ge=0, le=15)
    partnership_score: int = Field(0, ge=0, le=10)
    justification: str = ""
    signals: List[LeadSignal] = Field(default_factory=list)


class SWOTAnalysis(BaseModel):
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    opportunities: List[str] = Field(default_factory=list)
    threats: List[str] = Field(default_factory=list)


# ── Main Graph State ──────────────────────────────────────────

class IntelligenceState(BaseModel):
    """Shared state object passed through the LangGraph workflow."""

    # Input
    input: ResearchInput

    # Research phase
    raw_documents: List[Dict[str, Any]] = Field(default_factory=list)
    news_articles: List[Dict[str, Any]] = Field(default_factory=list)
    rag_chunks_count: int = 0

    # Company analysis
    company_overview: str = ""
    products_services: str = ""
    recent_activities: str = ""

    # Competitors
    identified_competitors: List[str] = Field(default_factory=list)
    competitor_analyses: List[CompetitorInfo] = Field(default_factory=list)
    competitor_table: str = ""

    # Marketing intelligence
    product_positioning: str = ""
    value_propositions: str = ""
    market_messaging: str = ""
    pricing_insights: str = ""
    market_differentiation: str = ""

    # Lead intelligence
    lead_signals: List[LeadSignal] = Field(default_factory=list)
    lead_score: Optional[LeadScore] = None
    outreach_recommendations: str = ""

    # Final report
    executive_summary: str = ""
    swot_analysis: Optional[SWOTAnalysis] = None
    sources: List[str] = Field(default_factory=list)
    final_report: str = ""

    # Workflow metadata
    errors: List[str] = Field(default_factory=list)
    agent_logs: List[str] = Field(default_factory=list)
    status: str = "initializing"

    class Config:
        arbitrary_types_allowed = True
