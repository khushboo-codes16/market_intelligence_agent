# agents/lead_generation_agent.py
# ============================================================
# Lead Generation Agent: signals, scoring, outreach angles
# ============================================================

import re
import json
from typing import Dict, Any, List

from models.state import IntelligenceState, LeadSignal, LeadScore
from tools.llm_client import get_llm_client
from rag.rag_pipeline import get_rag_pipeline
from utils.logger import logger
from config.settings import (
    LEAD_SCORE_HIRING_WEIGHT,
    LEAD_SCORE_FUNDING_WEIGHT,
    LEAD_SCORE_LAUNCH_WEIGHT,
    LEAD_SCORE_EXPANSION_WEIGHT,
    LEAD_SCORE_PARTNERSHIP_WEIGHT,
)


SYSTEM_PROMPT = """You are an expert B2B sales intelligence analyst.
You identify buying signals, growth indicators, and outreach opportunities from company news,
hiring patterns, product launches, and market movements.
Your insights directly help sales teams prioritize and personalize outreach."""


def lead_generation_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: Lead Generation Agent

    Responsibilities:
    1. Extract lead signals from all collected data
    2. Score the lead (0-100)
    3. Identify recent activities
    4. Generate outreach recommendations
    """
    intel = IntelligenceState(**state)
    company = intel.input.company_name
    product = intel.input.product_name or ""
    scenario = intel.input.product_scenario or ""
    intel.agent_logs.append("[LeadGenAgent] Starting lead intelligence analysis...")
    intel.status = "analyzing_leads"

    llm = get_llm_client()
    rag = get_rag_pipeline(collection_name=f"intel_{_sanitize(company)}")

    # ── 1. Extract Lead Signals ────────────────────────────────
    hiring_context = rag.retrieve_as_context(
        f"{company} hiring jobs positions team growth", top_k=4
    )
    funding_context = rag.retrieve_as_context(
        f"{company} funding investment series round valuation", top_k=4
    )
    launch_context = rag.retrieve_as_context(
        f"{company} product launch release announcement new feature", top_k=4
    )
    expansion_context = rag.retrieve_as_context(
        f"{company} expansion new market international office", top_k=3
    )
    partnership_context = rag.retrieve_as_context(
        f"{company} partnership integration collaboration announcement", top_k=3
    )

    # Add news article snippets
    news_context = _build_news_context(intel.news_articles)

    signals: List[LeadSignal] = []

    # Extract each signal type
    signals.extend(_extract_signals("hiring", hiring_context + "\n" + news_context, company, llm))
    signals.extend(_extract_signals("funding", funding_context + "\n" + news_context, company, llm))
    signals.extend(_extract_signals("launch", launch_context + "\n" + news_context, company, llm))
    signals.extend(_extract_signals("expansion", expansion_context + "\n" + news_context, company, llm))
    signals.extend(_extract_signals("partnership", partnership_context + "\n" + news_context, company, llm))

    intel.lead_signals = signals
    intel.agent_logs.append(f"[LeadGenAgent] Extracted {len(signals)} lead signals.")

    # ── 2. Compute Lead Score ──────────────────────────────────
    intel.lead_score = _compute_lead_score(signals)
    intel.agent_logs.append(f"[LeadGenAgent] Lead score: {intel.lead_score.total_score}/100")

    # ── 3. Recent Activities Summary ──────────────────────────
    all_context = "\n\n".join([hiring_context, funding_context, launch_context, news_context])
    intel.recent_activities = llm.complete(
        prompt=f"""Summarize the recent activities and notable events for {company} in the past 12 months.
Focus on: product launches, funding, partnerships, hiring surges, expansions, and news.

CONTEXT:
{all_context[:3000]}

Write a concise 3-5 paragraph summary of recent activities.""",
        system_prompt=SYSTEM_PROMPT,
        max_tokens=600,
    )

    # ── 4. Outreach Recommendations ───────────────────────────
    signals_summary = _format_signals_for_prompt(signals)
    score_info = intel.lead_score

    intel.outreach_recommendations = llm.complete(
        prompt=f"""Generate actionable outreach recommendations for a sales team targeting {company}.

Company: {company}
Product/Scenario context: {product or scenario or 'General B2B outreach'}
Lead Score: {score_info.total_score}/100

Key signals detected:
{signals_summary}

Provide:
1. Best outreach angle / hook (based on strongest signal)
2. Recommended timing and channel (email, LinkedIn, etc.)
3. 3 specific talking points personalized to their situation
4. Potential objections and how to handle them
5. Suggested next steps

Make recommendations specific and actionable.""",
        system_prompt=SYSTEM_PROMPT,
        max_tokens=800,
    )

    intel.agent_logs.append("[LeadGenAgent] Lead intelligence analysis complete.")
    intel.status = "leads_complete"
    return intel.model_dump()


def _extract_signals(
    signal_type: str,
    context: str,
    company: str,
    llm,
) -> List[LeadSignal]:
    """Use LLM to extract signals of a given type from context."""
    if not context or len(context) < 50:
        return []

    prompt = f"""Extract {signal_type} signals for {company} from the context below.

CONTEXT:
{context[:2000]}

Return a JSON array of signal objects. Each object must have:
{{
  "signal_type": "{signal_type}",
  "description": "Specific description of the signal",
  "source": "URL or source name if visible",
  "date_mentioned": "Date or period if mentioned (e.g. 'Q1 2025'), else empty string",
  "confidence": "high|medium|low"
}}

If no {signal_type} signals found, return: []
Return ONLY the JSON array."""

    raw = llm.complete(prompt=prompt, max_tokens=500)

    try:
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if match:
            items = json.loads(match.group())
            return [
                LeadSignal(
                    signal_type=item.get("signal_type", signal_type),
                    description=item.get("description", ""),
                    source=item.get("source", ""),
                    date_mentioned=item.get("date_mentioned", ""),
                    confidence=item.get("confidence", "medium"),
                )
                for item in items
                if item.get("description")
            ]
    except Exception as e:
        logger.debug(f"Signal extraction parse error ({signal_type}): {e}")

    return []


def _compute_lead_score(signals: List[LeadSignal]) -> LeadScore:
    """
    Score the lead 0-100 based on detected signals.

    Weights:
    - Hiring: up to 25 pts
    - Funding: up to 30 pts
    - Launch: up to 20 pts
    - Expansion: up to 15 pts
    - Partnership: up to 10 pts
    """
    CONFIDENCE_MULTIPLIER = {"high": 1.0, "medium": 0.6, "low": 0.3}

    # Count effective signals per type
    type_scores: Dict[str, float] = {
        "hiring": 0,
        "funding": 0,
        "launch": 0,
        "expansion": 0,
        "partnership": 0,
    }

    for signal in signals:
        stype = signal.signal_type.lower()
        multiplier = CONFIDENCE_MULTIPLIER.get(signal.confidence, 0.5)
        if stype in type_scores:
            type_scores[stype] += multiplier

    # Cap and weight each category
    def capped_score(raw: float, weight: int, max_signals: int = 3) -> int:
        return min(int((raw / max_signals) * weight), weight)

    hiring_score = capped_score(type_scores["hiring"], LEAD_SCORE_HIRING_WEIGHT)
    funding_score = capped_score(type_scores["funding"], LEAD_SCORE_FUNDING_WEIGHT)
    launch_score = capped_score(type_scores["launch"], LEAD_SCORE_LAUNCH_WEIGHT)
    expansion_score = capped_score(type_scores["expansion"], LEAD_SCORE_EXPANSION_WEIGHT)
    partnership_score = capped_score(type_scores["partnership"], LEAD_SCORE_PARTNERSHIP_WEIGHT)

    total = hiring_score + funding_score + launch_score + expansion_score + partnership_score

    # Build justification
    justification_parts = []
    if hiring_score > 0:
        justification_parts.append(f"Hiring activity ({hiring_score}/{LEAD_SCORE_HIRING_WEIGHT} pts): active recruitment signals detected")
    if funding_score > 0:
        justification_parts.append(f"Funding signals ({funding_score}/{LEAD_SCORE_FUNDING_WEIGHT} pts): investment activity found")
    if launch_score > 0:
        justification_parts.append(f"Product launches ({launch_score}/{LEAD_SCORE_LAUNCH_WEIGHT} pts): new product/feature announcements")
    if expansion_score > 0:
        justification_parts.append(f"Expansion signals ({expansion_score}/{LEAD_SCORE_EXPANSION_WEIGHT} pts): market/geographic growth")
    if partnership_score > 0:
        justification_parts.append(f"Partnerships ({partnership_score}/{LEAD_SCORE_PARTNERSHIP_WEIGHT} pts): active partnership activity")

    justification = (
        f"Total score: {total}/100. "
        + ("; ".join(justification_parts) if justification_parts else "Minimal signals detected.")
    )

    return LeadScore(
        total_score=total,
        hiring_score=hiring_score,
        funding_score=funding_score,
        launch_score=launch_score,
        expansion_score=expansion_score,
        partnership_score=partnership_score,
        justification=justification,
        signals=signals,
    )


def _build_news_context(articles: List[Dict]) -> str:
    parts = []
    for art in articles[:8]:
        title = art.get("search_title") or art.get("title", "")
        snippet = art.get("search_snippet", "")
        text_preview = art.get("text", "")[:300]
        source = art.get("url", "")
        if title or snippet:
            parts.append(f"[{source}]\nTitle: {title}\n{snippet}\n{text_preview}")
    return "\n\n".join(parts)


def _format_signals_for_prompt(signals: List[LeadSignal]) -> str:
    if not signals:
        return "No specific signals detected."
    lines = []
    for s in signals[:10]:
        lines.append(f"- [{s.signal_type.upper()}] ({s.confidence} confidence): {s.description}")
    return "\n".join(lines)


def _sanitize(name: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", name.lower().strip())[:40]


# Type hint for dict used in loop
from typing import Dict
