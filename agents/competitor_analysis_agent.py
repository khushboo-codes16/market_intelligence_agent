# agents/competitor_analysis_agent.py
# ============================================================
# Competitor Analysis Agent: discover, scrape, and compare
# ============================================================

import re
import json
from typing import Dict, Any, List

from models.state import IntelligenceState, CompetitorInfo
from tools.llm_client import get_llm_client
from tools.scraper import scrape_page, search_web_news
from rag.rag_pipeline import get_rag_pipeline
from utils.logger import logger
from utils.text_utils import truncate_text


SYSTEM_PROMPT = """You are a competitive intelligence analyst specializing in B2B SaaS and AI markets.
You rigorously identify and analyze competitors, their positioning, messaging, strengths, and weaknesses.
Your output is factual, structured, and directly useful for sales and marketing teams."""


def competitor_analysis_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: Competitor Analysis Agent

    Responsibilities:
    1. Identify 3-5 competitors from scraped context
    2. Scrape competitor websites
    3. Analyze each competitor
    4. Build comparison table
    """
    intel = IntelligenceState(**state)
    company = intel.input.company_name
    product = intel.input.product_name or ""
    scenario = intel.input.product_scenario or ""
    intel.agent_logs.append("[CompetitorAgent] Starting competitor identification...")
    intel.status = "analyzing_competitors"

    llm = get_llm_client()
    rag = get_rag_pipeline(collection_name=f"intel_{_sanitize(company)}")

    # ── 1. Identify Competitors ────────────────────────────────
    context = rag.retrieve_as_context(
        f"{company} competitors alternatives similar companies market", top_k=5
    )

    competitor_prompt = f"""Identify the top 4-5 direct competitors of {company}.
Product context: {product or 'N/A'}
Scenario: {scenario or 'N/A'}

Use the context below AND your knowledge.

CONTEXT:
{context}

Return ONLY a JSON array of competitor names (strings), e.g.:
["CompanyA", "CompanyB", "CompanyC", "CompanyD"]

Return only the JSON array, nothing else."""

    raw_competitors = llm.complete(prompt=competitor_prompt, max_tokens=300)

    # Parse competitor list
    competitors = _parse_json_list(raw_competitors)
    if not competitors:
        # Fallback: search for competitors
        search_results = search_web_news(f"{company} competitors alternatives top", num_results=5)
        competitors = _extract_company_names_from_search(search_results, company, llm)

    competitors = [c for c in competitors if c.lower() != company.lower()][:5]
    intel.identified_competitors = competitors
    intel.agent_logs.append(f"[CompetitorAgent] Identified competitors: {competitors}")

    # ── 2. Analyze Each Competitor ─────────────────────────────
    competitor_analyses: List[CompetitorInfo] = []

    for comp_name in competitors:
        intel.agent_logs.append(f"[CompetitorAgent] Analyzing: {comp_name}")
        comp_info = _analyze_competitor(comp_name, company, product, scenario, llm)
        competitor_analyses.append(comp_info)

    intel.competitor_analyses = competitor_analyses

    # ── 3. Build Comparison Table ──────────────────────────────
    intel.competitor_table = _build_comparison_table(company, competitor_analyses, llm)
    intel.agent_logs.append("[CompetitorAgent] Competitor analysis complete.")
    intel.status = "competitors_complete"
    return intel.model_dump()


def _analyze_competitor(
    comp_name: str,
    target_company: str,
    product: str,
    scenario: str,
    llm,
) -> CompetitorInfo:
    """Scrape and analyze a single competitor."""
    # Try to get competitor website content
    website = _guess_website(comp_name)
    comp_text = ""

    try:
        page = scrape_page(website)
        comp_text = truncate_text(page.get("text", ""), max_chars=2500)
    except Exception as e:
        logger.warning(f"Could not scrape {comp_name}: {e}")

    # Also search for info
    search_results = search_web_news(f"{comp_name} product features positioning", num_results=3)
    search_context = "\n".join([r.get("snippet", "") for r in search_results])

    combined_context = f"{comp_text}\n\nSearch snippets:\n{search_context}".strip()
    if not combined_context:
        combined_context = f"Limited data available for {comp_name}."

    analysis_prompt = f"""Analyze {comp_name} as a competitor to {target_company}.
Target product context: {product or 'N/A'}
Scenario: {scenario or 'N/A'}

COMPETITOR CONTEXT:
{combined_context}

Return a JSON object with these exact keys:
{{
  "description": "1-2 sentence company description",
  "positioning": "How they position their product",
  "value_proposition": "Their core value proposition",
  "strengths": ["strength1", "strength2", "strength3"],
  "weaknesses": ["weakness1", "weakness2", "weakness3"],
  "messaging_comparison": "How their messaging compares to {target_company}"
}}

Return ONLY the JSON object, no extra text."""

    raw = llm.complete(prompt=analysis_prompt, max_tokens=700)
    data = _parse_json_object(raw)

    return CompetitorInfo(
        name=comp_name,
        website=website,
        description=data.get("description", ""),
        positioning=data.get("positioning", ""),
        value_proposition=data.get("value_proposition", ""),
        strengths=data.get("strengths", []),
        weaknesses=data.get("weaknesses", []),
        messaging_comparison=data.get("messaging_comparison", ""),
    )


def _build_comparison_table(
    company: str,
    competitors: List[CompetitorInfo],
    llm,
) -> str:
    """Generate a structured comparison table in markdown format."""
    comp_summaries = []
    for c in competitors:
        comp_summaries.append(
            f"**{c.name}**: {c.description}\n"
            f"  Positioning: {c.positioning}\n"
            f"  Value Prop: {c.value_proposition}\n"
            f"  Strengths: {', '.join(c.strengths[:3])}\n"
            f"  Weaknesses: {', '.join(c.weaknesses[:3])}"
        )

    prompt = f"""Create a competitor comparison table for {company} vs its competitors.

Competitor data:
{"---".join(comp_summaries)}

Generate a markdown table comparing all competitors across these dimensions:
- Positioning
- Primary Value Proposition
- Target Audience
- Key Strengths
- Key Weaknesses
- Pricing Model (if known)

Make it business-ready and actionable."""

    return llm.complete(prompt=prompt, max_tokens=1200)


def _parse_json_list(text: str) -> List[str]:
    """Extract a JSON list from LLM output."""
    text = text.strip()
    match = re.search(r'\[.*?\]', text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            return [str(x).strip() for x in result if x]
        except json.JSONDecodeError:
            pass
    # Fallback: extract quoted strings
    return re.findall(r'"([^"]{2,50})"', text)


def _parse_json_object(text: str) -> Dict[str, Any]:
    """Extract a JSON object from LLM output."""
    text = text.strip()
    # Try to find JSON block
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            # Try to fix common issues
            fixed = match.group().replace("'", '"').replace('\n', ' ')
            try:
                return json.loads(fixed)
            except:
                pass
    return {}


def _extract_company_names_from_search(search_results, exclude, llm) -> List[str]:
    """Use LLM to extract company names from search snippets."""
    snippets = "\n".join([r.get("snippet", "") + " " + r.get("title", "") for r in search_results])
    prompt = f"""Extract company names from these search results (competitors to {exclude}).

SEARCH RESULTS:
{snippets}

Return ONLY a JSON array of company names: ["Company1", "Company2", ...]"""
    raw = llm.complete(prompt=prompt, max_tokens=200)
    return _parse_json_list(raw)


def _guess_website(name: str) -> str:
    known = {
        "openai": "https://openai.com",
        "anthropic": "https://anthropic.com",
        "google": "https://google.com",
        "microsoft": "https://microsoft.com",
        "elevenlabs": "https://elevenlabs.io",
        "notion": "https://notion.so",
        "salesforce": "https://salesforce.com",
        "hubspot": "https://hubspot.com",
        "slack": "https://slack.com",
        "zoom": "https://zoom.us",
        "stripe": "https://stripe.com",
        "github": "https://github.com",
        "linear": "https://linear.app",
        "figma": "https://figma.com",
        "vercel": "https://vercel.com",
        "supabase": "https://supabase.com",
        "airtable": "https://airtable.com",
        "cohere": "https://cohere.com",
        "mistral": "https://mistral.ai",
        "perplexity": "https://perplexity.ai",
        "claude": "https://anthropic.com",
        "gemini": "https://gemini.google.com",
        "deepmind": "https://deepmind.google",
        "meta ai": "https://ai.meta.com",
        "aws": "https://aws.amazon.com",
        "azure": "https://azure.microsoft.com",
        "gcp": "https://cloud.google.com",
        "retell ai": "https://retellai.com",
        "bland ai": "https://bland.ai",
        "vapi": "https://vapi.ai",
        "synthflow": "https://synthflow.ai",
    }
    key = name.lower().strip()
    if key in known:
        return known[key]
    slug = re.sub(r"[^a-z0-9]", "", key)
    return f"https://www.{slug}.com"


def _sanitize(name: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", name.lower().strip())[:40]
