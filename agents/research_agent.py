# agents/research_agent.py
# ============================================================
# Research Agent: scrapes company site + news, ingests into RAG
# ============================================================

from typing import Dict, Any

from models.state import IntelligenceState
from tools.scraper import scrape_company_website, scrape_news_articles
from tools.public_sources import (
    collect_public_intelligence_sources,
    summarise_public_source_counts,
)
from rag.rag_pipeline import get_rag_pipeline
from utils.logger import logger


def research_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: Research Agent

    Responsibilities:
    1. Discover company URL if not provided
    2. Scrape company website (homepage + sub-pages)
    3. Collect news / press releases
    4. Ingest all collected text into ChromaDB RAG
    """
    intel = IntelligenceState(**state)
    company = intel.input.company_name
    website = intel.input.company_website or ""
    intel.agent_logs.append("[ResearchAgent] Starting data collection...")
    intel.status = "researching"

    # ── 1. Resolve website if not given ────────────────────────
    if not website:
        website = _guess_website(company)
        intel.input.company_website = website
        intel.agent_logs.append(f"[ResearchAgent] Resolved website: {website}")

    # ── 2. Scrape company website ──────────────────────────────
    web_pages = scrape_company_website(website)
    intel.raw_documents.extend(web_pages)
    intel.sources.extend([p["url"] for p in web_pages if p.get("url")])
    intel.agent_logs.append(f"[ResearchAgent] Scraped {len(web_pages)} website pages.")

    # ── 3. Collect news & press releases ──────────────────────
    news_topics = [
        f"{company} news 2024",
        f"{company} funding announcement",
        f"{company} product launch",
        f"{company} hiring expansion",
        f"{company} partnership press release",
    ]

    if intel.input.product_name:
        news_topics.append(f"{company} {intel.input.product_name}")

    articles = scrape_news_articles(company, topics=news_topics)
    intel.news_articles.extend(articles)

    for art in articles:
        if art.get("url"):
            intel.sources.append(art["url"])
            art["source_type"] = "news"

    intel.agent_logs.append(f"[ResearchAgent] Collected {len(articles)} news articles.")

    # ── 4. Public enrichment sources ───────────────────────────
    # Best-effort connectors for public jobs, funding, reviews,
    # Product Hunt, RSS/press, and traffic pages/snippets.
    enrichment_docs = collect_public_intelligence_sources(company, website)
    intel.raw_documents.extend(enrichment_docs)
    intel.sources.extend([doc["url"] for doc in enrichment_docs if doc.get("url")])

    source_counts = summarise_public_source_counts(enrichment_docs)
    if source_counts:
        counts_str = ", ".join(f"{k}: {v}" for k, v in sorted(source_counts.items()))
        intel.agent_logs.append(f"[ResearchAgent] Public enrichment collected ({counts_str}).")
    else:
        intel.agent_logs.append("[ResearchAgent] Public enrichment found no additional documents.")

    # ── 5. Ingest into RAG ─────────────────────────────────────
    rag = get_rag_pipeline(collection_name=f"intel_{_sanitize(company)}")
    rag.reset_collection()  # Fresh collection per run

    docs_to_ingest = []
    for page in web_pages + enrichment_docs:
        if page.get("text"):
            docs_to_ingest.append({
                "text": page["text"],
                "source": page.get("url", website),
                "title": page.get("title", ""),
            })
    for art in articles:
        if art.get("text"):
            docs_to_ingest.append({
                "text": art["text"],
                "source": art.get("url", "news"),
                "title": art.get("title", art.get("search_title", "")),
            })

    count = rag.ingest_documents(docs_to_ingest)
    intel.rag_chunks_count = count
    intel.agent_logs.append(f"[ResearchAgent] Ingested {count} chunks into ChromaDB.")

    # ── 6. Compute data confidence based on what we collected ──
    # This drives the confidence badges shown in the UI on every section.
    pages_scraped = len(web_pages) + len(enrichment_docs)
    articles_scraped = len(articles)

    if count >= 30 and pages_scraped >= 3:
        base_confidence = "high"
    elif count >= 10 or pages_scraped >= 1:
        base_confidence = "medium"
    else:
        base_confidence = "low"

    # Pricing is harder to scrape reliably — cap one level lower
    pricing_confidence = {"high": "medium", "medium": "low", "low": "low"}[base_confidence]

    # News-dependent sections benefit from articles
    news_confidence = base_confidence if articles_scraped >= 3 else (
        "medium" if base_confidence == "high" else "low"
    )

    intel.data_confidence = {
        "company_overview":       base_confidence,
        "products_services":      base_confidence,
        "product_positioning":    base_confidence,
        "value_propositions":     base_confidence,
        "market_messaging":       base_confidence,
        "pricing_insights":       pricing_confidence,
        "market_differentiation": base_confidence,
        "competitor_analysis":    "medium",   # always partial — external scraping
        "lead_signals":           news_confidence,
        "lead_score":             news_confidence,
        "outreach_recommendations": news_confidence,
        "recent_activities":      news_confidence,
        "swot_analysis":          base_confidence,
    }
    intel.agent_logs.append(
        f"[ResearchAgent] Data confidence set to '{base_confidence}' "
        f"({pages_scraped} pages, {articles_scraped} articles, {count} chunks)."
    )

    # Deduplicate sources
    intel.sources = list(dict.fromkeys(intel.sources))
    intel.status = "research_complete"

    return intel.model_dump()


def _guess_website(company_name: str) -> str:
    """Heuristic: try common domain patterns for well-known companies."""
    known = {
        "openai": "https://openai.com",
        "anthropic": "https://anthropic.com",
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
    }
    key = company_name.lower().strip()
    if key in known:
        return known[key]
    # Generic guess
    slug = key.replace(" ", "").replace(".", "")
    return f"https://www.{slug}.com"


def _sanitize(name: str) -> str:
    """Sanitize collection name for ChromaDB (alphanumeric + underscore)."""
    import re
    return re.sub(r"[^a-z0-9_]", "_", name.lower().strip())[:40]
