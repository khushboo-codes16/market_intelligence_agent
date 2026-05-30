# agents/research_agent.py
# ============================================================
# Research Agent: scrapes company site + news, ingests into RAG
# ============================================================

from typing import Dict, Any

from models.state import IntelligenceState
from tools.scraper import scrape_company_website, scrape_news_articles
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

    # ── 4. Ingest into RAG ─────────────────────────────────────
    rag = get_rag_pipeline(collection_name=f"intel_{_sanitize(company)}")
    rag.reset_collection()  # Fresh collection per run

    docs_to_ingest = []
    for page in web_pages:
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
