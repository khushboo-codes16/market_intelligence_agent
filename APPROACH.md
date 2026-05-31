# Approach Notes

## Architecture Decisions
- Why LangGraph: explicit state machine, debuggable agent routing
- Why Groq: free tier, fastest inference, no cost barrier
- Why ChromaDB + SentenceTransformers: fully local, no API cost

## Data Source Rationale
- Company websites: primary source of positioning/messaging
- DuckDuckGo HTML search: no API key needed, public data
- News scraping: signals for lead intelligence

## AI Usage
- LLM for: competitor ID, analysis, signal extraction, scoring justification
- Embeddings for: RAG retrieval via SentenceTransformers

## Limitations
- Some sites block scrapers (Cloudflare, etc.)
- Lead scoring is heuristic, not ML-based
- News freshness depends on DuckDuckGo index

## Future Improvements
- Add LinkedIn job postings scraper for stronger hiring signals
- Add Product Hunt / G2 review scraper
- Replace heuristic scoring with a trained classifier
- Add PDF export for reports