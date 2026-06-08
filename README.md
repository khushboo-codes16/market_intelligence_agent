# Market Intelligence Agent

An AI-powered business intelligence assistant for product marketing, competitor analysis, and lead generation research.

This project collects public web data, builds a RAG index, runs specialized intelligence agents in parallel, and produces a business-ready markdown report with competitor insights, product positioning, lead signals, outreach recommendations, and a cold email draft.

## Key Capabilities

- Streamlit dashboard for interactive analysis
- CLI mode for terminal-based company runs
- Public website and news scraping with RAG retrieval
- Product marketing, competitor, and lead generation analysis
- Markdown report generation and optional PDF export
- Disk-backed LLM response cache for faster repeat runs
- Run history, batch analysis, and compare past reports
- Configurable Groq LLM settings, token limits, and concurrency

## Project Structure

- `app.py` — main entry point; Streamlit default, CLI if args are passed
- `ui/dashboard.py` — Streamlit UI, report viewer, batch analysis, PDF export
- `workflow/graph.py` — pipeline orchestration and parallel agent execution
- `agents/` — research, product marketing, competitor analysis, lead generation, report generation
- `rag/rag_pipeline.py` — ChromaDB + SentenceTransformer retrieval pipeline
- `tools/llm_client.py` — Groq LLM wrapper with caching, retry, and rate-limit handling
- `tools/scraper.py` — website and news scraping helpers
- `reports/report_saver.py` — markdown report save/load, run history, PDF generation
- `models/state.py` — shared Pydantic state models for pipeline data
- `config/settings.py` — environment-based configuration
- `tests/` — unit and integration tests for key workflows

## Architecture Overview

1. **Input** — user submits company name, website, product, and scenario
2. **Research Agent** — scrapes public sources, extracts context, builds a RAG index
3. **Parallel analysis** — product marketing, competitor analysis, and lead generation agents run independently
4. **Report Generation** — assembles final markdown report and SWOT
5. **Output** — dashboard render, report save, PDF download, run history

## Running the Project

### Streamlit Dashboard

```bash
streamlit run ui/dashboard.py --server.fileWatcherType none
```

### CLI Mode

```bash
python app.py "Google" --website "https://google.com" --product "Search" --scenario "Consumer search intelligence"
```

### Example CLI Command

```bash
python app.py "OpenAI" --website "https://openai.com" --product "ChatGPT" --scenario "Enterprise AI assistant"
```

## Configuration

Copy the example environment file and set your Groq API key:

```bash
cp .env.example .env
```

Edit `.env` with at least:

```text
GROQ_API_KEY=your_groq_api_key_here
LLM_MODEL=llama-3.3-70b-versatile
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=4096
LLM_MAX_CONCURRENT=2
```

### Important tuning knobs

- `LLM_MODEL` — Groq model name to use
- `LLM_MAX_TOKENS` — maximum response tokens per call
- `LLM_MAX_CONCURRENT` — limit concurrent Groq requests to reduce rate-limit errors
- `CHROMA_PERSIST_DIR` — local vector store path
- `TOP_K_RESULTS` — number of RAG results returned for each prompt

## LLM Rate Limit Notes

This project uses Groq. If you hit rate limits or daily token caps, try:

- lowering `LLM_MAX_CONCURRENT` to `1`
- lowering `LLM_MAX_TOKENS`
- using a smaller model if available
- re-running with cached responses for repeated companies

The LLM client includes retry logic and parses Groq retry suggestions when the provider returns a wait interval.

## What the Report Includes

- Executive summary
- Company overview
- Products and services
- Competitor analysis and comparison table
- Product positioning
- Market messaging
- Value propositions
- Pricing insights
- Market differentiation
- Recent activity summary
- Lead signals
- Lead score breakdown
- Outreach recommendations
- Personalized cold email draft
- SWOT analysis
- Sources and run details

## Public Data Sources

The system uses only public, crawlable sources such as:

- company website pages
- public news search results
- public competitor websites
- public jobs pages and search snippets
- public review and funding listing pages
- public RSS/press feeds and traffic snippets

## Testing

Run the test suite with:

```bash
pytest -q
```

## Notes

- The agent is designed for marketing and sales intelligence, not for confidential or paid data.
- Repeated company runs are faster due to the response cache stored in `data/llm_cache.json`.
- Saved reports are stored in `reports/` and run metadata is maintained in `reports/run_history.json`.
