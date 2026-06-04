# AI-Powered Business & Market Intelligence Agent

Built for the DS Intern Hackathon, May 2026.

This project is an AI assistant for product marketing and lead generation research. A user enters a company name, website, product, or scenario, and the system gathers public information, builds a RAG index, runs multiple intelligence agents, and produces a business-readable report with competitors, positioning, lead signals, scores, outreach recommendations, sources, and downloadable outputs.

## What It Does

The agent supports both hackathon tracks:

| Track | Implemented Capability |
|-------|------------------------|
| Product Marketing Intelligence | Analyzes company overview, products, positioning, value propositions, messaging, pricing, differentiation, competitors, and SWOT. |
| Lead Generation Intelligence | Extracts buying signals from public data, computes a 0-100 lead score, summarizes recent activity, and generates outreach recommendations plus a cold email draft. |

Main features:

- Streamlit dashboard for interactive use
- CLI mode for running the pipeline from terminal
- Public website and news scraping
- Best-effort public enrichment from LinkedIn Jobs, Crunchbase, Tracxn, G2, Capterra, Product Hunt, RSS/press feeds, and SimilarWeb pages/snippets
- RAG pipeline with ChromaDB and SentenceTransformers
- Groq LLM integration
- Parallel execution of product marketing, competitor, and lead generation agents
- Disk-backed LLM response cache
- Markdown report generation
- PDF export from the dashboard
- Run history and side-by-side comparison of past runs
- Batch company analysis from pasted names or CSV
- Data confidence badges per report section
- Signal timeline visualization
- Streaming LLM executive briefing in the dashboard
- Unit and integration tests with external services mocked

## Project Flow

```text
User input
  |
  v
Research Agent
  - Resolves/uses company website
  - Scrapes public company pages
  - Collects public news/search results
  - Cleans and chunks text
  - Stores chunks in ChromaDB
  |
  v
Parallel analysis phase
  - Product Marketing Agent
  - Competitor Analysis Agent
  - Lead Generation Agent
  |
  v
Report Generation Agent
  - SWOT
  - Executive summary
  - Final markdown report
  |
  v
Streamlit dashboard / CLI output / saved report
```

## Architecture

| Module | Purpose |
|--------|---------|
| `app.py` | Main entry point. Runs CLI mode when arguments are provided, otherwise launches Streamlit. |
| `ui/dashboard.py` | Streamlit dashboard with inputs, tabs, downloads, history, comparison, batch analysis, and visualizations. |
| `workflow/graph.py` | LangGraph workflow and parallel agent orchestration. |
| `agents/research_agent.py` | Public data collection and RAG ingestion. |
| `agents/product_marketing_agent.py` | Product, positioning, messaging, pricing, and differentiation analysis. |
| `agents/competitor_analysis_agent.py` | Competitor discovery, competitor profiling, and comparison table generation. |
| `agents/lead_generation_agent.py` | Signal extraction, lead scoring, outreach recommendations, and email draft generation. |
| `agents/report_generation_agent.py` | SWOT, executive summary, and final report compilation. |
| `rag/rag_pipeline.py` | ChromaDB vector store and SentenceTransformer embeddings. |
| `tools/scraper.py` | Website/news scraping helpers using requests, BeautifulSoup, and Trafilatura. |
| `tools/llm_client.py` | Groq client wrapper with retry logic and LLM caching. |
| `reports/report_saver.py` | Markdown report saving, run history, and PDF generation. |
| `models/state.py` | Pydantic models for shared pipeline state and structured outputs. |
| `utils/` | Logging, confidence badges, text cleaning, chunking, and helpers. |
| `tests/` | Unit and mocked integration tests. |

## Setup

### 1. Clone the Project

```bash
git clone <repo-url>
cd market_intel_agent
```

### 2. Create a Python Environment

Python 3.11 is recommended.

```bash
conda create --name market-intel-agent python=3.11
conda activate market-intel-agent
```

Or with `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The first run may download the SentenceTransformers embedding model.

### 4. Configure Environment Variables

Copy the example environment file:

```bash
cp .env.example .env
```

Then add your Groq API key:

```text
GROQ_API_KEY=your_groq_api_key_here
```

You can get a Groq API key from `https://console.groq.com`.

## Running the Project

### Streamlit Dashboard

```bash
streamlit run ui/dashboard.py
```

The dashboard usually opens at:

```text
http://localhost:8501
```

You can enter:

- Company name
- Company website
- Product name
- Product or market scenario

### CLI Mode

```bash
python app.py OpenAI --website https://openai.com --product "ChatGPT"
python app.py ElevenLabs --website https://elevenlabs.io --scenario "AI voice agent for lead generation"
python app.py Salesforce --product "Salesforce Einstein"
```

CLI mode prints a preview of the generated report and saves the full markdown report under `reports/`.

## Example Input

```text
Company Name: ElevenLabs
Website: https://elevenlabs.io
Scenario: A company launching a voice-based AI agent for taking calls and generating leads
```

## Example Output Sections

The generated report includes:

1. Executive summary
2. Company overview
3. Products and services
4. Competitor analysis
5. Competitor comparison table
6. Product positioning
7. Market messaging
8. Value propositions
9. Pricing insights
10. Market differentiation
11. Recent activities
12. Lead signals
13. Lead score with breakdown
14. Outreach recommendations
15. Cold email draft
16. SWOT analysis
17. Sources and references

## Data Sources

The system uses only public data sources:

- Company website pages
- Public news/search results through DuckDuckGo HTML search
- Public competitor websites
- User-provided public URLs
- Public LinkedIn Jobs pages/search snippets for hiring signals
- Public Crunchbase and Tracxn pages/search snippets for funding signals
- Public G2 and Capterra pages/search snippets for review and pain-point signals
- Public Product Hunt pages/search snippets for launch/community signals
- Public RSS/Atom feeds discovered from company websites
- Public SimilarWeb pages/search snippets for directional traffic signals

The project does not use login-protected, paid, confidential, or private data sources.

## AI Usage

AI is used for meaningful analysis steps, not only formatting.

| Component | AI Usage |
|-----------|----------|
| Company/product analysis | LLM analysis grounded in retrieved public context |
| Competitor identification | LLM extracts likely direct competitors |
| Competitor profiling | LLM summarizes positioning, strengths, weaknesses, and messaging |
| Signal extraction | LLM extracts funding, hiring, launch, expansion, and partnership signals |
| Outreach recommendations | LLM creates personalized sales angles and talking points |
| Email draft | LLM writes a short cold outreach email based on the strongest signal |
| SWOT analysis | LLM generates structured strengths, weaknesses, opportunities, and threats |
| Executive summary | LLM summarizes the final intelligence report |
| RAG retrieval | SentenceTransformer embeddings retrieve relevant scraped context |
| Coding/design help | Claude/Codex assistance was used during development |

## Lead Scoring

Lead score is computed from detected signals and capped at 100.

| Signal Type | Max Points | Why It Matters |
|-------------|------------|----------------|
| Funding | 30 | Fresh capital often means available budget and growth intent. |
| Hiring | 25 | Hiring indicates expansion and possible need for new tools. |
| Product launch | 20 | Launch activity creates needs around marketing, sales, and operations. |
| Expansion | 15 | New regions or markets often require new infrastructure and vendors. |
| Partnership | 10 | Partnerships show ecosystem activity and integration readiness. |

Each signal also has a confidence level:

| Confidence | Multiplier |
|------------|------------|
| High | 1.0 |
| Medium | 0.6 |
| Low | 0.3 |

## Dashboard Features

The Streamlit dashboard includes:

- Input sidebar with quick examples
- Pipeline progress indicator
- Overview tab
- Competitor tab
- Marketing intelligence tab
- Lead intelligence tab
- SWOT tab
- Sources and execution logs tab
- Batch analysis tab
- Markdown download
- PDF download
- Streaming executive briefing button
- Run history loader
- Past-run comparison view
- LLM cache clear button

## Reports and Runtime Files

Generated runtime files are stored locally:

| Path | Purpose |
|------|---------|
| `reports/*.md` | Saved markdown intelligence reports |
| `reports/run_history.json` | Lightweight run history for the UI |
| `data/chroma_db/` | ChromaDB vector database |
| `data/llm_cache.json` | Disk-backed LLM response cache |
| `data/app.log` | Application logs when enabled |

Most runtime files should not be committed unless they are intentionally included as sample output.

## Testing

Run tests with:

```bash
pytest -q
```

The tests cover:

- Pydantic models
- Lead scoring logic
- LLM cache behavior
- Report saving and run history
- Scraper helpers with mocked network calls
- Confidence badge helpers
- Signal date parsing
- Mocked pipeline integration and parallel merge behavior

Current local status: `99 passed, 1 skipped`.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | Required | Groq API key. |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Groq model name. |
| `LLM_TEMPERATURE` | `0.3` | LLM sampling temperature. |
| `LLM_MAX_TOKENS` | `4096` | Max tokens per LLM response. |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | ChromaDB persistence directory. |
| `CHROMA_COLLECTION_NAME` | `market_intel` | Default Chroma collection name. |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model. |
| `REQUEST_TIMEOUT` | `15` | HTTP request timeout in seconds. |
| `REQUEST_DELAY` | `1.5` | Delay between scraping requests. |
| `MAX_PAGES_PER_DOMAIN` | `5` | Max pages scraped from one domain. |
| `USER_AGENT` | Browser-like user agent | User agent for HTTP requests. |
| `CHUNK_SIZE` | `800` | RAG chunk size in characters. |
| `CHUNK_OVERLAP` | `150` | RAG chunk overlap in characters. |
| `TOP_K_RESULTS` | `5` | Default number of retrieved RAG chunks. |
| `LEAD_SCORE_HIRING_WEIGHT` | `25` | Max hiring signal score. |
| `LEAD_SCORE_FUNDING_WEIGHT` | `30` | Max funding signal score. |
| `LEAD_SCORE_LAUNCH_WEIGHT` | `20` | Max launch signal score. |
| `LEAD_SCORE_EXPANSION_WEIGHT` | `15` | Max expansion signal score. |
| `LEAD_SCORE_PARTNERSHIP_WEIGHT` | `10` | Max partnership signal score. |
| `LOG_LEVEL` | `INFO` | Logging verbosity. |
| `LOG_FILE` | `./data/app.log` | Log file path. |
| `REPORTS_DIR` | `./reports` | Saved reports directory. |

## Troubleshooting

### `GROQ_API_KEY is not set`

Create `.env` from `.env.example` and add your Groq API key.

### First run is slow

The embedding model is downloaded on first use. Later runs are faster.

### Website scraping returns little or no content

Some sites block scraping or render content with JavaScript. Try providing a more specific public URL, such as a pricing, blog, product, or about page.

### ChromaDB errors

Stop the app, delete `data/chroma_db/`, and rerun the pipeline.

### Low-quality output for obscure companies

Add the company website and a clear product/scenario description. More public context improves the report.

### PDF export fails

Make sure `weasyprint` and `markdown` are installed from `requirements.txt`. Linux systems may also need native WeasyPrint dependencies depending on the environment.

## Known Limitations

- Scraping quality depends on public website accessibility.
- Some websites block automated requests.
- News freshness depends on DuckDuckGo search results.
- Lead scoring is heuristic, not trained on real conversion outcomes.
- LinkedIn Jobs, Crunchbase, Tracxn, G2, Capterra, Product Hunt, RSS, and SimilarWeb connectors are best-effort public-page/snippet collectors, not official paid API integrations.
- The system uses public information only and should not be treated as a substitute for verified CRM or financial data.

## Future Improvements

- Replace best-effort public-page collectors with official APIs where legally permitted.
- Improve public source extraction with source-specific parsers.
- Add CRM export/integration.
- Train or calibrate lead scoring on historical conversion data.
- Add scheduled monitoring and alerts when a company score changes.

## Hackathon Checklist Status

| Requirement | Status |
|-------------|--------|
| Working demo or script | Done |
| Company/domain/scenario input | Done |
| Prompt or AI workflow | Done in agent code and `APPROACH.md` |
| Public data sources | Done |
| Final report with sources and recommendations | Done |
| README with setup/run steps | Done |
| Approach notes | Done in `APPROACH.md` |
| AI usage disclosure | Done |
| Manual evaluation sheet | Still needs a separate file |
| Recorded walkthrough/demo artifact | Still needs to be added for submission |

## License and Ethics

This project is intended for educational and hackathon use. It uses public, legally permitted sources only. Do not use it to scrape login-protected, paid, private, confidential, or restricted data.
