# Project Approach

## Goal

Build an AI-powered business intelligence assistant that helps marketing and sales teams understand a target company, its competitors, and its lead potential from public data.

The system should convert unstructured public information into structured business insights, including:

- product and positioning analysis
- competitor discovery and comparison
- buying signal extraction
- lead scoring and outreach recommendations
- a cohesive markdown report with SWOT

## Core Workflow

1. **Input**
   - company name
   - optional website URL
   - optional product name
   - optional scenario or use case

2. **Research Agent**
   - scrapes public website pages and related sources
   - collects news and public intelligence snippets
   - cleans and chunks text
   - stores content in a local RAG index (ChromaDB)

3. **Parallel agents**
   - product marketing analysis
   - competitor analysis
   - lead generation analysis
   
   These agents run in parallel because they consume the same RAG context and do not depend on each other.

4. **Report Generation**
   - final executive summary
   - SWOT analysis
   - report assembly in markdown

5. **Output**
   - Streamlit dashboard view
   - saved markdown report
   - optional PDF export
   - run history and comparison data

## Architecture

The project is organized as follows:

- `workflow/graph.py` — pipeline orchestration using a simple state graph for research and report routing
- `agents/research_agent.py` — public data collection and RAG ingestion
- `agents/product_marketing_agent.py` — messaging, positioning, pricing, and differentiation analysis
- `agents/competitor_analysis_agent.py` — competitor identification, profiles, and comparison table
- `agents/lead_generation_agent.py` — signal extraction, lead scoring, outreach recommendations, and email drafting
- `agents/report_generation_agent.py` — final report generation and SWOT assembly
- `tools/scraper.py` — website and public news scraping utilities
- `tools/llm_client.py` — Groq client wrapper with caching, retry logic, and rate-limit management
- `rag/rag_pipeline.py` — ChromaDB vector store and retrieval functions
- `reports/report_saver.py` — markdown + PDF saving and run history management
- `ui/dashboard.py` — Streamlit interface with tabs, export, batch analysis, and comparison
- `models/state.py` — shared state model definitions using Pydantic

## Key Design Decisions

### RAG first, then analysis

Collecting and indexing public context before analysis allows all agents to work from the same evidence pool. This improves consistency and reduces repeated scraping/LLM calls.

### Parallel agent execution

The three core analysis agents are independent after research completes. Running them in parallel using `ThreadPoolExecutor` improves pipeline speed while keeping shared state safe.

### Rate-limit-aware LLM handling

Groq LLM calls are wrapped in `tools/llm_client.py`. The wrapper includes:

- response caching in `data/llm_cache.json`
- concurrency control via `LLM_MAX_CONCURRENT`
- retry/backoff behavior for rate-limit responses
- daily token limit detection to fail cleanly when quota is exhausted

### Public data only

The project intentionally uses only public sources, avoiding login-protected or private data. Public sources include:

- company websites
- news/search results
- competitor pages
- jobs/recruiting pages
- review and funding listing snippets
- Product Hunt, RSS, and traffic snippets

### Structured state modeling

The pipeline state is defined in `models/state.py`. This makes agent input/output more reliable and easier to inspect in logs or tests.

### UI + CLI support

The project supports both:

- Streamlit dashboard for interactive exploration, full report rendering, PDF download, and batch runs
- CLI mode for quick terminal execution and saved markdown output

## Agent Responsibilities

### Research Agent

- resolves the company URL when needed
- scrapes homepage and internal pages
- retrieves public news/articles around the company
- constructs a RAG index in ChromaDB
- provides context for downstream agents

### Product Marketing Agent

- generates company overview
- describes products and services
- analyzes positioning and messaging
- extracts value propositions
- finds pricing insights
- summarizes differentiation

### Competitor Analysis Agent

- identifies direct competitors via RAG context
- analyzes competitor strengths, weaknesses, and messaging
- builds a comparison table

### Lead Generation Agent

- extracts signals for hiring, funding, launch, expansion, and partnership
- computes a lead score from 0–100
- generates outreach recommendations
- drafts a personalized cold email

### Report Generation Agent

- compiles findings into a final markdown report
- generates SWOT analysis
- includes sources and summary notes

## Practical Tradeoffs

- **Parallelism** improves speed, but must be balanced with LLM provider limits.
- **Caching** reduces repeated API usage for identical prompts.
- **Smaller models / lower token settings** can be used if Groq rate limits are triggered.
- **Markdown-first output** keeps the report portable, readable, and easy to save or export.

## What to update next

If you want to extend this project further, the next improvements could be:

- provider fallback support for local or alternative LLMs
- richer source extraction from additional public APIs
- stronger prompt validation and structured output enforcement
- better signal provenance and confidence scoring
- a more robust retry/queueing layer for daily token quotas
