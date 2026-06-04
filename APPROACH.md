# Approach Notes

## Problem Understanding

The hackathon challenge is to build an AI assistant that helps a marketing or sales user understand a target company, its competitors, recent public activity, positioning, and possible outreach opportunities.

My project focuses on two use cases:

- **Product Marketing Intelligence:** understand how a company or product is positioned, messaged, packaged, priced, and differentiated.
- **Lead Generation Intelligence:** identify public buying signals such as funding, hiring, launches, expansion, and partnerships, then convert those signals into a lead score and outreach recommendations.

The goal is not just to summarize a website. The system turns unstructured public data into a structured business intelligence report that a marketing or sales person can quickly use.

## High-Level Architecture

The project uses a multi-agent workflow:

```text
Input
  |
  v
Research Agent
  |
  v
RAG Index
  |
  v
Parallel Analysis Phase
  - Product Marketing Agent
  - Competitor Analysis Agent
  - Lead Generation Agent
  |
  v
Report Generation Agent
  |
  v
Dashboard / CLI / Saved Report
```

The implementation is organized around:

- `workflow/graph.py` for orchestration
- `agents/` for specialized intelligence agents
- `rag/` for vector retrieval
- `tools/` for scraping and LLM access
- `models/` for shared Pydantic state
- `ui/` for the Streamlit dashboard
- `reports/` for markdown/PDF report handling and run history

## Architecture Decisions

### LangGraph for Orchestration

I used LangGraph because the workflow has clear phases and shared state:

1. Research must happen first.
2. Product marketing, competitor analysis, and lead generation can run independently after research.
3. Report generation must happen after all analysis agents finish.

The graph keeps this workflow explicit and easier to debug than a loose script.

### Parallel Agent Execution

After the research phase, the product marketing, competitor, and lead generation agents run in parallel using `ThreadPoolExecutor`.

This is useful because all three agents read from the same RAG context and do not depend on each other's output. It reduces total pipeline time compared with running all analysis agents sequentially.

### Groq for LLM Calls

Groq was selected because it provides fast LLM inference and is practical for a hackathon demo. The project uses an environment variable for the API key, so secrets are not hard-coded.

### ChromaDB and SentenceTransformers for RAG

The project uses:

- ChromaDB for local vector storage
- SentenceTransformers for local embeddings

This keeps retrieval inexpensive and avoids needing another hosted embedding API. Each company run gets a separate Chroma collection so context from different companies does not mix.

### Pydantic State Models

The shared pipeline state is defined in `models/state.py`. This makes agent inputs and outputs more consistent and helps catch invalid lead scores or malformed state.

## Data Source Rationale

The system uses public data only:

| Source | Why It Is Used |
|--------|----------------|
| Company websites | Best source for positioning, messaging, products, pricing, and value propositions. |
| Public news/search results | Useful for funding, hiring, launches, partnerships, and recent activity signals. |
| Public competitor websites | Helps compare competitor positioning and messaging. |
| User-provided URLs | Improves accuracy when the user knows the best company/product page. |
| LinkedIn Jobs public pages/snippets | Adds directional hiring and department-growth signals. |
| Crunchbase and Tracxn public pages/snippets | Adds directional funding, investor, and company-stage signals. |
| G2 and Capterra public pages/snippets | Adds review, pain-point, alternative, and customer-language signals. |
| Product Hunt public pages/snippets | Adds launch and community-reception signals. |
| RSS/Atom and press feeds | Adds direct company-published activity signals where feeds are available. |
| SimilarWeb public pages/snippets | Adds directional traffic and market-interest signals. |

I avoided login-protected, paid, restricted, or confidential sources. This keeps the project aligned with the hackathon guardrails.

## Data Handling

The research phase:

1. Uses a provided website or guesses a likely company domain.
2. Scrapes the homepage and selected internal pages.
3. Searches public web/news results for company activity.
4. Collects best-effort public enrichment from jobs, funding, review, Product Hunt, RSS/press, and traffic sources.
5. Cleans scraped text.
6. Splits text into paragraph/sentence-aware chunks.
7. Embeds chunks with SentenceTransformers.
8. Stores chunks in ChromaDB.
9. Deduplicates source URLs.

The system also assigns data confidence levels based on how much public data was collected. These confidence levels are shown in the dashboard so the user can see which sections are strongly backed by scraped data and which are more inferential.

## AI Workflow and Prompt Flow

The project uses LLM calls for structured reasoning and summarization.

### Product Marketing Agent

Prompts ask the LLM to analyze:

- Company overview
- Products and services
- Product positioning
- Value propositions
- Market messaging
- Pricing insights
- Market differentiation

Each prompt is grounded with retrieved RAG context from scraped public data.

### Competitor Analysis Agent

The competitor agent:

1. Retrieves context about alternatives and competitors.
2. Prompts the LLM to return a JSON list of competitors.
3. Scrapes/searches competitor information.
4. Prompts the LLM for structured competitor profiles.
5. Generates a comparison table.

The JSON parsing has fallback logic because LLMs sometimes return extra text around structured outputs.

### Lead Generation Agent

The lead generation agent retrieves context for five signal categories:

- Hiring
- Funding
- Product launch
- Expansion
- Partnership

It prompts the LLM to return structured signal objects with:

- Signal type
- Description
- Source
- Date mentioned
- Confidence

Then the code computes a lead score and asks the LLM to generate outreach recommendations and a short cold email draft.

### Report Generation Agent

The report agent:

- Generates SWOT analysis
- Writes the executive summary
- Compiles the final markdown report

The report includes analysis, scores, recommendations, and source references.

## Lead Scoring Approach

Lead score is heuristic and based on detected public signals.

| Signal | Max Score | Reasoning |
|--------|-----------|-----------|
| Funding | 30 | New funding often means budget and growth plans. |
| Hiring | 25 | Hiring indicates expansion and operational need. |
| Product launch | 20 | Launches create marketing, sales, and tooling needs. |
| Expansion | 15 | New regions or markets often need new vendors and infrastructure. |
| Partnership | 10 | Partnerships show ecosystem activity and openness. |

Each signal has a confidence multiplier:

| Confidence | Multiplier |
|------------|------------|
| High | 1.0 |
| Medium | 0.6 |
| Low | 0.3 |

The final score is capped at 100 and includes a written justification.

## User Interface

The Streamlit dashboard provides:

- Company, website, product, and scenario inputs
- Quick examples
- Pipeline progress display
- Overview, competitor, marketing, lead, SWOT, and sources tabs
- Lead score breakdown
- Detected signal groups
- Signal timeline chart
- Confidence badges
- Sources and agent execution logs
- Markdown report download
- PDF report download
- Streaming executive briefing generation
- Run history
- Side-by-side comparison of past runs
- Batch analysis from pasted names or CSV
- LLM cache clear button

The project also supports CLI mode through `app.py`.

## Report Outputs

The final report is business-readable and includes:

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
- Lead score and score breakdown
- Detected lead signals
- Outreach recommendations
- SWOT analysis
- Source references

Reports are saved as markdown under `reports/`. The dashboard can also export PDF bytes using WeasyPrint.

## Performance Improvements

The current project includes:

- Parallel execution for independent analysis agents
- Disk-backed LLM response caching
- Streaming LLM responses for executive briefings in the dashboard
- ChromaDB persistence
- Configurable RAG chunk size and top-k retrieval
- Paragraph/sentence-aware chunking

These improvements make repeated runs faster and reduce unnecessary LLM calls.

## Testing and Evaluation

The project includes tests for:

- Pydantic state models
- Lead scoring
- LLM cache
- Report saving and run history
- Scraper helpers with network calls mocked
- Signal date parsing for timeline charts
- Confidence badge helpers
- Mocked pipeline integration
- Parallel state merge behavior

Current local status:

- `99 passed, 1 skipped`.
- Public enrichment connectors are covered with mocked unit tests.
- Streaming cache replay is covered without calling the live LLM API.

The hackathon PDF asks for a manual evaluation sheet. That should still be added as a separate file before final submission.

## AI Usage Disclosure

AI is used in these places:

- Competitor identification
- Competitor analysis
- Product and messaging analysis
- Signal extraction
- Outreach recommendations
- Cold email draft generation
- SWOT generation
- Executive summary generation
- Code/design assistance during development

Embeddings are used for semantic retrieval from public scraped text.

## Guardrails and Ethics

The project follows these constraints:

- Uses public data only.
- Does not scrape login-protected or private sources.
- Does not include hard-coded API secrets.
- Uses `.env.example` for configuration documentation.
- Keeps `.env` ignored through `.gitignore`.
- Shows sources where available.
- Shows confidence indicators to avoid overclaiming certainty.

## Limitations

- Some websites block automated scraping.
- JavaScript-heavy sites may not expose useful HTML content.
- News freshness depends on public search results.
- Lead scoring is heuristic, not trained from CRM conversion data.
- Funding, hiring, review, launch, feed, and traffic signals are extracted from public pages and snippets, so they should be treated as directional rather than fully verified.
- Competitor discovery can be imperfect if public context is thin.
- LLM outputs may still need manual review before business use.
- LinkedIn Jobs, Crunchbase, Tracxn, G2, Capterra, Product Hunt, RSS, and SimilarWeb connectors are best-effort public-page/snippet collectors, not official paid API integrations.

## Future Improvements

High-value future work:

- Add a manual evaluation sheet for known companies and expected outputs.
- Add a recorded walkthrough for submission.
- Replace best-effort public-page collectors with official APIs where legally permitted.
- Improve source-specific parsing for jobs, funding, reviews, launch, traffic, and feed data.
- Calibrate lead scoring using historical conversion examples.
- Add scheduled monitoring and alerts when scores change.
- Add CRM export for Salesforce or HubSpot.
- Improve structured-output reliability with stricter schemas or tool calling.

## Summary

This project implements a focused business intelligence agent for sales and product marketing research. It collects public data, grounds LLM analysis with RAG, runs specialized agents, scores lead opportunities, and produces a report that is useful for outreach and competitive understanding.

The strongest parts are the end-to-end workflow, multi-agent design, RAG grounding, lead scoring, dashboard, and report generation. The biggest remaining submission tasks are adding a manual evaluation sheet and a recorded walkthrough/demo artifact.
