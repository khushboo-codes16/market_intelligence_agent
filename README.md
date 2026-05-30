# 🔍 AI-Powered Business & Market Intelligence Agent

> **DS Intern Hackathon — May 2026**  
> Automated competitive intelligence and lead scoring powered by LangGraph, Groq, and RAG.

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     STREAMLIT DASHBOARD (ui/)                        │
│  Input: Company Name, Website, Product, Scenario                     │
│  Output: Overview · Competitors · Marketing · Leads · SWOT · Sources│
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│              LANGGRAPH SUPERVISOR ORCHESTRATOR (workflow/)           │
│                                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐ │
│  │ Research │→│ Product  │→│Competitor│→│ Lead Gen │→│Report│ │
│  │  Agent   │  │Marketing │  │Analysis  │  │  Agent   │  │Agent │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──┬───┘ │
└───────┼─────────────┼─────────────┼─────────────┼────────────┼─────┘
        │             │             │             │            │
        ▼             ▼             ▼             ▼            ▼
┌───────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│  SCRAPER  │  │  GROQ    │  │  GROQ    │  │  GROQ    │  │  GROQ    │
│  + News   │  │  LLM     │  │  LLM     │  │  LLM     │  │  LLM     │
│  (tools/) │  │(Llama 3) │  │(Llama 3) │  │(Llama 3) │  │(Llama 3) │
└───────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    RAG PIPELINE (rag/)                               │
│  ChromaDB (vector store) + SentenceTransformers (embeddings)        │
│  Chunk → Embed → Store → Retrieve → Context → LLM                  │
└─────────────────────────────────────────────────────────────────────┘
```

### Agent Descriptions

| Agent | Role |
|-------|------|
| **Supervisor Agent** | Orchestrates the LangGraph workflow, routes between agents based on status |
| **Research Agent** | Scrapes company website + news articles, ingests into ChromaDB RAG |
| **Product Marketing Agent** | Analyzes positioning, messaging, value props, pricing, differentiation |
| **Competitor Analysis Agent** | Discovers competitors, profiles them, builds comparison tables |
| **Lead Generation Agent** | Extracts buying signals, computes lead score, generates outreach recs |
| **Report Generation Agent** | Compiles SWOT, executive summary, and the final BI report |

---

## 🚀 Quick Start

### 1. Clone and Set Up

```bash
git clone <repo-url>
cd market_intel

# Create conda environment
conda create --name agent python=3.11
conda activate agent  

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Get your free Groq API key at: https://console.groq.com

### 3. Run Streamlit Dashboard

```bash
streamlit run ui/dashboard.py
```

The app will open at `http://localhost:8501`

### 4. Run CLI Mode (Optional)

```bash
python app.py OpenAI --website https://openai.com --product "ChatGPT"
python app.py ElevenLabs --scenario "AI voice agent for lead generation"
python app.py Salesforce --product "Salesforce Einstein"
```

---

## 📁 Project Structure

```
market_intel/
│
├── agents/                         # Multi-agent implementations
│   ├── __init__.py
│   ├── research_agent.py           # Data collection & RAG ingestion
│   ├── product_marketing_agent.py  # Marketing intelligence analysis
│   ├── competitor_analysis_agent.py # Competitor discovery & profiling
│   ├── lead_generation_agent.py    # Signal extraction & lead scoring
│   └── report_generation_agent.py  # Final report compilation
│
├── workflow/                       # LangGraph orchestration
│   ├── __init__.py
│   └── graph.py                    # Supervisor + StateGraph definition
│
├── tools/                          # Reusable tool modules
│   ├── __init__.py
│   ├── scraper.py                  # Web scraping (BS4 + Trafilatura)
│   └── llm_client.py              # Groq API wrapper
│
├── rag/                            # RAG pipeline
│   ├── __init__.py
│   └── rag_pipeline.py            # ChromaDB + SentenceTransformers
│
├── models/                         # Pydantic state models
│   ├── __init__.py
│   └── state.py                   # IntelligenceState + sub-models
│
├── ui/                             # Streamlit frontend
│   ├── __init__.py
│   └── dashboard.py               # Full multi-tab dashboard
│
├── reports/                        # Report management
│   ├── __init__.py
│   └── report_saver.py
│
├── config/                         # Configuration management
│   ├── __init__.py
│   └── settings.py                # Env-var based settings
│
├── utils/                          # Shared utilities
│   ├── __init__.py
│   ├── logger.py                  # Loguru logger
│   └── text_utils.py              # Cleaning, chunking, normalization
│
├── data/                           # Runtime data (gitignored)
│   └── chroma_db/                 # ChromaDB persistence
│
├── app.py                         # Main entry point
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🔧 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | **Required** | Your Groq API key |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Groq model to use |
| `LLM_TEMPERATURE` | `0.3` | LLM sampling temperature |
| `LLM_MAX_TOKENS` | `4096` | Max tokens per LLM call |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | ChromaDB storage path |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformers model |
| `CHUNK_SIZE` | `800` | RAG chunk size (chars) |
| `CHUNK_OVERLAP` | `150` | RAG chunk overlap (chars) |
| `TOP_K_RESULTS` | `5` | RAG retrieval top-k |
| `REQUEST_TIMEOUT` | `15` | HTTP request timeout (s) |
| `REQUEST_DELAY` | `1.5` | Delay between scrape requests (s) |
| `MAX_PAGES_PER_DOMAIN` | `5` | Max pages to scrape per site |

---

## 📊 Sample Inputs & Outputs

### Input
```
Company Name: ElevenLabs
Website: https://elevenlabs.io
Scenario: A company launching a voice-based AI agent for taking calls and generating leads
```

### Output Structure
1. **Executive Summary** — Boardroom-ready 3-paragraph overview
2. **Company Overview** — Mission, ICP, scale
3. **Products & Services** — Feature breakdown
4. **Competitor Analysis** — Table + profiles (Murf, Resemble, PlayHT, etc.)
5. **Product Positioning** — Positioning themes and target audience
6. **Market Messaging** — Brand voice, taglines, CTAs
7. **Pricing Insights** — Tiers and strategy
8. **Recent Activities** — Funding, launches, news
9. **Lead Signals** — Specific hiring, funding, launch signals
10. **Lead Score** — 0-100 with breakdown + justification
11. **Outreach Recommendations** — Personalized angles and talking points
12. **SWOT Analysis** — 4-quadrant strategic assessment
13. **Sources** — All URLs scraped and referenced

---

## 🧠 AI Usage Disclosure

As required by hackathon rules, here's where AI was used:

| Component | AI Usage |
|-----------|----------|
| Competitor identification | LLM (Groq Llama 3.3) |
| Company/product analysis | LLM with RAG context |
| Signal extraction | LLM structured output |
| Lead score justification | LLM text generation |
| SWOT generation | LLM structured output |
| Executive summary | LLM text generation |
| Outreach recommendations | LLM text generation |
| Text embedding | SentenceTransformers |
| Code assistance | Claude (disclosed) |

---

## 🐛 Troubleshooting

### "GROQ_API_KEY is not set"
→ add your key from https://console.groq.com

### "No module named 'chromadb'"
→ Run: `pip install -r requirements.txt`

### "Could not fetch homepage"
→ The company website may block scrapers. Try providing a different URL or leave blank.

### Slow first run
→ First run downloads the SentenceTransformers embedding model (~90MB). Cached after first use.

### ChromaDB errors
→ Delete `./data/chroma_db` directory and restart.

### Low-quality output for obscure companies
→ Provide the website URL explicitly. The LLM will use its knowledge when scraping fails.

---

## 🔮 Architecture Decisions

1. **LangGraph over plain Python** — State machine makes agent coordination explicit and debuggable
2. **Groq (not OpenAI)** — Free tier available, fastest inference for hackathon
3. **ChromaDB + SentenceTransformers** — Local, no API cost, fast for our data volumes
4. **Trafilatura + BS4 fallback** — Better content extraction than BS4 alone
5. **DuckDuckGo HTML** — No API key required for news search
6. **Pydantic state models** — Type safety throughout the pipeline
7. **Per-company collections** — Each run gets its own ChromaDB collection, no cross-contamination

---

## 📈 Lead Scoring System

The lead score (0-100) is computed from 5 signal categories:

| Signal | Max Score | Rationale |
|--------|-----------|-----------|
| Funding | 30 pts | Strongest buy signal — fresh capital = budget available |
| Hiring | 25 pts | Growth mode = tool purchases, willingness to try new vendors |
| Product Launch | 20 pts | Need for marketing/sales enablement tools |
| Expansion | 15 pts | New markets = new infrastructure needs |
| Partnerships | 10 pts | Ecosystem openness, integration readiness |

Each signal is multiplied by a confidence factor (high: 1.0, medium: 0.6, low: 0.3).

---

## 👨‍💻 Author

Built for DS Intern Hackathon — May 2026.  
Tech stack: Python · LangGraph · Groq · ChromaDB · SentenceTransformers · Streamlit · BeautifulSoup · Trafilatura
