# config/settings.py
# ============================================================
# Centralized configuration management
# ============================================================

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# ── LLM ──────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))

# ── ChromaDB ─────────────────────────────────────────────────
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "data" / "chroma_db"))
CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "market_intel")

# ── Embeddings ───────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── Web Scraping ─────────────────────────────────────────────
REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "15"))
REQUEST_DELAY: float = float(os.getenv("REQUEST_DELAY", "1.5"))
MAX_PAGES_PER_DOMAIN: int = int(os.getenv("MAX_PAGES_PER_DOMAIN", "5"))
USER_AGENT: str = os.getenv(
    "USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# ── RAG ──────────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))

# ── Lead Scoring Weights ─────────────────────────────────────
LEAD_SCORE_HIRING_WEIGHT: int = int(os.getenv("LEAD_SCORE_HIRING_WEIGHT", "25"))
LEAD_SCORE_FUNDING_WEIGHT: int = int(os.getenv("LEAD_SCORE_FUNDING_WEIGHT", "30"))
LEAD_SCORE_LAUNCH_WEIGHT: int = int(os.getenv("LEAD_SCORE_LAUNCH_WEIGHT", "20"))
LEAD_SCORE_EXPANSION_WEIGHT: int = int(os.getenv("LEAD_SCORE_EXPANSION_WEIGHT", "15"))
LEAD_SCORE_PARTNERSHIP_WEIGHT: int = int(os.getenv("LEAD_SCORE_PARTNERSHIP_WEIGHT", "10"))

# ── Logging ──────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE: str = os.getenv("LOG_FILE", str(BASE_DIR / "data" / "app.log"))

# ── Reports ──────────────────────────────────────────────────
REPORTS_DIR: str = os.getenv("REPORTS_DIR", str(BASE_DIR / "reports"))

# ── Ensure Directories Exist ─────────────────────────────────
for _dir in [CHROMA_PERSIST_DIR, REPORTS_DIR, str(BASE_DIR / "data")]:
    Path(_dir).mkdir(parents=True, exist_ok=True)
