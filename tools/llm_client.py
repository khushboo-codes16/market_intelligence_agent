# tools/llm_client.py
# ============================================================
# Groq LLM client with retry and token management
# ============================================================
# STEP 2 IMPROVEMENT: LLM response cache
# Responses are keyed by (prompt + system_prompt) hash.
# Re-running the same company costs zero API calls.
# Cache persists in memory for the process lifetime.
# ============================================================

import hashlib
import json
from pathlib import Path
from typing import Iterator, Optional, List, Dict

from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS
from utils.logger import logger

# Disk-backed cache file so it survives restarts
_CACHE_FILE = Path(__file__).resolve().parent.parent / "data" / "llm_cache.json"
_memory_cache: Dict[str, str] = {}


def _load_cache() -> None:
    """Load cache from disk into memory on first use."""
    global _memory_cache
    if _CACHE_FILE.exists():
        try:
            _memory_cache = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
            logger.info(f"[LLMCache] Loaded {len(_memory_cache)} cached responses from disk.")
        except Exception:
            _memory_cache = {}


def _save_cache() -> None:
    """Persist memory cache to disk."""
    try:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_FILE.write_text(
            json.dumps(_memory_cache, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as e:
        logger.warning(f"[LLMCache] Could not save cache: {e}")


def _cache_key(prompt: str, system_prompt: str = "") -> str:
    raw = f"{system_prompt}||{prompt}"
    return hashlib.md5(raw.encode()).hexdigest()


def clear_llm_cache() -> int:
    """Clear all cached LLM responses. Returns number of entries cleared."""
    global _memory_cache
    count = len(_memory_cache)
    _memory_cache = {}
    if _CACHE_FILE.exists():
        _CACHE_FILE.unlink()
    logger.info(f"[LLMCache] Cleared {count} cached entries.")
    return count


class LLMClient:
    """Thin wrapper around Groq SDK with retry logic and response caching."""

    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set. Please configure your .env file.")
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = LLM_MODEL
        self.temperature = LLM_TEMPERATURE
        self.max_tokens = LLM_MAX_TOKENS
        _load_cache()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Send a chat completion request and return the response text."""
        all_messages = []
        if system_prompt:
            all_messages.append({"role": "system", "content": system_prompt})
        all_messages.extend(messages)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=all_messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_cache: bool = True,
    ) -> str:
        """
        Single-turn completion with optional caching.
        Cache is skipped when use_cache=False (e.g. for real-time signal extraction).
        """
        key = _cache_key(prompt, system_prompt or "")

        if use_cache and key in _memory_cache:
            logger.debug(f"[LLMCache] HIT — returning cached response.")
            return _memory_cache[key]

        result = self.chat(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if use_cache:
            _memory_cache[key] = result
            _save_cache()

        return result

    def stream_complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_cache: bool = True,
    ) -> Iterator[str]:
        """
        Stream a single-turn completion chunk by chunk.

        If a cached response exists, it is replayed in sentence-sized chunks.
        Otherwise chunks are yielded from Groq's streaming API and the final
        response is saved to cache after streaming completes.
        """
        key = _cache_key(prompt, system_prompt or "")

        if use_cache and key in _memory_cache:
            cached = _memory_cache[key]
            for chunk in _replay_chunks(cached):
                yield chunk
            return

        all_messages = []
        if system_prompt:
            all_messages.append({"role": "system", "content": system_prompt})
        all_messages.append({"role": "user", "content": prompt})

        collected: List[str] = []
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=all_messages,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                stream=True,
            )
            for event in stream:
                delta = event.choices[0].delta.content or ""
                if not delta:
                    continue
                collected.append(delta)
                yield delta
        except Exception as e:
            logger.error(f"Streaming LLM call failed: {e}")
            raise

        if use_cache and collected:
            _memory_cache[key] = "".join(collected)
            _save_cache()


# Module-level singleton
_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


def _replay_chunks(text: str, chunk_size: int = 180) -> Iterator[str]:
    """Replay cached text in readable chunks for Streamlit streaming."""
    if not text:
        return
    parts = []
    current = ""
    for piece in text.split(" "):
        if len(current) + len(piece) + 1 > chunk_size:
            parts.append(current + " ")
            current = piece
        else:
            current = f"{current} {piece}".strip()
    if current:
        parts.append(current)
    for part in parts:
        yield part
