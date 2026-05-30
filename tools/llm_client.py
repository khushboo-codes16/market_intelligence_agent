# tools/llm_client.py
# ============================================================
# Groq LLM client with retry and token management
# ============================================================

from typing import Optional, List, Dict
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS
from utils.logger import logger


class LLMClient:
    """Thin wrapper around Groq SDK with retry logic."""

    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set. Please configure your .env file.")
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = LLM_MODEL
        self.temperature = LLM_TEMPERATURE
        self.max_tokens = LLM_MAX_TOKENS

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
    ) -> str:
        """Simplified single-turn completion."""
        return self.chat(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )


# Module-level singleton
_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
