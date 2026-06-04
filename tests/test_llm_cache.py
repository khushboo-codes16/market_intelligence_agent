# tests/test_llm_cache.py
# ============================================================
# Unit tests: LLM response cache (no Groq API calls)
# ============================================================

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestLLMCache:
    """Test cache logic by patching _CACHE_FILE to a temp path and Groq client."""

    def _make_patched_client(self, tmp_path: Path):
        """Return an LLMClient with Groq and cache file patched."""
        cache_file = tmp_path / "test_cache.json"

        with patch("tools.llm_client.GROQ_API_KEY", "test-key"), \
             patch("tools.llm_client._CACHE_FILE", cache_file), \
             patch("tools.llm_client._memory_cache", {}), \
             patch("groq.Groq") as mock_groq:

            mock_response = MagicMock()
            mock_response.choices[0].message.content = "mocked LLM response"
            mock_groq.return_value.chat.completions.create.return_value = mock_response

            from tools.llm_client import LLMClient
            client = LLMClient.__new__(LLMClient)
            client.client = mock_groq.return_value
            client.model = "test-model"
            client.temperature = 0.3
            client.max_tokens = 100
            return client, cache_file, mock_groq

    def test_cache_key_is_deterministic(self):
        from tools.llm_client import _cache_key
        key1 = _cache_key("hello world", "system prompt")
        key2 = _cache_key("hello world", "system prompt")
        assert key1 == key2

    def test_cache_key_differs_for_different_prompts(self):
        from tools.llm_client import _cache_key
        assert _cache_key("prompt A") != _cache_key("prompt B")

    def test_cache_key_differs_for_different_system_prompts(self):
        from tools.llm_client import _cache_key
        assert _cache_key("same", "sys A") != _cache_key("same", "sys B")

    def test_clear_cache_returns_count(self, tmp_path):
        import json
        from tools.llm_client import _cache_key
        cache_file = tmp_path / "cache.json"
        test_cache = {_cache_key("p1"): "r1", _cache_key("p2"): "r2"}
        cache_file.write_text(json.dumps(test_cache))

        with patch("tools.llm_client._CACHE_FILE", cache_file), \
             patch("tools.llm_client._memory_cache", dict(test_cache)):
            from tools.llm_client import clear_llm_cache
            count = clear_llm_cache()

        assert count == 2
        assert not cache_file.exists()

    def test_use_cache_false_bypasses_cache(self, tmp_path):
        """use_cache=False should always call the LLM, never read from cache."""
        from tools.llm_client import _cache_key, _memory_cache
        key = _cache_key("test prompt", "")
        pre_populated = {key: "cached value"}

        with patch("tools.llm_client._memory_cache", pre_populated), \
             patch("tools.llm_client.GROQ_API_KEY", "test-key"), \
             patch("tools.llm_client._CACHE_FILE", tmp_path / "c.json"):

            mock_resp = MagicMock()
            mock_resp.choices[0].message.content = "fresh LLM value"

            from tools.llm_client import LLMClient
            client = LLMClient.__new__(LLMClient)
            client.model = "m"
            client.temperature = 0.3
            client.max_tokens = 100
            client.client = MagicMock()
            client.client.chat.completions.create.return_value = mock_resp

            result = client.complete("test prompt", use_cache=False)
            assert result == "fresh LLM value"
            # Confirm API was actually called
            client.client.chat.completions.create.assert_called_once()

    def test_stream_complete_replays_cached_value(self):
        """Cached streaming should yield text chunks without calling Groq."""
        from tools.llm_client import _cache_key
        key = _cache_key("stream prompt", "")
        cached = {key: "This is a cached streaming response for the dashboard."}

        with patch("tools.llm_client._memory_cache", cached):
            from tools.llm_client import LLMClient
            client = LLMClient.__new__(LLMClient)
            client.client = MagicMock()
            client.model = "m"
            client.temperature = 0.3
            client.max_tokens = 100

            chunks = list(client.stream_complete("stream prompt"))

        assert "".join(chunks) == cached[key]
        client.client.chat.completions.create.assert_not_called()
