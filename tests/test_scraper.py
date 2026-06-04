# tests/test_scraper.py
# ============================================================
# Unit tests: scraper utilities (no real network calls)
# ============================================================

import pytest
from unittest.mock import patch, MagicMock


class TestFetchUrl:
    @patch("tools.scraper.requests.get")
    def test_returns_html_on_success(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = "<html><body>Hello</body></html>"
        mock_get.return_value.raise_for_status = MagicMock()

        from tools.scraper import fetch_url
        result = fetch_url("https://example.com")
        assert result is not None
        assert "Hello" in result

    @patch("tools.scraper.requests.get")
    def test_returns_none_on_http_error(self, mock_get):
        import requests
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.return_value.raise_for_status.side_effect = requests.HTTPError(response=mock_resp)

        from tools.scraper import fetch_url
        result = fetch_url("https://example.com/404")
        assert result is None


class TestExtractText:
    def test_bs4_extracts_text(self):
        from tools.scraper import extract_text_bs4
        html = "<html><body><p>Hello world this is test content for extraction.</p></body></html>"
        text = extract_text_bs4(html)
        assert "Hello world" in text

    def test_bs4_removes_script_tags(self):
        from tools.scraper import extract_text_bs4
        html = "<html><body><script>alert('bad')</script><p>Good content</p></body></html>"
        text = extract_text_bs4(html)
        assert "alert" not in text
        assert "Good content" in text


class TestDiscoverInternalLinks:
    def test_finds_pricing_links(self):
        from tools.scraper import discover_internal_links
        html = """<html><body>
            <a href="/pricing">Pricing</a>
            <a href="/about">About</a>
            <a href="https://other.com/page">External</a>
        </body></html>"""
        links = discover_internal_links("https://example.com", html)
        assert any("pricing" in l for l in links)

    def test_excludes_external_links(self):
        from tools.scraper import discover_internal_links
        html = '<html><body><a href="https://external.com/pricing">Ext</a></body></html>'
        links = discover_internal_links("https://example.com", html)
        assert not any("external.com" in l for l in links)

    def test_excludes_fragment_links(self):
        from tools.scraper import discover_internal_links
        html = '<html><body><a href="/pricing#section">Frag</a></body></html>'
        links = discover_internal_links("https://example.com", html)
        # Fragment links should be excluded
        assert not any("#" in l for l in links)

    def test_respects_max_links(self):
        from tools.scraper import discover_internal_links
        links_html = "".join(
            f'<a href="/pricing-{i}">P{i}</a>' for i in range(30)
        )
        html = f"<html><body>{links_html}</body></html>"
        links = discover_internal_links("https://example.com", html, max_links=5)
        assert len(links) <= 5


class TestSearchWebNews:
    @patch("tools.scraper.requests.get")
    def test_returns_list_on_success(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = """
        <html><body>
        <div class="result">
            <h2 class="result__title"><a href="https://techcrunch.com/story">TC Story</a></h2>
            <a class="result__url">techcrunch.com/story</a>
            <a class="result__snippet">Great news story about funding</a>
        </div>
        </body></html>
        """
        from tools.scraper import search_web_news
        results = search_web_news("test company funding")
        assert isinstance(results, list)

    @patch("tools.scraper.requests.get")
    def test_returns_empty_on_error(self, mock_get):
        mock_get.side_effect = Exception("Network error")
        from tools.scraper import search_web_news
        results = search_web_news("test query")
        assert results == []
