# tests/test_public_sources.py
# ============================================================
# Unit tests for best-effort public enrichment connectors
# ============================================================

from unittest.mock import patch


def test_funding_sources_uses_public_search_snippets():
    from tools.public_sources import scrape_funding_sources

    with patch("tools.public_sources.fetch_url", return_value=None), \
         patch("tools.public_sources.search_web_news", return_value=[
             {
                 "title": "TestCo raises Series B",
                 "url": "https://example.com/funding",
                 "snippet": "TestCo raised $20M from growth investors.",
             }
         ]):
        docs = scrape_funding_sources("TestCo")

    assert len(docs) == 1
    assert docs[0]["source_type"] == "funding"
    assert "Series B" in docs[0]["text"]


def test_linkedin_jobs_public_handles_blocked_direct_page():
    from tools.public_sources import scrape_linkedin_jobs_public

    with patch("tools.public_sources.fetch_url", return_value=None), \
         patch("tools.public_sources.search_web_news", return_value=[
             {
                 "title": "TestCo jobs on LinkedIn",
                 "url": "https://www.linkedin.com/jobs/view/123",
                 "snippet": "Hiring account executives and product engineers.",
             }
         ]):
        docs = scrape_linkedin_jobs_public("TestCo")

    assert len(docs) == 1
    assert docs[0]["source_type"] == "jobs"
    assert "account executives" in docs[0]["text"]


def test_rss_feed_parser_extracts_items():
    from tools.public_sources import scrape_rss_and_press_feeds

    homepage_html = """
    <html><head>
      <link rel="alternate" type="application/rss+xml" href="/feed.xml" />
    </head><body></body></html>
    """
    feed_xml = """
    <rss><channel>
      <item>
        <title>TestCo launches new automation product</title>
        <link>https://testco.com/blog/launch</link>
        <pubDate>Mon, 01 Jun 2026 10:00:00 GMT</pubDate>
        <description>Launch announcement with customer details.</description>
      </item>
    </channel></rss>
    """

    def fake_fetch(url):
        if url == "https://testco.com":
            return homepage_html
        if url == "https://testco.com/feed.xml":
            return feed_xml
        return None

    with patch("tools.public_sources.fetch_url", side_effect=fake_fetch):
        docs = scrape_rss_and_press_feeds("TestCo", "https://testco.com")

    assert len(docs) == 1
    assert docs[0]["source_type"] == "rss"
    assert "launches new automation product" in docs[0]["text"]


def test_collect_public_sources_dedupes_and_survives_failures():
    from tools.public_sources import collect_public_intelligence_sources

    doc = {
        "url": "public-search:reviews",
        "title": "Review snippets",
        "text": "Customers mention pricing, support, implementation, and alternatives. " * 3,
        "source_type": "reviews",
    }

    with patch("tools.public_sources.scrape_linkedin_jobs_public", side_effect=Exception("blocked")), \
         patch("tools.public_sources.scrape_funding_sources", return_value=[doc]), \
         patch("tools.public_sources.scrape_review_sources", return_value=[doc]), \
         patch("tools.public_sources.scrape_product_hunt_sources", return_value=[]), \
         patch("tools.public_sources.scrape_rss_and_press_feeds", return_value=[]), \
         patch("tools.public_sources.scrape_similarweb_public", return_value=[]):
        docs = collect_public_intelligence_sources("TestCo", "https://testco.com")

    assert len(docs) == 1
    assert docs[0]["source_type"] == "reviews"
