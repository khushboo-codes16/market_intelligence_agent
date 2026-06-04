# tools/public_sources.py
# ============================================================
# Best-effort public data connectors for extra market signals
# ============================================================

import re
import xml.etree.ElementTree as ET
from collections import Counter
from typing import Dict, List, Optional
from urllib.parse import quote_plus, urljoin, urlparse

from bs4 import BeautifulSoup

from tools.scraper import fetch_url, extract_text_bs4, search_web_news
from utils.logger import logger
from utils.text_utils import clean_text


PUBLIC_SOURCE_TYPES = {
    "jobs",
    "funding",
    "reviews",
    "product_hunt",
    "rss",
    "traffic",
}


def collect_public_intelligence_sources(
    company_name: str,
    company_website: str = "",
) -> List[Dict[str, str]]:
    """
    Collect optional public enrichment documents.

    These connectors are intentionally best-effort: many public sources block
    automated access or expose limited HTML. A failed source returns no docs
    and never stops the main pipeline.
    """
    collectors = [
        scrape_linkedin_jobs_public,
        scrape_funding_sources,
        scrape_review_sources,
        scrape_product_hunt_sources,
        scrape_rss_and_press_feeds,
        scrape_similarweb_public,
    ]

    docs: List[Dict[str, str]] = []
    for collector in collectors:
        try:
            if collector is scrape_rss_and_press_feeds:
                docs.extend(collector(company_name, company_website))
            elif collector is scrape_similarweb_public:
                docs.extend(collector(company_name, company_website))
            else:
                docs.extend(collector(company_name))
        except Exception as exc:
            collector_name = getattr(collector, "__name__", repr(collector))
            logger.warning(f"[PublicSources] {collector_name} failed: {exc}")

    return _dedupe_docs(docs)


def scrape_linkedin_jobs_public(company_name: str) -> List[Dict[str, str]]:
    """
    Collect public LinkedIn Jobs signals.

    LinkedIn frequently blocks direct automated HTML access. We still try the
    public jobs search page and supplement with public search snippets. No
    login-protected data is accessed.
    """
    docs: List[Dict[str, str]] = []
    query = quote_plus(company_name)
    url = f"https://www.linkedin.com/jobs/search/?keywords={query}"
    html = fetch_url(url)

    if html:
        soup = BeautifulSoup(html, "lxml")
        text = extract_text_bs4(html)
        titles = _extract_job_titles(soup)
        summary_parts = [
            f"Public LinkedIn Jobs page for {company_name}.",
            f"Detected job title snippets: {', '.join(titles[:12])}" if titles else "",
            text[:2500],
        ]
        docs.append(_make_doc(
            source_type="jobs",
            url=url,
            title=f"{company_name} LinkedIn Jobs",
            text="\n".join(part for part in summary_parts if part),
        ))

    snippets = search_web_news(
        f'site:linkedin.com/jobs "{company_name}" jobs hiring',
        num_results=5,
    )
    snippet_text = _format_search_results(snippets)
    if snippet_text:
        docs.append(_make_doc(
            source_type="jobs",
            url="https://www.linkedin.com/jobs/",
            title=f"{company_name} LinkedIn Jobs search snippets",
            text=(
                f"Public search snippets related to {company_name} LinkedIn job postings.\n"
                f"{snippet_text}"
            ),
        ))

    return docs


def scrape_funding_sources(company_name: str) -> List[Dict[str, str]]:
    """Collect public funding signals from Crunchbase, Tracxn, and search snippets."""
    docs: List[Dict[str, str]] = []
    slug = _company_slug(company_name)
    candidate_urls = [
        f"https://www.crunchbase.com/organization/{slug}",
        f"https://tracxn.com/d/companies/{slug}",
    ]

    for url in candidate_urls:
        page = _fetch_public_page(url)
        if page:
            page["source_type"] = "funding"
            docs.append(page)

    snippets = search_web_news(
        f'"{company_name}" funding raised series investors crunchbase tracxn',
        num_results=6,
    )
    snippet_text = _format_search_results(snippets)
    if snippet_text:
        docs.append(_make_doc(
            source_type="funding",
            url="public-search:funding",
            title=f"{company_name} funding search snippets",
            text=(
                f"Public funding-related snippets for {company_name}.\n"
                f"{snippet_text}"
            ),
        ))

    return docs


def scrape_review_sources(company_name: str) -> List[Dict[str, str]]:
    """Collect public review/pain-point signals from G2 and Capterra pages/snippets."""
    docs: List[Dict[str, str]] = []
    slug = _company_slug(company_name)
    candidate_urls = [
        f"https://www.g2.com/products/{slug}/reviews",
        f"https://www.capterra.com/p/{slug}/",
    ]

    for url in candidate_urls:
        page = _fetch_public_page(url)
        if page:
            page["source_type"] = "reviews"
            docs.append(page)

    snippets = search_web_news(
        f'"{company_name}" reviews G2 Capterra pros cons pricing alternatives',
        num_results=6,
    )
    snippet_text = _format_search_results(snippets)
    if snippet_text:
        docs.append(_make_doc(
            source_type="reviews",
            url="public-search:reviews",
            title=f"{company_name} review search snippets",
            text=(
                f"Public review snippets for {company_name}; useful for pain points, "
                f"customer language, alternatives, and market perception.\n{snippet_text}"
            ),
        ))

    return docs


def scrape_product_hunt_sources(company_name: str) -> List[Dict[str, str]]:
    """Collect public Product Hunt launch/community signals."""
    docs: List[Dict[str, str]] = []
    slug = _company_slug(company_name)
    candidate_urls = [
        f"https://www.producthunt.com/products/{slug}",
        f"https://www.producthunt.com/search?q={quote_plus(company_name)}",
    ]

    for url in candidate_urls:
        page = _fetch_public_page(url)
        if page:
            page["source_type"] = "product_hunt"
            docs.append(page)

    snippets = search_web_news(
        f'site:producthunt.com "{company_name}" launch product hunt upvotes',
        num_results=5,
    )
    snippet_text = _format_search_results(snippets)
    if snippet_text:
        docs.append(_make_doc(
            source_type="product_hunt",
            url="public-search:product-hunt",
            title=f"{company_name} Product Hunt snippets",
            text=(
                f"Public Product Hunt launch/community snippets for {company_name}.\n"
                f"{snippet_text}"
            ),
        ))

    return docs


def scrape_rss_and_press_feeds(company_name: str, company_website: str = "") -> List[Dict[str, str]]:
    """Discover and parse public RSS/Atom/press feeds from the company site."""
    docs: List[Dict[str, str]] = []
    if not company_website:
        return docs

    homepage = _normalise_url(company_website)
    html = fetch_url(homepage)
    feed_urls = _discover_feed_urls(homepage, html or "")

    for feed_url in feed_urls[:5]:
        feed_html = fetch_url(feed_url)
        if not feed_html:
            continue
        entries = _parse_feed_entries(feed_html, limit=8)
        if not entries:
            continue
        text = "\n\n".join(
            f"- {entry['title']}\n  Date: {entry['date'] or 'N/A'}\n  Link: {entry['link']}\n  {entry['summary']}"
            for entry in entries
        )
        docs.append(_make_doc(
            source_type="rss",
            url=feed_url,
            title=f"{company_name} RSS/press feed",
            text=f"Recent public RSS/press feed items for {company_name}:\n{text}",
        ))

    return docs


def scrape_similarweb_public(company_name: str, company_website: str = "") -> List[Dict[str, str]]:
    """Collect public traffic/category snippets from SimilarWeb pages/search."""
    docs: List[Dict[str, str]] = []
    domain = _domain_from_url(company_website)
    if domain:
        url = f"https://www.similarweb.com/website/{domain}/"
        page = _fetch_public_page(url)
        if page:
            page["source_type"] = "traffic"
            docs.append(page)

    snippets = search_web_news(
        f'"{company_name}" similarweb traffic ranking visits growth',
        num_results=5,
    )
    snippet_text = _format_search_results(snippets)
    if snippet_text:
        docs.append(_make_doc(
            source_type="traffic",
            url="public-search:traffic",
            title=f"{company_name} traffic estimate snippets",
            text=(
                f"Public traffic/ranking snippets for {company_name}. Treat these as directional, "
                f"not verified analytics.\n{snippet_text}"
            ),
        ))

    return docs


def summarise_public_source_counts(docs: List[Dict[str, str]]) -> Dict[str, int]:
    """Return a source_type -> count summary for logs/UI metadata."""
    return dict(Counter(doc.get("source_type", "unknown") for doc in docs))


def _fetch_public_page(url: str) -> Optional[Dict[str, str]]:
    html = fetch_url(url)
    if not html:
        return None

    soup = BeautifulSoup(html, "lxml")
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else url
    text = extract_text_bs4(html)
    if len(text) < 120:
        return None

    return _make_doc(
        source_type="public",
        url=url,
        title=title,
        text=text[:3500],
    )


def _make_doc(source_type: str, url: str, title: str, text: str) -> Dict[str, str]:
    return {
        "url": url,
        "title": title,
        "text": clean_text(text),
        "meta_description": "",
        "source_type": source_type,
    }


def _dedupe_docs(docs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    deduped: List[Dict[str, str]] = []
    seen = set()
    for doc in docs:
        text = doc.get("text", "")
        if len(text) < 80:
            continue
        key = (doc.get("url", ""), doc.get("source_type", ""), text[:120])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(doc)
    return deduped


def _company_slug(company_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", company_name.lower()).strip("-")
    return slug or "company"


def _normalise_url(url: str) -> str:
    return url if url.startswith(("http://", "https://")) else f"https://{url}"


def _domain_from_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(_normalise_url(url))
    return parsed.netloc.lower().removeprefix("www.")


def _format_search_results(results: List[Dict[str, str]]) -> str:
    lines = []
    for item in results:
        title = item.get("title", "").strip()
        url = item.get("url", "").strip()
        snippet = item.get("snippet", "").strip()
        if title or snippet:
            lines.append(f"- {title}\n  URL: {url}\n  Snippet: {snippet}")
    return "\n".join(lines)


def _extract_job_titles(soup: BeautifulSoup) -> List[str]:
    titles: List[str] = []
    selectors = [
        "h3.base-search-card__title",
        ".base-search-card__title",
        "[class*=job-title]",
        "h3",
    ]
    for selector in selectors:
        for node in soup.select(selector):
            text = clean_text(node.get_text(" ", strip=True))
            if 3 <= len(text) <= 120 and text not in titles:
                titles.append(text)
        if titles:
            break
    return titles


def _discover_feed_urls(homepage: str, html: str) -> List[str]:
    candidates: List[str] = []
    if html:
        soup = BeautifulSoup(html, "lxml")
        for link in soup.find_all("link", href=True):
            link_type = (link.get("type") or "").lower()
            rel = " ".join(link.get("rel") or []).lower()
            if "rss" in link_type or "atom" in link_type or "alternate" in rel:
                candidates.append(urljoin(homepage, link["href"]))

    common_paths = [
        "/feed",
        "/feed.xml",
        "/rss",
        "/rss.xml",
        "/blog/feed",
        "/blog/rss.xml",
        "/news/rss.xml",
        "/press/rss.xml",
    ]
    for path in common_paths:
        candidates.append(urljoin(homepage, path))

    seen = set()
    unique = []
    for url in candidates:
        if url not in seen:
            seen.add(url)
            unique.append(url)
    return unique


def _parse_feed_entries(feed_xml: str, limit: int = 8) -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    try:
        root = ET.fromstring(feed_xml.encode("utf-8"))
    except ET.ParseError:
        return entries

    for item in root.findall(".//item")[:limit]:
        entries.append({
            "title": _xml_text(item, "title"),
            "link": _xml_text(item, "link"),
            "date": _xml_text(item, "pubDate"),
            "summary": _xml_text(item, "description"),
        })

    if entries:
        return entries

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    for entry in root.findall(".//atom:entry", ns)[:limit]:
        link_node = entry.find("atom:link", ns)
        entries.append({
            "title": _xml_text(entry, "atom:title", ns),
            "link": link_node.get("href", "") if link_node is not None else "",
            "date": _xml_text(entry, "atom:updated", ns) or _xml_text(entry, "atom:published", ns),
            "summary": _xml_text(entry, "atom:summary", ns) or _xml_text(entry, "atom:content", ns),
        })
    return entries


def _xml_text(node: ET.Element, path: str, ns: Optional[Dict[str, str]] = None) -> str:
    child = node.find(path, ns or {})
    if child is None or child.text is None:
        return ""
    return clean_text(child.text)
