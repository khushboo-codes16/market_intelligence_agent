# tools/scraper.py
# ============================================================
# Web scraping and content extraction
# ============================================================

import time
import re
from typing import Optional, List, Dict
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
import trafilatura
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config.settings import REQUEST_TIMEOUT, REQUEST_DELAY, MAX_PAGES_PER_DOMAIN, USER_AGENT
from utils.logger import logger
from utils.text_utils import clean_text


HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
)
def fetch_url(url: str) -> Optional[str]:
    """Fetch raw HTML from a URL with retries."""
    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
        )
        response.raise_for_status()
        return response.text
    except requests.HTTPError as e:
        logger.warning(f"HTTP error {e.response.status_code} for {url}")
        return None
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return None


def extract_text_trafilatura(html: str, url: str = "") -> str:
    """Use trafilatura for clean main-content extraction."""
    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        no_fallback=False,
        favor_recall=True,
    )
    return clean_text(text or "")


def extract_text_bs4(html: str) -> str:
    """Fallback extraction using BeautifulSoup."""
    soup = BeautifulSoup(html, "lxml")
    # Remove noise tags
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "iframe"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    return clean_text(text)


def scrape_page(url: str) -> Dict[str, str]:
    """
    Scrape a single URL and return structured data.
    Returns: {url, title, text, meta_description}
    """
    result = {"url": url, "title": "", "text": "", "meta_description": ""}
    logger.info(f"Scraping: {url}")

    html = fetch_url(url)
    if not html:
        return result

    # Parse metadata with BS4
    soup = BeautifulSoup(html, "lxml")
    title_tag = soup.find("title")
    result["title"] = title_tag.get_text(strip=True) if title_tag else ""

    meta_desc = soup.find("meta", attrs={"name": "description"})
    if meta_desc:
        result["meta_description"] = meta_desc.get("content", "")

    # Try trafilatura first, fallback to BS4
    text = extract_text_trafilatura(html, url)
    if len(text) < 200:
        text = extract_text_bs4(html)

    result["text"] = text
    time.sleep(REQUEST_DELAY)
    return result


def discover_internal_links(url: str, html: str, max_links: int = 10) -> List[str]:
    """Find relevant internal links (pricing, about, blog, products, careers)."""
    soup = BeautifulSoup(html, "lxml")
    base = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
    priority_patterns = [
        r"pricing", r"product", r"feature", r"about", r"blog",
        r"news", r"press", r"career", r"job", r"solution", r"platform",
    ]
    found: List[str] = []
    seen = {url}

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"].strip()
        full_url = urljoin(base, href)
        parsed = urlparse(full_url)

        # Same domain only
        if parsed.netloc != urlparse(url).netloc:
            continue
        if full_url in seen:
            continue
        if parsed.fragment:
            continue

        path = parsed.path.lower()
        if any(re.search(pat, path) for pat in priority_patterns):
            found.append(full_url)
            seen.add(full_url)

    return found[:max_links]


def scrape_company_website(base_url: str) -> List[Dict[str, str]]:
    """
    Scrape a company website: homepage + up to MAX_PAGES_PER_DOMAIN internal pages.
    Returns list of page data dicts.
    """
    if not base_url.startswith("http"):
        base_url = "https://" + base_url

    pages: List[Dict[str, str]] = []
    homepage_html = fetch_url(base_url)
    if not homepage_html:
        logger.warning(f"Could not fetch homepage: {base_url}")
        return pages

    home_data = scrape_page(base_url)
    pages.append(home_data)

    # Discover and scrape sub-pages
    sub_links = discover_internal_links(base_url, homepage_html, max_links=MAX_PAGES_PER_DOMAIN * 2)
    scraped = 0
    for link in sub_links:
        if scraped >= MAX_PAGES_PER_DOMAIN - 1:
            break
        page_data = scrape_page(link)
        if page_data["text"] and len(page_data["text"]) > 100:
            pages.append(page_data)
            scraped += 1

    logger.info(f"Scraped {len(pages)} pages from {base_url}")
    return pages


def search_web_news(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """
    Search for news via DuckDuckGo HTML (no API key required).
    Returns list of {title, url, snippet}.
    """
    results: List[Dict[str, str]] = []
    search_url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"

    try:
        response = requests.get(search_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        soup = BeautifulSoup(response.text, "lxml")

        for result in soup.select(".result")[:num_results]:
            title_el = result.select_one(".result__title")
            snippet_el = result.select_one(".result__snippet")
            link_el = result.select_one(".result__url")

            if title_el and link_el:
                title = title_el.get_text(strip=True)
                snippet = snippet_el.get_text(strip=True) if snippet_el else ""
                href = title_el.find("a")
                url = href["href"] if href and href.get("href") else link_el.get_text(strip=True)

                # DuckDuckGo redirect URLs
                if url.startswith("//duckduckgo.com/l/?"):
                    match = re.search(r"uddg=([^&]+)", url)
                    if match:
                        url = requests.utils.unquote(match.group(1))

                results.append({"title": title, "url": url, "snippet": snippet})
    except Exception as e:
        logger.warning(f"News search failed for '{query}': {e}")

    time.sleep(REQUEST_DELAY)
    return results


def scrape_news_articles(company_name: str, topics: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """
    Search for and scrape news articles about a company.
    Returns list of scraped article data.
    """
    if topics is None:
        topics = [
            f"{company_name} funding announcement",
            f"{company_name} product launch",
            f"{company_name} hiring expansion",
            f"{company_name} partnership",
            f"{company_name} news 2024 2025",
        ]

    articles: List[Dict[str, str]] = []
    seen_urls = set()

    for topic in topics[:4]:  # Limit queries
        search_results = search_web_news(topic, num_results=3)
        for sr in search_results:
            url = sr.get("url", "")
            if not url or url in seen_urls:
                continue
            if not url.startswith("http"):
                continue
            seen_urls.add(url)

            page = scrape_page(url)
            if page["text"] and len(page["text"]) > 150:
                page["search_snippet"] = sr.get("snippet", "")
                page["search_title"] = sr.get("title", "")
                articles.append(page)

    logger.info(f"Collected {len(articles)} news articles for {company_name}")
    return articles
