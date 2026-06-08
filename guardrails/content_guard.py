# guardrails/content_guard.py
# ============================================================
# GUARDRAIL 4 — Content Safety
# Filters scraped web content before storing in RAG.
# Catches: very short pages, error pages, adult content,
#          cookie banners, login walls, captcha pages.
# ============================================================

import re
from typing import Dict, Any, List


# ── Signals that a page is NOT useful content ─────────────────
ERROR_PAGE_SIGNALS = [
    "404 not found",
    "403 forbidden",
    "page not found",
    "access denied",
    "this page doesn't exist",
    "404 error",
]

LOGIN_WALL_SIGNALS = [
    "sign in to continue",
    "log in to view",
    "create an account to",
    "please log in",
    "you must be signed in",
    "subscription required",
]

COOKIE_BANNER_SIGNALS = [
    "we use cookies",
    "accept all cookies",
    "cookie policy",
    "gdpr",
    "privacy preferences",
]

CAPTCHA_SIGNALS = [
    "verify you are human",
    "complete the captcha",
    "robot check",
    "are you a robot",
    "cloudflare",
    "please enable javascript",
]

# Minimum useful content length
MIN_CONTENT_LENGTH = 150


def is_useful_page(page_data: Dict[str, Any]) -> tuple:
    """
    Check if a scraped page contains useful content.
    Returns (is_useful, reason_if_not).
    """
    text  = (page_data.get("text") or "").strip()
    title = (page_data.get("title") or "").lower()
    url   = (page_data.get("url") or "")

    # Empty content
    if not text:
        return False, "Empty page content"

    # Too short to be useful
    if len(text) < MIN_CONTENT_LENGTH:
        return False, f"Page too short ({len(text)} chars)"

    lower_text = text.lower()

    # Error pages
    for signal in ERROR_PAGE_SIGNALS:
        if signal in lower_text or signal in title:
            return False, f"Error page detected: '{signal}'"

    # Login walls
    for signal in LOGIN_WALL_SIGNALS:
        if signal in lower_text:
            return False, f"Login wall detected: '{signal}'"

    # Captcha / bot detection
    for signal in CAPTCHA_SIGNALS:
        if signal in lower_text:
            return False, f"Bot detection page: '{signal}'"

    # Page that is ONLY cookie consent (very short after removing banner)
    cookie_count = sum(1 for s in COOKIE_BANNER_SIGNALS if s in lower_text)
    if cookie_count >= 3 and len(text) < 500:
        return False, "Cookie consent page only"

    return True, ""


def filter_documents(
    documents: List[Dict[str, Any]],
) -> tuple:
    """
    Filter a list of scraped documents.
    Returns (useful_docs, filtered_count, filter_reasons).
    """
    useful   = []
    filtered = []

    for doc in documents:
        is_useful, reason = is_useful_page(doc)
        if is_useful:
            useful.append(doc)
        else:
            filtered.append({
                "url":    doc.get("url", ""),
                "reason": reason,
            })

    return useful, len(filtered), filtered


def sanitize_text(text: str) -> str:
    """
    Clean scraped text before storing in RAG.
    Removes: excessive whitespace, null bytes, HTML remnants,
             cookie consent text, navigation boilerplate.
    """
    if not text:
        return ""

    # Remove null bytes
    text = text.replace("\x00", "")

    # Remove HTML entities
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"&#\d+;", " ", text)

    # Remove URLs (they add noise to embeddings)
    text = re.sub(r"https?://\S+", "[URL]", text)

    # Remove email addresses
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "[EMAIL]", text)

    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Remove very short lines (navigation items, menu entries)
    lines = text.splitlines()
    lines = [l.strip() for l in lines if len(l.strip()) > 20 or l.strip() == ""]
    text  = "\n".join(lines)

    return text.strip()


def rate_content_quality(text: str) -> str:
    """
    Rate the quality of scraped content.
    Returns: "high" | "medium" | "low"
    """
    if not text:
        return "low"
    length = len(text)
    words  = len(text.split())

    if length > 2000 and words > 300:
        return "high"
    if length > 500 and words > 80:
        return "medium"
    return "low"