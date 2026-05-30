# utils/text_utils.py
# ============================================================
# Text cleaning, chunking, and normalization helpers
# ============================================================

import re
from typing import List


def clean_text(text: str) -> str:
    """Remove noise, normalise whitespace, strip boilerplate."""
    if not text:
        return ""
    # Remove null bytes
    text = text.replace("\x00", "")
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Remove non-printable characters except newlines
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", "", text)
    # Strip leading / trailing whitespace per line
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """
    Split text into overlapping chunks by character count,
    attempting to break on paragraph / sentence boundaries.
    """
    if not text:
        return []

    # Prefer paragraph breaks
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]

    chunks: List[str] = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= chunk_size:
            current_chunk = (current_chunk + "\n\n" + para).strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
                # Start next chunk with overlap
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = (overlap_text + "\n\n" + para).strip()
            else:
                # Paragraph itself larger than chunk_size — split by sentence
                sentences = re.split(r"(?<=[.!?])\s+", para)
                for sent in sentences:
                    if len(current_chunk) + len(sent) + 1 <= chunk_size:
                        current_chunk = (current_chunk + " " + sent).strip()
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                            current_chunk = (overlap_text + " " + sent).strip()
                        else:
                            chunks.append(sent[:chunk_size])
                            current_chunk = sent[chunk_size - overlap:]

    if current_chunk:
        chunks.append(current_chunk)

    return [c for c in chunks if len(c) > 50]


def extract_domain(url: str) -> str:
    """Return bare domain from a URL."""
    url = re.sub(r"https?://", "", url)
    url = url.split("/")[0].split("?")[0]
    return url.lower().strip()


def truncate_text(text: str, max_chars: int = 2000) -> str:
    """Truncate text to max_chars, preserving word boundaries."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > max_chars * 0.8:
        truncated = truncated[:last_space]
    return truncated + "..."


def normalize_company_name(name: str) -> str:
    """Lowercase and strip punctuation for consistent matching."""
    return re.sub(r"[^a-z0-9\s]", "", name.lower()).strip()
