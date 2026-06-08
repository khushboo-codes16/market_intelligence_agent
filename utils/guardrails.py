# Add to utils/guardrails.py
import re

def detect_pii(text: str) -> bool:
    """Basic PII detection - credit cards, emails, phones"""
    patterns = [
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        r'\b\d{4}[-]?\d{4}[-]?\d{4}[-]?\d{4}\b',  # Credit card
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone
    ]
    for pattern in patterns:
        if re.search(pattern, text):
            return True
    return False

# In your LLM client:
if detect_pii(llm_response):
    return {"error": "PII detected - response blocked"}