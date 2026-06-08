# guardrails/input_guard.py
# ============================================================
# GUARDRAIL 1 — Input Validation
# Validates user input BEFORE the pipeline starts.
# Catches: empty inputs, junk, injections, too-long strings.
# ============================================================

import re
from dataclasses import dataclass
from typing import Tuple


# ── Blocked patterns ──────────────────────────────────────────
INJECTION_PATTERNS = [
    r"<script.*?>",          # XSS
    r"javascript:",          # JS injection
    r"ignore previous",      # prompt injection
    r"ignore all instructions",
    r"you are now",          # jailbreak
    r"forget everything",
    r"system prompt",
    r"DROP TABLE",           # SQL injection
    r"\.\./",                # path traversal
]

# Known test/garbage inputs
JUNK_INPUTS = {
    "test", "asdf", "qwerty", "hello", "abc",
    "123", "xxx", "none", "null", "undefined",
    "company", "enter company", "your company",
}

# Max length limits
MAX_COMPANY_LEN  = 100
MAX_WEBSITE_LEN  = 200
MAX_SCENARIO_LEN = 500


@dataclass
class ValidationResult:
    is_valid:     bool
    error:        str  = ""
    cleaned_name: str  = ""
    cleaned_url:  str  = ""


def validate_inputs(
    company_name:     str,
    company_website:  str = "",
    product_name:     str = "",
    product_scenario: str = "",
) -> ValidationResult:
    """
    Validate all user inputs before the pipeline starts.
    Returns ValidationResult with is_valid=False and an error message
    if anything is wrong.
    """

    # ── 1. Company name required ──────────────────────────────
    if not company_name or not company_name.strip():
        return ValidationResult(
            is_valid=False,
            error="Company name is required. Please enter a company name."
        )

    name = company_name.strip()

    # ── 2. Length check ───────────────────────────────────────
    if len(name) > MAX_COMPANY_LEN:
        return ValidationResult(
            is_valid=False,
            error=f"Company name too long ({len(name)} chars). Max {MAX_COMPANY_LEN}."
        )

    # ── 3. Junk / meaningless input ───────────────────────────
    if name.lower() in JUNK_INPUTS:
        return ValidationResult(
            is_valid=False,
            error=f"'{name}' doesn't look like a real company name. "
                  "Please enter a valid company (e.g. OpenAI, Salesforce)."
        )

    # ── 4. Injection / malicious patterns ─────────────────────
    all_text = f"{name} {company_website} {product_name} {product_scenario}"
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, all_text, re.IGNORECASE):
            return ValidationResult(
                is_valid=False,
                error="Input contains invalid characters or patterns. "
                      "Please enter a real company name."
            )

    # ── 5. Company name format check ─────────────────────────
    # Allow letters, numbers, spaces, dots, hyphens, ampersands
    if not re.match(r"^[a-zA-Z0-9\s\.\-&',()]+$", name):
        return ValidationResult(
            is_valid=False,
            error="Company name contains invalid characters. "
                  "Only letters, numbers, spaces, and basic punctuation allowed."
        )

    # ── 6. Must have at least 2 characters ───────────────────
    if len(name.replace(" ", "")) < 2:
        return ValidationResult(
            is_valid=False,
            error="Company name too short. Please enter at least 2 characters."
        )

    # ── 7. Website URL validation (optional field) ────────────
    clean_url = ""
    if company_website and company_website.strip():
        url = company_website.strip()
        if len(url) > MAX_WEBSITE_LEN:
            return ValidationResult(
                is_valid=False,
                error=f"Website URL too long. Max {MAX_WEBSITE_LEN} characters."
            )
        # Must start with http or be a domain
        if not re.match(r"^(https?://)?[a-zA-Z0-9][\w\-\.]+\.[a-z]{2,}", url):
            return ValidationResult(
                is_valid=False,
                error=f"'{url}' doesn't look like a valid website URL. "
                      "Example: https://openai.com"
            )
        # Ensure https prefix
        if not url.startswith("http"):
            url = "https://" + url
        clean_url = url

    # ── 8. Scenario length check ──────────────────────────────
    if product_scenario and len(product_scenario) > MAX_SCENARIO_LEN:
        return ValidationResult(
            is_valid=False,
            error=f"Scenario too long ({len(product_scenario)} chars). "
                  f"Max {MAX_SCENARIO_LEN} characters."
        )

    # ── All good ──────────────────────────────────────────────
    return ValidationResult(
        is_valid=True,
        cleaned_name=name,
        cleaned_url=clean_url,
    )