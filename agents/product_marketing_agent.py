# agents/product_marketing_agent.py
# ============================================================
# Product Marketing Agent: positioning, messaging, pricing
# ============================================================

import re
from typing import Dict, Any

from models.state import IntelligenceState
from tools.llm_client import get_llm_client
from rag.rag_pipeline import get_rag_pipeline
from utils.logger import logger


SYSTEM_PROMPT = """You are a senior product marketing analyst specializing in B2B SaaS and AI companies.
You analyze how companies position and message their products, identify value propositions,
differentiation strategies, and market messaging. Your analyses are concise, data-driven,
and grounded in evidence from actual company content.
Always cite specific examples from the provided context."""


def product_marketing_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: Product Marketing Agent

    Analyzes:
    - Product positioning
    - Value propositions
    - Market messaging
    - Pricing insights
    - Market differentiation
    - Products and services overview
    """
    intel = IntelligenceState(**state)
    company = intel.input.company_name
    product = intel.input.product_name or company
    scenario = intel.input.product_scenario or ""
    intel.agent_logs.append("[ProductMarketingAgent] Starting marketing analysis...")
    intel.status = "analyzing_marketing"

    llm = get_llm_client()
    rag = get_rag_pipeline(collection_name=f"intel_{_sanitize(company)}")

    # ── Company Overview ───────────────────────────────────────
    context = rag.retrieve_as_context(
        f"{company} company overview products services mission", top_k=5
    )
    intel.company_overview = llm.complete(
        prompt=_build_prompt(
            task="company_overview",
            company=company,
            context=context,
            scenario=scenario,
        ),
        system_prompt=SYSTEM_PROMPT,
        max_tokens=800,
    )
    intel.agent_logs.append("[ProductMarketingAgent] Company overview generated.")

    # ── Products & Services ────────────────────────────────────
    context = rag.retrieve_as_context(
        f"{product} features capabilities offerings pricing", top_k=5
    )
    intel.products_services = llm.complete(
        prompt=_build_prompt("products_services", company, context, product=product),
        system_prompt=SYSTEM_PROMPT,
        max_tokens=800,
    )

    # ── Product Positioning ────────────────────────────────────
    context = rag.retrieve_as_context(
        f"{company} positioning target audience use cases differentiation", top_k=5
    )
    intel.product_positioning = llm.complete(
        prompt=_build_prompt("positioning", company, context, product=product, scenario=scenario),
        system_prompt=SYSTEM_PROMPT,
        max_tokens=700,
    )

    # ── Value Propositions ─────────────────────────────────────
    context = rag.retrieve_as_context(
        f"{company} value proposition benefits ROI why choose", top_k=4
    )
    intel.value_propositions = llm.complete(
        prompt=_build_prompt("value_propositions", company, context, product=product),
        system_prompt=SYSTEM_PROMPT,
        max_tokens=600,
    )

    # ── Market Messaging ───────────────────────────────────────
    context = rag.retrieve_as_context(
        f"{company} tagline headline messaging brand voice", top_k=4
    )
    intel.market_messaging = llm.complete(
        prompt=_build_prompt("messaging", company, context, product=product, scenario=scenario),
        system_prompt=SYSTEM_PROMPT,
        max_tokens=600,
    )

    # ── Pricing Insights ───────────────────────────────────────
    context = rag.retrieve_as_context(
        f"{company} pricing plans tiers cost enterprise free trial", top_k=4
    )
    intel.pricing_insights = llm.complete(
        prompt=_build_prompt("pricing", company, context),
        system_prompt=SYSTEM_PROMPT,
        max_tokens=500,
    )

    # ── Market Differentiation ─────────────────────────────────
    context = rag.retrieve_as_context(
        f"{company} competitive advantage unique different compared to alternatives", top_k=4
    )
    intel.market_differentiation = llm.complete(
        prompt=_build_prompt("differentiation", company, context, product=product),
        system_prompt=SYSTEM_PROMPT,
        max_tokens=500,
    )

    intel.agent_logs.append("[ProductMarketingAgent] All marketing analyses complete.")
    intel.status = "marketing_complete"
    return intel.model_dump()


def _build_prompt(
    task: str,
    company: str,
    context: str,
    product: str = "",
    scenario: str = "",
) -> str:
    product_str = f" (product: {product})" if product and product != company else ""
    scenario_str = f"\nScenario context: {scenario}" if scenario else ""

    prompts = {
        "company_overview": f"""Analyze the following content about {company}{product_str} and write a concise company overview.
{scenario_str}

Include:
- What the company does
- Core mission / vision
- Primary customer segments
- Company stage and scale (if inferable)

CONTEXT:
{context}

Write a clear, factual 3-5 paragraph overview. Base it on the context provided.""",

        "products_services": f"""Based on the following content about {company}{product_str}, describe their products and services.
{scenario_str}

Include:
- Core product(s) and platform
- Key features and capabilities
- Target use cases
- Any notable integrations or ecosystem

CONTEXT:
{context}

Be specific and concrete. Use bullet points where appropriate.""",

        "positioning": f"""Analyze the product positioning of {company}{product_str}.
{scenario_str}

Cover:
- Who they position themselves for (ICP)
- How they position against alternatives
- Key positioning themes (e.g., ease of use, power, enterprise, developer-first)
- Taglines or positioning statements found

CONTEXT:
{context}""",

        "value_propositions": f"""Extract and analyze the core value propositions of {company}{product_str}.
{scenario_str}

List:
- Primary value proposition (the #1 promise)
- 3-5 supporting value props
- Proof points or evidence cited

CONTEXT:
{context}""",

        "messaging": f"""Analyze the market messaging strategy of {company}{product_str}.
{scenario_str}

Cover:
- Brand voice and tone
- Key messaging pillars
- Emotional vs rational appeal
- Call-to-actions used
- Notable headlines or taglines

CONTEXT:
{context}""",

        "pricing": f"""Based on available information, describe {company}'s pricing model.

Cover:
- Pricing tiers (if visible)
- Free tier / trial availability
- Enterprise / custom pricing approach
- Pricing strategy (value-based, usage-based, per-seat, etc.)
- Any notable pricing signals

If no direct pricing is available, infer from the context.

CONTEXT:
{context}""",

        "differentiation": f"""Identify how {company}{product_str} differentiates itself in the market.
{scenario_str}

Cover:
- Unique capabilities vs. generic alternatives
- Technology or IP advantages
- Go-to-market differentiation
- Brand differentiation
- Moats (data, network effects, integrations)

CONTEXT:
{context}""",
    }
    return prompts.get(task, f"Analyze {company} based on this context:\n\n{context}")


def _sanitize(name: str) -> str:
    import re
    return re.sub(r"[^a-z0-9_]", "_", name.lower().strip())[:40]
