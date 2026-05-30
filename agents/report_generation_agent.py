# agents/report_generation_agent.py
# ============================================================
# Report Generation Agent: compiles the final BI report
# ============================================================

import re
from datetime import datetime
from typing import Dict, Any, List

from models.state import IntelligenceState, SWOTAnalysis
from tools.llm_client import get_llm_client
from rag.rag_pipeline import get_rag_pipeline
from utils.logger import logger


SYSTEM_PROMPT = """You are a senior business intelligence analyst and report writer.
You produce executive-quality intelligence reports that are clear, structured,
and immediately useful for marketing and sales professionals.
Your reports are evidence-based, concise, and highlight the most actionable insights."""


def report_generation_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: Report Generation Agent

    Responsibilities:
    1. Generate SWOT analysis
    2. Write executive summary
    3. Compile full markdown report
    """
    intel = IntelligenceState(**state)
    company = intel.input.company_name
    intel.agent_logs.append("[ReportAgent] Compiling final business intelligence report...")
    intel.status = "generating_report"

    llm = get_llm_client()

    # ── 1. SWOT Analysis ───────────────────────────────────────
    intel.swot_analysis = _generate_swot(intel, llm)
    intel.agent_logs.append("[ReportAgent] SWOT analysis generated.")

    # ── 2. Executive Summary ───────────────────────────────────
    intel.executive_summary = _generate_executive_summary(intel, llm)
    intel.agent_logs.append("[ReportAgent] Executive summary generated.")

    # ── 3. Compile Full Report ─────────────────────────────────
    intel.final_report = _compile_report(intel)
    intel.agent_logs.append("[ReportAgent] Full report compiled.")
    intel.status = "complete"

    return intel.model_dump()


def _generate_swot(intel: IntelligenceState, llm) -> SWOTAnalysis:
    company = intel.input.company_name

    swot_prompt = f"""Generate a SWOT analysis for {company} based on the following intelligence:

Company Overview: {intel.company_overview[:500]}

Product Positioning: {intel.product_positioning[:400]}

Market Differentiation: {intel.market_differentiation[:400]}

Competitors: {', '.join(intel.identified_competitors)}

Lead Signals: {_format_signals(intel.lead_signals)}

Return ONLY a JSON object:
{{
  "strengths": ["strength1", "strength2", "strength3", "strength4"],
  "weaknesses": ["weakness1", "weakness2", "weakness3"],
  "opportunities": ["opportunity1", "opportunity2", "opportunity3"],
  "threats": ["threat1", "threat2", "threat3"]
}}

Make each item specific and substantive, not generic."""

    raw = llm.complete(prompt=swot_prompt, system_prompt=SYSTEM_PROMPT, max_tokens=600)

    import json
    data = {}
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
        except:
            pass

    return SWOTAnalysis(
        strengths=data.get("strengths", ["Strong product market fit", "Innovative technology"]),
        weaknesses=data.get("weaknesses", ["Limited public pricing transparency", "Competitive market"]),
        opportunities=data.get("opportunities", ["Market expansion", "Partnership opportunities"]),
        threats=data.get("threats", ["Intense competition", "Rapid market changes"]),
    )


def _generate_executive_summary(intel: IntelligenceState, llm) -> str:
    company = intel.input.company_name
    lead_score = intel.lead_score.total_score if intel.lead_score else 0
    signal_count = len(intel.lead_signals)

    prompt = f"""Write a concise executive summary for the business intelligence report on {company}.

Data available:
- Company Overview: {intel.company_overview[:400]}
- Products: {intel.products_services[:300]}
- Positioning: {intel.product_positioning[:300]}
- Competitors identified: {', '.join(intel.identified_competitors[:5])}
- Lead Score: {lead_score}/100
- Signals detected: {signal_count} ({_signal_type_breakdown(intel.lead_signals)})
- Key outreach angle: {intel.outreach_recommendations[:200]}

Write a 2-3 paragraph executive summary that covers:
1. Who the company is and what they do
2. Their market position and key differentiators
3. Lead intelligence summary and recommended action

Make it boardroom-ready: clear, confident, actionable."""

    return llm.complete(prompt=prompt, system_prompt=SYSTEM_PROMPT, max_tokens=500)


def _compile_report(intel: IntelligenceState) -> str:
    """Build the complete markdown report."""
    company = intel.input.company_name
    now = datetime.now().strftime("%B %d, %Y")
    lead_score = intel.lead_score
    swot = intel.swot_analysis

    # Format lead score bar
    score_val = lead_score.total_score if lead_score else 0
    score_label = _score_label(score_val)
    score_bar = _score_bar(score_val)

    # Format competitors table
    comp_table = intel.competitor_table or _fallback_comp_table(intel.identified_competitors)

    # Format SWOT
    swot_md = _format_swot_md(swot)

    # Format signals
    signals_md = _format_signals_md(intel.lead_signals)

    # Format sources
    sources_md = _format_sources_md(intel.sources)

    report = f"""# Business Intelligence Report: {company}
**Generated:** {now}  
**Product Context:** {intel.input.product_name or intel.input.product_scenario or 'General Market Intelligence'}

---

## 📋 Executive Summary

{intel.executive_summary}

---

## 🏢 Company Overview

{intel.company_overview}

---

## 🛍️ Products & Services

{intel.products_services}

---

## 🎯 Competitor Analysis

### Identified Competitors
{', '.join(intel.identified_competitors) if intel.identified_competitors else 'No competitors identified'}

### Comparison Table

{comp_table}

### Detailed Competitor Profiles

{_format_competitor_profiles(intel.competitor_analyses)}

---

## 📢 Product Positioning Analysis

{intel.product_positioning}

---

## 💬 Market Messaging Analysis

{intel.market_messaging}

### Value Propositions
{intel.value_propositions}

### Pricing Insights
{intel.pricing_insights}

### Market Differentiation
{intel.market_differentiation}

---

## 📰 Recent Activities

{intel.recent_activities}

---

## 🔍 Lead Intelligence

### Lead Score: {score_val}/100 — {score_label}

```
{score_bar}
```

**Score Breakdown:**
| Category | Score | Max |
|----------|-------|-----|
| 🏗️ Hiring Activity | {lead_score.hiring_score if lead_score else 0} | 25 |
| 💰 Funding Signals | {lead_score.funding_score if lead_score else 0} | 30 |
| 🚀 Product Launches | {lead_score.launch_score if lead_score else 0} | 20 |
| 🌍 Expansion Signals | {lead_score.expansion_score if lead_score else 0} | 15 |
| 🤝 Partnerships | {lead_score.partnership_score if lead_score else 0} | 10 |
| **TOTAL** | **{score_val}** | **100** |

**Justification:** {lead_score.justification if lead_score else 'No score data'}

### Detected Signals

{signals_md}

---

## 📧 Outreach Recommendations

{intel.outreach_recommendations}

---

## ⚡ SWOT Analysis

{swot_md}

---

## 🔗 Sources & References

{sources_md}

---

*Report generated by AI-Powered Business & Market Intelligence Agent*  
*Data sourced from publicly available information only*  
*{now}*
"""
    return report


def _format_competitor_profiles(analyses) -> str:
    if not analyses:
        return "*No detailed competitor profiles available.*"
    parts = []
    for c in analyses:
        strengths = "\n".join([f"  - {s}" for s in c.strengths]) if c.strengths else "  - N/A"
        weaknesses = "\n".join([f"  - {w}" for w in c.weaknesses]) if c.weaknesses else "  - N/A"
        parts.append(f"""#### {c.name}
**Website:** {c.website}  
**Description:** {c.description}  
**Positioning:** {c.positioning}  
**Value Proposition:** {c.value_proposition}  

**Strengths:**
{strengths}

**Weaknesses:**
{weaknesses}

**Messaging Comparison:** {c.messaging_comparison}
""")
    return "\n---\n".join(parts)


def _format_swot_md(swot: SWOTAnalysis) -> str:
    if not swot:
        return "*SWOT analysis not available.*"

    def fmt(items: List[str]) -> str:
        return "\n".join([f"- {item}" for item in items]) if items else "- N/A"

    return f"""| | |
|--|--|
| **💪 Strengths** | **🔴 Weaknesses** |
| {" · ".join(swot.strengths)} | {" · ".join(swot.weaknesses)} |
| **🌟 Opportunities** | **⚠️ Threats** |
| {" · ".join(swot.opportunities)} | {" · ".join(swot.threats)} |

**Strengths:**
{fmt(swot.strengths)}

**Weaknesses:**
{fmt(swot.weaknesses)}

**Opportunities:**
{fmt(swot.opportunities)}

**Threats:**
{fmt(swot.threats)}"""


def _format_signals_md(signals) -> str:
    if not signals:
        return "*No specific signals detected.*"

    by_type: Dict[str, List] = {}
    for s in signals:
        by_type.setdefault(s.signal_type, []).append(s)

    parts = []
    type_icons = {
        "hiring": "👥",
        "funding": "💰",
        "launch": "🚀",
        "expansion": "🌍",
        "partnership": "🤝",
    }
    for stype, slist in by_type.items():
        icon = type_icons.get(stype, "📌")
        parts.append(f"#### {icon} {stype.capitalize()} Signals")
        for s in slist:
            date_str = f" ({s.date_mentioned})" if s.date_mentioned else ""
            source_str = f" — *{s.source}*" if s.source else ""
            confidence_str = f" `{s.confidence}`"
            parts.append(f"- {s.description}{date_str}{source_str}{confidence_str}")
        parts.append("")

    return "\n".join(parts)


def _format_sources_md(sources: List[str]) -> str:
    if not sources:
        return "*No sources tracked.*"
    lines = []
    for i, url in enumerate(list(dict.fromkeys(sources))[:20], 1):
        lines.append(f"{i}. {url}")
    return "\n".join(lines)


def _fallback_comp_table(competitors: List[str]) -> str:
    if not competitors:
        return "*No competitor data available.*"
    rows = ["| Company | Status |", "|---------|--------|"]
    for c in competitors:
        rows.append(f"| {c} | Identified |")
    return "\n".join(rows)


def _score_label(score: int) -> str:
    if score >= 80:
        return "🔥 HOT LEAD"
    elif score >= 60:
        return "✅ WARM LEAD"
    elif score >= 40:
        return "🔵 MODERATE"
    elif score >= 20:
        return "🟡 COOL"
    else:
        return "⚪ LOW SIGNAL"


def _score_bar(score: int) -> str:
    filled = int(score / 5)
    empty = 20 - filled
    return f"[{'█' * filled}{'░' * empty}] {score}/100"


def _format_signals(signals) -> str:
    if not signals:
        return "None detected"
    return "; ".join([f"{s.signal_type}: {s.description[:60]}" for s in signals[:5]])


def _signal_type_breakdown(signals) -> str:
    from collections import Counter
    if not signals:
        return "none"
    counts = Counter(s.signal_type for s in signals)
    return ", ".join([f"{k}: {v}" for k, v in counts.items()])


from typing import List, Dict
