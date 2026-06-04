# ui/dashboard.py
# ============================================================
# Streamlit Dashboard for Market Intelligence Agent
# ============================================================

import sys
import os
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure parent directory is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False

from workflow.graph import run_intelligence_pipeline
from reports.report_saver import save_report, generate_pdf_bytes, load_run_history
from tools.llm_client import clear_llm_cache, get_llm_client
from models.state import IntelligenceState
from utils.confidence import confidence_html


# ── Page Configuration ────────────────────────────────────────

st.set_page_config(
    page_title="AI Market Intelligence Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .metric-card {
        background: #f8f9fa;
        border-left: 4px solid #0f3460;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .score-high { color: #ff4444; font-weight: bold; font-size: 1.5rem; }
    .score-medium { color: #ff9800; font-weight: bold; font-size: 1.5rem; }
    .score-low { color: #4caf50; font-weight: bold; font-size: 1.5rem; }
    .signal-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.8rem;
        margin: 0.2rem;
        font-weight: 500;
    }
    .agent-log {
        background: #1e1e1e;
        color: #00ff00;
        font-family: monospace;
        font-size: 0.8rem;
        padding: 1rem;
        border-radius: 8px;
        max-height: 200px;
        overflow-y: auto;
    }
    .stProgress > div > div { background-color: #0f3460; }
    div[data-testid="metric-container"] {
        background-color: #f0f4ff;
        border: 1px solid #dce4f5;
        border-radius: 8px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────

def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>🔍 AI Market Intelligence Agent</h1>
        <p style="font-size: 1.1rem; opacity: 0.85; margin: 0;">
            Automated Business & Competitive Intelligence powered by LangGraph + Groq + RAG
        </p>
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────

def render_sidebar() -> Dict[str, str]:
    with st.sidebar:
        st.image("https://img.shields.io/badge/Powered%20by-LangGraph-blue?style=for-the-badge", width=200)
        st.markdown("---")
        st.markdown("### 🎯 Target Configuration")

        company_name = st.text_input(
            "Company Name *",
            placeholder="e.g. OpenAI, Notion, HubSpot",
            help="The company you want to research",
        )

        company_website = st.text_input(
            "Company Website (optional)",
            placeholder="e.g. https://openai.com",
            help="Provide the URL for more accurate scraping",
        )

        st.markdown("---")
        st.markdown("### 🛍️ Product Context (optional)")

        product_name = st.text_input(
            "Product Name",
            placeholder="e.g. GPT-4, Notion AI",
            help="Specific product to focus the analysis on",
        )

        product_scenario = st.text_area(
            "Product Scenario",
            placeholder="e.g. A company launching a voice-based AI agent for lead generation",
            help="Describe the scenario for Track 1 or Track 2 analysis",
            height=100,
        )

        st.markdown("---")
        st.markdown("### 🔧 Quick Examples")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("OpenAI", use_container_width=True):
                st.session_state["prefill"] = {
                    "company": "OpenAI",
                    "website": "https://openai.com",
                    "product": "ChatGPT",
                    "scenario": "",
                }
                st.rerun()
            if st.button("Anthropic", use_container_width=True):
                st.session_state["prefill"] = {
                    "company": "Anthropic",
                    "website": "https://anthropic.com",
                    "product": "Claude",
                    "scenario": "",
                }
                st.rerun()
        with col2:
            if st.button("ElevenLabs", use_container_width=True):
                st.session_state["prefill"] = {
                    "company": "ElevenLabs",
                    "website": "https://elevenlabs.io",
                    "product": "Voice AI",
                    "scenario": "AI voice agent for customer calls",
                }
                st.rerun()
            if st.button("Notion", use_container_width=True):
                st.session_state["prefill"] = {
                    "company": "Notion",
                    "website": "https://notion.so",
                    "product": "Notion AI",
                    "scenario": "",
                }
                st.rerun()

        # Apply prefill
        if "prefill" in st.session_state:
            p = st.session_state["prefill"]
            company_name = p.get("company", company_name)
            company_website = p.get("website", company_website)
            product_name = p.get("product", product_name)
            product_scenario = p.get("scenario", product_scenario)
            del st.session_state["prefill"]

        # ── Run History Panel ─────────────────────────────────
        st.markdown("---")
        st.markdown("### 📂 Run History")
        history = load_run_history()
        if history:
            history_options = {
                f"{h['company']} — {h['datetime']} (Score: {h['lead_score']})": h
                for h in history[:10]
            }
            labels = list(history_options.keys())

            selected_label = st.selectbox(
                "Load a past run",
                [""] + labels,
                label_visibility="collapsed",
                key="history_select_load",
            )
            if selected_label and selected_label in history_options:
                entry = history_options[selected_label]
                md_path = entry.get("filepath", "")
                if md_path and Path(md_path).exists():
                    if st.button("📂 Load This Run", use_container_width=True):
                        report_text = Path(md_path).read_text(encoding="utf-8")
                        st.session_state["result_state"] = {
                            "input": {"company_name": entry["company"]},
                            "final_report": report_text,
                            "lead_score": {"total_score": entry["lead_score"]},
                            "identified_competitors": entry.get("competitors", []),
                            "lead_signals": [],
                            "sources": [],
                            "agent_logs": ["[History] Loaded from saved run."],
                            "rag_chunks_count": entry.get("rag_chunks", 0),
                            "_from_history": True,
                        }
                        st.rerun()

            # ── Compare two past runs ──────────────────────────
            if len(labels) >= 2:
                st.markdown("**⚖️ Compare two runs:**")
                col_a, col_b = st.columns(2)
                with col_a:
                    cmp_a = st.selectbox("Company A", [""] + labels, key="cmp_a", label_visibility="collapsed")
                with col_b:
                    cmp_b = st.selectbox("Company B", [""] + labels, key="cmp_b", label_visibility="collapsed")

                if st.button("⚖️ Compare", use_container_width=True):
                    if cmp_a and cmp_b and cmp_a != cmp_b:
                        st.session_state["compare_a"] = history_options[cmp_a]
                        st.session_state["compare_b"] = history_options[cmp_b]
                        st.rerun()
                    else:
                        st.warning("Select two different runs to compare.")
        else:
            st.caption("No past runs yet.")

        # ── Cache Controls ────────────────────────────────────
        st.markdown("---")
        st.markdown("### ⚡ LLM Cache")
        st.caption("Cache saves API calls on re-runs of the same company.")
        if st.button("🗑️ Clear LLM Cache", use_container_width=True):
            count = clear_llm_cache()
            st.success(f"Cleared {count} cached responses.")

        st.markdown("---")
        st.markdown("""
        <small>
        📌 Uses public data sources only<br>
        🤖 LangGraph multi-agent pipeline<br>
        🧠 Groq Llama 3.3 70B LLM<br>
        📦 ChromaDB vector store<br>
        </small>
        """, unsafe_allow_html=True)

    return {
        "company_name": company_name,
        "company_website": company_website,
        "product_name": product_name,
        "product_scenario": product_scenario,
    }


# ── Progress Display ──────────────────────────────────────────

# Pipeline stages reflect the new parallel architecture
# Phase 1: Research (sequential) → Phase 2: 3 agents in parallel → Phase 3: Report
PIPELINE_STAGES = [
    (10,  "🔎 Research Agent",      "Scraping website & news, building RAG index..."),
    (35,  "⚡ Parallel Phase",       "Marketing · Competitor · Lead Gen running simultaneously..."),
    (55,  "📢 Marketing Agent",      "Analyzing positioning, messaging & value props..."),
    (70,  "⚔️  Competitor Agent",    "Identifying & profiling competitors..."),
    (85,  "🎯 Lead Gen Agent",       "Extracting signals & computing lead score..."),
    (95,  "📄 Report Agent",         "Compiling final intelligence report..."),
]


def run_with_progress(inputs: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """Run pipeline with live progress updates reflecting parallel execution."""

    progress_container = st.container()
    with progress_container:
        st.markdown("### ⚙️ Running Intelligence Pipeline")
        st.caption("⚡ Marketing, Competitor & Lead Gen agents run in **parallel** — ~90s total")

        progress_bar = st.progress(0)
        stage_placeholder = st.empty()

        result_container: Dict[str, Any] = {"result": None, "error": None}

        def run_pipeline():
            try:
                result = run_intelligence_pipeline(
                    company_name=inputs["company_name"],
                    company_website=inputs["company_website"],
                    product_name=inputs["product_name"],
                    product_scenario=inputs["product_scenario"],
                )
                result_container["result"] = result
            except Exception as e:
                result_container["error"] = str(e)

        thread = threading.Thread(target=run_pipeline)
        thread.start()

        # Animate through stages while pipeline runs in background
        stage_idx = 0
        tick = 0
        while thread.is_alive():
            progress, name, desc = PIPELINE_STAGES[stage_idx]
            progress_bar.progress(progress)
            stage_placeholder.markdown(f"**{name}** — {desc}")
            time.sleep(4)
            tick += 1
            # Advance stage roughly every 12s (3 ticks), but never past second-to-last
            if tick % 3 == 0 and stage_idx < len(PIPELINE_STAGES) - 2:
                stage_idx += 1

        thread.join()

        if result_container["error"]:
            st.error(f"Pipeline failed: {result_container['error']}")
            return None

        progress_bar.progress(100)
        stage_placeholder.markdown("**✅ Pipeline complete!**")
        time.sleep(0.5)
        progress_container.empty()

        return result_container["result"]


# ── Tab Renderers ─────────────────────────────────────────────

def render_overview_tab(state: Dict[str, Any]):
    col1, col2, col3, col4 = st.columns(4)
    lead_score = state.get("lead_score") or {}
    score_val = lead_score.get("total_score", 0) if lead_score else 0
    competitor_count = len(state.get("identified_competitors", []))
    signal_count = len(state.get("lead_signals", []))
    source_count = len(state.get("sources", []))

    col1.metric("🎯 Lead Score", f"{score_val}/100", delta=_score_delta(score_val))
    col2.metric("⚔️ Competitors", competitor_count)
    col3.metric("📡 Signals", signal_count)
    col4.metric("🔗 Sources", source_count)

    conf = state.get("data_confidence", {})

    st.markdown("---")
    st.markdown(
        f"### 📋 Executive Summary {confidence_html(conf.get('company_overview', 'low'))}",
        unsafe_allow_html=True,
    )
    st.markdown(state.get("executive_summary", "*Not available*"))

    st.markdown("---")
    st.markdown(
        f"### 🏢 Company Overview {confidence_html(conf.get('company_overview', 'low'))}",
        unsafe_allow_html=True,
    )
    st.markdown(state.get("company_overview", "*Not available*"))

    st.markdown("---")
    st.markdown(
        f"### 📰 Recent Activities {confidence_html(conf.get('recent_activities', 'low'))}",
        unsafe_allow_html=True,
    )
    st.markdown(state.get("recent_activities", "*Not available*"))

    report_text = state.get("final_report", "")
    if report_text:
        st.markdown("---")
        st.markdown("### ⚡ Streaming Executive Briefing")
        st.caption("Generates a short live-streamed briefing from the completed report.")
        if st.button("▶️ Stream Briefing", use_container_width=True, key="stream_briefing_btn"):
            _stream_executive_briefing(report_text)


def render_competitors_tab(state: Dict[str, Any]):
    conf = state.get("data_confidence", {})
    st.markdown(
        f"### ⚔️ Competitor Analysis {confidence_html(conf.get('competitor_analysis', 'medium'))}",
        unsafe_allow_html=True,
    )
    st.caption("Competitor data is always medium confidence — sourced from external scraping + LLM knowledge.")

    competitors = state.get("identified_competitors", [])
    if competitors:
        st.markdown("**Identified Competitors:** " + " · ".join([f"`{c}`" for c in competitors]))
    else:
        st.info("No competitors identified.")

    st.markdown("---")
    st.markdown("### 📊 Comparison Table")
    comp_table = state.get("competitor_table", "")
    if comp_table:
        st.markdown(comp_table)
    else:
        st.info("Comparison table not available.")

    st.markdown("---")
    st.markdown("### 🔍 Detailed Profiles")
    analyses = state.get("competitor_analyses", [])
    if analyses:
        for comp in analyses:
            with st.expander(f"🏢 {comp.get('name', 'Unknown')}", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Website:** {comp.get('website', 'N/A')}")
                    st.markdown(f"**Description:** {comp.get('description', 'N/A')}")
                    st.markdown(f"**Positioning:** {comp.get('positioning', 'N/A')}")
                    st.markdown(f"**Value Proposition:** {comp.get('value_proposition', 'N/A')}")
                with col2:
                    strengths = comp.get("strengths", [])
                    weaknesses = comp.get("weaknesses", [])
                    if strengths:
                        st.markdown("**💪 Strengths:**")
                        for s in strengths:
                            st.markdown(f"- {s}")
                    if weaknesses:
                        st.markdown("**⚠️ Weaknesses:**")
                        for w in weaknesses:
                            st.markdown(f"- {w}")

                msg_comp = comp.get("messaging_comparison", "")
                if msg_comp:
                    st.markdown(f"**Messaging Comparison:** {msg_comp}")
    else:
        st.info("No detailed competitor profiles available.")


def render_marketing_tab(state: Dict[str, Any]):
    conf = state.get("data_confidence", {})
    st.markdown("### 📢 Product Marketing Intelligence")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🛍️ Products & Services",
        "🎯 Positioning",
        "💬 Messaging",
        "💰 Pricing",
        "🌟 Differentiation",
    ])

    with tab1:
        st.markdown(
            f"#### Products & Services {confidence_html(conf.get('products_services', 'low'))}",
            unsafe_allow_html=True,
        )
        st.markdown(state.get("products_services", "*Not available*"))

    with tab2:
        st.markdown(
            f"#### Positioning {confidence_html(conf.get('product_positioning', 'low'))}",
            unsafe_allow_html=True,
        )
        st.markdown(state.get("product_positioning", "*Not available*"))

    with tab3:
        st.markdown(
            f"#### Messaging {confidence_html(conf.get('market_messaging', 'low'))}",
            unsafe_allow_html=True,
        )
        st.markdown(state.get("market_messaging", "*Not available*"))
        st.markdown("---")
        st.markdown(
            f"#### 💡 Value Propositions {confidence_html(conf.get('value_propositions', 'low'))}",
            unsafe_allow_html=True,
        )
        st.markdown(state.get("value_propositions", "*Not available*"))

    with tab4:
        st.markdown(
            f"#### Pricing {confidence_html(conf.get('pricing_insights', 'low'))}",
            unsafe_allow_html=True,
        )
        st.markdown(state.get("pricing_insights", "*Not available*"))

    with tab5:
        st.markdown(
            f"#### Differentiation {confidence_html(conf.get('market_differentiation', 'low'))}",
            unsafe_allow_html=True,
        )
        st.markdown(state.get("market_differentiation", "*Not available*"))


def render_lead_tab(state: Dict[str, Any]):
    conf = state.get("data_confidence", {})
    st.markdown(
        f"### 🎯 Lead Generation Intelligence {confidence_html(conf.get('lead_score', 'low'))}",
        unsafe_allow_html=True,
    )

    lead_score = state.get("lead_score") or {}
    if lead_score:
        score = lead_score.get("total_score", 0)
        label = _score_label(score)
        color = _score_color(score)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"### Lead Score")
            st.markdown(f"<h1 style='color:{color}'>{score}/100</h1>", unsafe_allow_html=True)
            st.markdown(f"**{label}**")

        with col2:
            st.markdown("**Score Breakdown:**")
            score_data = {
                "Category": ["Hiring", "Funding", "Product Launch", "Expansion", "Partnerships"],
                "Score": [
                    lead_score.get("hiring_score", 0),
                    lead_score.get("funding_score", 0),
                    lead_score.get("launch_score", 0),
                    lead_score.get("expansion_score", 0),
                    lead_score.get("partnership_score", 0),
                ],
                "Max": [25, 30, 20, 15, 10],
            }
            df = pd.DataFrame(score_data)
            df["Utilization"] = (df["Score"] / df["Max"] * 100).round(0).astype(int).astype(str) + "%"
            st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("**Justification:**")
        st.info(lead_score.get("justification", "No justification available."))

    st.markdown("---")
    st.markdown(
        f"### 📡 Detected Signals {confidence_html(conf.get('lead_signals', 'low'))}",
        unsafe_allow_html=True,
    )

    signals = state.get("lead_signals", [])
    if signals:
        signal_types = {}
        for s in signals:
            stype = s.get("signal_type", "other") if isinstance(s, dict) else s.signal_type
            signal_types.setdefault(stype, []).append(s)

        type_icons = {
            "hiring": "👥",
            "funding": "💰",
            "launch": "🚀",
            "expansion": "🌍",
            "partnership": "🤝",
        }

        for stype, slist in signal_types.items():
            icon = type_icons.get(stype, "📌")
            with st.expander(f"{icon} {stype.capitalize()} Signals ({len(slist)})", expanded=True):
                for sig in slist:
                    if isinstance(sig, dict):
                        desc = sig.get("description", "")
                        conf = sig.get("confidence", "medium")
                        source = sig.get("source", "")
                        date = sig.get("date_mentioned", "")
                    else:
                        desc = sig.description
                        conf = sig.confidence
                        source = sig.source
                        date = sig.date_mentioned

                    conf_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(conf, "⚪")
                    st.markdown(f"{conf_color} **{desc}**")
                    if date or source:
                        st.caption(f"{date}{' · ' if date and source else ''}{source}")
    else:
        st.info("No specific signals detected.")

    st.markdown("---")
    st.markdown(
        f"### 📧 Outreach Recommendations {confidence_html(conf.get('outreach_recommendations', 'low'))}",
        unsafe_allow_html=True,
    )
    st.markdown(state.get("outreach_recommendations", "*Not available*"))

    # ── STEP 6: Email Draft ───────────────────────────────────
    email_draft = state.get("outreach_email_draft", "")
    if email_draft:
        st.markdown("---")
        st.markdown("### ✉️ Ready-to-Send Email Draft")
        st.caption("Personalised to the strongest detected signal. Edit before sending.")
        st.code(email_draft, language="text")
        st.download_button(
            "📋 Copy / Download Email",
            data=email_draft,
            file_name=f"outreach_{state.get('input', {}).get('company_name', 'company').lower().replace(' ', '_')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

    # ── STEP 7: Signal Timeline Chart ────────────────────────
    st.markdown("---")
    st.markdown("### 📅 Signal Timeline")
    _render_signal_timeline(signals)


def render_swot_tab(state: Dict[str, Any]):
    conf = state.get("data_confidence", {})
    st.markdown(
        f"### ⚡ SWOT Analysis {confidence_html(conf.get('swot_analysis', 'low'))}",
        unsafe_allow_html=True,
    )
    swot = state.get("swot_analysis") or {}

    if swot:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 💪 Strengths")
            strengths = swot.get("strengths", []) if isinstance(swot, dict) else swot.strengths
            for s in strengths:
                st.success(f"✅ {s}")

            st.markdown("#### 🌟 Opportunities")
            opps = swot.get("opportunities", []) if isinstance(swot, dict) else swot.opportunities
            for o in opps:
                st.info(f"💡 {o}")

        with col2:
            st.markdown("#### 🔴 Weaknesses")
            weaknesses = swot.get("weaknesses", []) if isinstance(swot, dict) else swot.weaknesses
            for w in weaknesses:
                st.warning(f"⚠️ {w}")

            st.markdown("#### ⚠️ Threats")
            threats = swot.get("threats", []) if isinstance(swot, dict) else swot.threats
            for t in threats:
                st.error(f"🚨 {t}")
    else:
        st.info("SWOT analysis not available.")


def render_sources_tab(state: Dict[str, Any]):
    conf = state.get("data_confidence", {})
    st.markdown("### 🔗 Sources & References")

    # ── Confidence legend ──────────────────────────────────────
    if conf:
        with st.expander("📊 Data Confidence Summary", expanded=True):
            st.caption("How much scraped data backed each section of this report.")
            cols = st.columns(3)
            section_labels = {
                "company_overview":       "Company Overview",
                "products_services":      "Products & Services",
                "product_positioning":    "Positioning",
                "market_messaging":       "Messaging",
                "pricing_insights":       "Pricing",
                "market_differentiation": "Differentiation",
                "competitor_analysis":    "Competitor Analysis",
                "lead_signals":           "Lead Signals",
                "lead_score":             "Lead Score",
                "recent_activities":      "Recent Activities",
                "swot_analysis":          "SWOT Analysis",
            }
            for i, (key, label) in enumerate(section_labels.items()):
                level = conf.get(key, "low")
                badge = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(level, "⚪")
                cols[i % 3].markdown(f"{badge} **{label}**: {level.upper()}")

    sources = state.get("sources", [])
    if sources:
        unique_sources = list(dict.fromkeys(sources))
        st.markdown(f"**Total sources collected:** {len(unique_sources)}")

        # Categorize
        website_sources = [s for s in unique_sources if "news" not in s.lower()]
        news_sources = [s for s in unique_sources if any(
            domain in s.lower() for domain in ["techcrunch", "bloomberg", "reuters", "verge", "wired"]
        )]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🌐 Website Sources:**")
            for i, src in enumerate(website_sources[:15], 1):
                st.markdown(f"{i}. [{src[:60]}...]({src})" if len(src) > 60 else f"{i}. [{src}]({src})")

        with col2:
            if news_sources:
                st.markdown("**📰 News Sources:**")
                for i, src in enumerate(news_sources[:15], 1):
                    st.markdown(f"{i}. [{src[:60]}...]({src})" if len(src) > 60 else f"{i}. [{src}]({src})")

    else:
        st.info("No sources tracked.")

    st.markdown("---")
    st.markdown("### 🤖 Agent Execution Log")
    logs = state.get("agent_logs", [])
    if logs:
        log_text = "\n".join(logs)
        st.code(log_text, language="text")
    else:
        st.info("No agent logs available.")


def _stream_executive_briefing(report_text: str):
    """Stream a short LLM-generated briefing in the dashboard."""
    prompt = f"""Create a concise executive briefing from this business intelligence report.

Return:
1. One-paragraph situation summary
2. Three most important insights
3. Recommended next action

REPORT:
{report_text[:8000]}
"""
    try:
        llm = get_llm_client()
        stream = llm.stream_complete(
            prompt=prompt,
            system_prompt="You write concise business briefings for sales and marketing leaders.",
            max_tokens=500,
            use_cache=True,
        )
        if hasattr(st, "write_stream"):
            st.write_stream(stream)
        else:
            st.markdown("".join(list(stream)))
    except Exception as exc:
        st.warning(f"Streaming briefing failed: {exc}")


# ── Signal Timeline (Step 7) ──────────────────────────────────

def _render_signal_timeline(signals: list):
    """Plot detected signals on a timeline using Plotly."""
    if not signals:
        st.info("No signals to plot.")
        return

    if not _PLOTLY_AVAILABLE:
        st.caption("Install `plotly` for the timeline chart: `pip install plotly`")
        return

    # Build rows — include signals with or without dates
    rows = []
    for s in signals:
        if isinstance(s, dict):
            stype = s.get("signal_type", "other")
            desc = s.get("description", "")
            date_str = s.get("date_mentioned", "")
            conf = s.get("confidence", "medium")
        else:
            stype = s.signal_type
            desc = s.description
            date_str = s.date_mentioned
            conf = s.confidence

        # Normalise date string to something parseable
        parsed_date = _parse_signal_date(date_str)
        rows.append({
            "Signal Type": stype.capitalize(),
            "Description": desc[:80] + ("…" if len(desc) > 80 else ""),
            "Date": parsed_date,
            "Confidence": conf.capitalize(),
            "has_date": parsed_date is not None,
        })

    dated = [r for r in rows if r["has_date"]]
    undated = [r for r in rows if not r["has_date"]]

    if dated:
        df = pd.DataFrame(dated)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date")

        color_map = {"High": "#e63946", "Medium": "#f4a261", "Low": "#2a9d8f"}
        fig = px.scatter(
            df,
            x="Date",
            y="Signal Type",
            color="Confidence",
            color_discrete_map=color_map,
            hover_data={"Description": True, "Confidence": True, "Date": True, "Signal Type": False},
            size_max=14,
            title="Signals over Time",
        )
        fig.update_traces(marker=dict(size=14, line=dict(width=1, color="white")))
        fig.update_layout(
            height=320,
            margin=dict(l=0, r=0, t=40, b=0),
            plot_bgcolor="#f8f9fa",
            paper_bgcolor="white",
            yaxis_title="",
            xaxis_title="",
            legend_title="Confidence",
        )
        st.plotly_chart(fig, use_container_width=True)

    if undated:
        st.caption(f"⚪ {len(undated)} signal(s) with no date detected — not shown on timeline:")
        for r in undated:
            st.markdown(f"- **{r['Signal Type']}**: {r['Description']}")


def _parse_signal_date(date_str: str):
    """Try to extract a parseable date from signal date strings like 'Q1 2025', 'Jan 2025', '2024'."""
    if not date_str:
        return None
    import re as _re
    from datetime import datetime

    # Try direct parse
    for fmt in ("%B %Y", "%b %Y", "%Y-%m-%d", "%Y-%m", "%B %d, %Y"):
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            pass

    # Quarter: Q1 2025 → Jan 1 2025
    q_match = _re.search(r"Q([1-4])\s*(\d{4})", date_str)
    if q_match:
        quarter_start = {"1": "01", "2": "04", "3": "07", "4": "10"}
        month = quarter_start[q_match.group(1)]
        return datetime.strptime(f"{q_match.group(2)}-{month}-01", "%Y-%m-%d")

    # Just a year: 2024 → Jan 1 2024
    yr_match = _re.search(r"\b(20\d{2})\b", date_str)
    if yr_match:
        try:
            return datetime.strptime(f"{yr_match.group(1)}-01-01", "%Y-%m-%d")
        except ValueError:
            pass

    return None


# ── Score Helpers ─────────────────────────────────────────────

def _score_label(score: int) -> str:
    if score >= 80:
        return "🔥 HOT LEAD"
    elif score >= 60:
        return "✅ WARM LEAD"
    elif score >= 40:
        return "🔵 MODERATE"
    elif score >= 20:
        return "🟡 COOL LEAD"
    else:
        return "⚪ LOW SIGNAL"


def _score_color(score: int) -> str:
    if score >= 70:
        return "#ff4444"
    elif score >= 40:
        return "#ff9800"
    else:
        return "#4caf50"


def _score_delta(score: int) -> str:
    if score >= 70:
        return "Hot Lead"
    elif score >= 40:
        return "Warm Lead"
    else:
        return "Low Signal"


# ── Batch Analysis Tab (Step 8) ───────────────────────────────

def render_batch_tab():
    """
    STEP 8: Accept up to 10 companies via CSV or text input,
    run all of them, and output a ranked lead score table.
    """
    st.markdown("### 📊 Batch Company Analysis")
    st.caption("Rank multiple companies by lead score in one run. Max 10 companies.")

    # Input method toggle
    input_method = st.radio(
        "Input method",
        ["Paste names", "Upload CSV"],
        horizontal=True,
        label_visibility="collapsed",
    )

    companies: list = []

    if input_method == "Paste names":
        raw = st.text_area(
            "Company names (one per line)",
            placeholder="OpenAI\nAnthropic\nElevenLabs\nNotion",
            height=140,
        )
        if raw.strip():
            companies = [c.strip() for c in raw.strip().splitlines() if c.strip()]
    else:
        uploaded = st.file_uploader("Upload CSV (one company name per row, no header)", type=["csv"])
        if uploaded:
            df_upload = pd.read_csv(uploaded, header=None)
            companies = df_upload.iloc[:, 0].dropna().astype(str).tolist()

    companies = companies[:10]  # hard cap

    if companies:
        st.markdown(f"**{len(companies)} companies queued:** " + " · ".join([f"`{c}`" for c in companies]))

    run_batch = st.button(
        "🚀 Run Batch Analysis",
        disabled=not companies,
        type="primary",
        use_container_width=True,
    )

    if run_batch and companies:
        results = []
        progress = st.progress(0)
        status = st.empty()

        for i, company in enumerate(companies):
            status.markdown(f"**Analysing {i+1}/{len(companies)}:** `{company}`...")
            try:
                state = run_intelligence_pipeline(company_name=company)
                lead_score = state.get("lead_score") or {}
                score_val = (
                    lead_score.get("total_score", 0)
                    if isinstance(lead_score, dict)
                    else getattr(lead_score, "total_score", 0)
                )
                competitors = state.get("identified_competitors", [])
                signals = state.get("lead_signals", [])
                results.append({
                    "Company": company,
                    "Lead Score": score_val,
                    "Grade": _score_label(score_val),
                    "Competitors Found": len(competitors),
                    "Signals Detected": len(signals),
                    "Top Competitors": ", ".join(competitors[:3]),
                })
            except Exception as e:
                results.append({
                    "Company": company,
                    "Lead Score": 0,
                    "Grade": "❌ Error",
                    "Competitors Found": 0,
                    "Signals Detected": 0,
                    "Top Competitors": str(e)[:60],
                })
            progress.progress(int((i + 1) / len(companies) * 100))

        status.markdown("**✅ Batch complete!**")
        st.session_state["batch_results"] = results

    # Show results
    if "batch_results" in st.session_state:
        results = st.session_state["batch_results"]
        df_results = pd.DataFrame(results).sort_values("Lead Score", ascending=False).reset_index(drop=True)
        df_results.index += 1  # rank from 1

        st.markdown("#### 🏆 Ranked Results")
        st.dataframe(df_results, use_container_width=True)

        # Bar chart
        if _PLOTLY_AVAILABLE and len(df_results) > 1:
            fig = px.bar(
                df_results,
                x="Company",
                y="Lead Score",
                color="Lead Score",
                color_continuous_scale=["#4caf50", "#ff9800", "#ff4444"],
                range_color=[0, 100],
                text="Lead Score",
                title="Lead Score Comparison",
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(
                height=350,
                showlegend=False,
                plot_bgcolor="#f8f9fa",
                paper_bgcolor="white",
                yaxis=dict(range=[0, 110]),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        # CSV download
        csv_data = df_results.to_csv(index=False)
        st.download_button(
            "📥 Download Results CSV",
            data=csv_data,
            file_name="batch_lead_scores.csv",
            mime="text/csv",
            use_container_width=True,
        )

        if st.button("🗑️ Clear Batch Results", use_container_width=True):
            del st.session_state["batch_results"]
            st.rerun()


# ── Compare View (Step 4 extension) ──────────────────────────

def render_compare_view(entry_a: dict, entry_b: dict):
    """Side-by-side comparison of two past runs from run_history.json."""
    co_a = entry_a["company"]
    co_b = entry_b["company"]

    st.markdown(f"## ⚖️ Comparing: **{co_a}** vs **{co_b}**")
    if st.button("✖ Close Comparison", use_container_width=False):
        del st.session_state["compare_a"]
        del st.session_state["compare_b"]
        st.rerun()

    # ── Score comparison ──────────────────────────────────────
    st.markdown("### 🎯 Lead Score")
    col1, col2 = st.columns(2)
    score_a = entry_a.get("lead_score", 0)
    score_b = entry_b.get("lead_score", 0)
    with col1:
        color_a = "#ff4444" if score_a >= 70 else "#ff9800" if score_a >= 40 else "#4caf50"
        st.markdown(f"**{co_a}**")
        st.markdown(f"<h1 style='color:{color_a}'>{score_a}/100</h1>", unsafe_allow_html=True)
        st.caption(entry_a.get("datetime", ""))
    with col2:
        color_b = "#ff4444" if score_b >= 70 else "#ff9800" if score_b >= 40 else "#4caf50"
        st.markdown(f"**{co_b}**")
        st.markdown(f"<h1 style='color:{color_b}'>{score_b}/100</h1>", unsafe_allow_html=True)
        st.caption(entry_b.get("datetime", ""))

    # Bar chart if plotly available
    if _PLOTLY_AVAILABLE:
        fig = px.bar(
            x=[co_a, co_b],
            y=[score_a, score_b],
            color=[score_a, score_b],
            color_continuous_scale=["#4caf50", "#ff9800", "#ff4444"],
            range_color=[0, 100],
            text=[score_a, score_b],
            labels={"x": "Company", "y": "Lead Score"},
        )
        fig.update_traces(textposition="outside", width=0.4)
        fig.update_layout(
            height=280, showlegend=False,
            plot_bgcolor="#f8f9fa", paper_bgcolor="white",
            yaxis=dict(range=[0, 115]), coloraxis_showscale=False,
            margin=dict(t=20, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Side-by-side metrics ──────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Key Metrics")
    metrics = [
        ("RAG Chunks", "rag_chunks"),
        ("Competitors Found", "competitor_count"),
        ("Signals Detected", "signal_count"),
    ]
    cols = st.columns(len(metrics))
    for col, (label, key) in zip(cols, metrics):
        val_a = entry_a.get(key, 0)
        val_b = entry_b.get(key, 0)
        delta = val_b - val_a
        col.metric(f"{label}", f"{co_a}: {val_a} | {co_b}: {val_b}", delta=f"{co_b} +{delta}" if delta > 0 else None)

    # ── Competitors side by side ──────────────────────────────
    st.markdown("---")
    st.markdown("### ⚔️ Competitors")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**{co_a}**")
        for c in entry_a.get("competitors", []):
            st.markdown(f"- `{c}`")
    with col2:
        st.markdown(f"**{co_b}**")
        for c in entry_b.get("competitors", []):
            st.markdown(f"- `{c}`")

    # ── Full reports side by side ─────────────────────────────
    st.markdown("---")
    st.markdown("### 📄 Full Reports Side by Side")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**{co_a}**")
        path_a = entry_a.get("filepath", "")
        if path_a and Path(path_a).exists():
            with st.expander("View report", expanded=False):
                st.markdown(Path(path_a).read_text(encoding="utf-8"))
        else:
            st.caption("Report file not found.")
    with col2:
        st.markdown(f"**{co_b}**")
        path_b = entry_b.get("filepath", "")
        if path_b and Path(path_b).exists():
            with st.expander("View report", expanded=False):
                st.markdown(Path(path_b).read_text(encoding="utf-8"))
        else:
            st.caption("Report file not found.")


# ── Main App ──────────────────────────────────────────────────

def main():
    render_header()
    inputs = render_sidebar()

    # Main content area
    if "result_state" not in st.session_state:
        # Welcome screen
        st.markdown("""
        ## 👋 Welcome to AI Market Intelligence Agent

        This tool automates **business and competitive intelligence** gathering for any company.
        It uses a **multi-agent LangGraph pipeline** powered by **Groq's Llama 3.3** to:

        - 🔎 **Research** company websites and news
        - 📢 **Analyze** product positioning and messaging
        - ⚔️ **Profile** competitors with comparison tables
        - 🎯 **Score** lead quality based on buying signals
        - 📄 **Generate** a complete BI report

        ### How to use:
        1. Enter a **company name** in the sidebar (required)
        2. Optionally add the website URL, product name, or scenario
        3. Click **🚀 Generate Intelligence**

        ### Example inputs:
        - `OpenAI` — analyze ChatGPT/GPT-4 ecosystem
        - `ElevenLabs` + scenario: "AI voice agent for lead gen"
        - `Salesforce` + product: "Salesforce Einstein"
        """)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(
                "https://img.shields.io/badge/LangGraph-Multi--Agent-blue?style=for-the-badge",
                width=250,
            )

        st.markdown("---")
        render_batch_tab()

    # Generate button
    with st.sidebar:
        st.markdown("---")
        generate_btn = st.button(
            "🚀 Generate Intelligence",
            use_container_width=True,
            type="primary",
            disabled=not inputs["company_name"].strip(),
        )

        if "result_state" in st.session_state:
            if st.button("🗑️ Clear Results", use_container_width=True):
                del st.session_state["result_state"]
                st.rerun()

            state = st.session_state["result_state"]
            report_text = state.get("final_report", "")
            company_slug = inputs["company_name"].lower().replace(" ", "_")

            # Save on first load (not for history-loaded states)
            if report_text and not state.get("_from_history"):
                save_report(state)

            # Markdown download
            st.download_button(
                "📥 Download .md",
                data=report_text,
                file_name=f"intel_{company_slug}.md",
                mime="text/markdown",
                use_container_width=True,
            )

            # PDF download — generate on click
            if st.button("📄 Download PDF", use_container_width=True):
                with st.spinner("Generating PDF..."):
                    pdf_bytes = generate_pdf_bytes(report_text)
                if pdf_bytes:
                    st.download_button(
                        "⬇️ Save PDF now",
                        data=pdf_bytes,
                        file_name=f"intel_{company_slug}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                else:
                    st.warning(
                        "PDF export requires `weasyprint` and `markdown`.\n\n"
                        "Install with: `pip install weasyprint markdown`"
                    )

    # Run pipeline
    if generate_btn and inputs["company_name"].strip():
        with st.spinner(""):
            result = run_with_progress(inputs)
        if result:
            st.session_state["result_state"] = result
            st.success(f"✅ Intelligence report generated for **{inputs['company_name']}**!")
            st.rerun()

    # ── Compare view (triggered from sidebar) ────────────────
    if "compare_a" in st.session_state and "compare_b" in st.session_state:
        render_compare_view(st.session_state["compare_a"], st.session_state["compare_b"])
        return  # don't show normal results while comparing

    # Display results
    if "result_state" in st.session_state:
        state = st.session_state["result_state"]
        company = state.get("input", {}).get("company_name", "Unknown")

        st.markdown(f"## 📊 Intelligence Report: {company}")

        tabs = st.tabs([
            "📋 Overview",
            "⚔️ Competitors",
            "📢 Marketing Intel",
            "🎯 Lead Intelligence",
            "⚡ SWOT Analysis",
            "🔗 Sources",
            "📊 Batch Analysis",
        ])

        with tabs[0]:
            render_overview_tab(state)
        with tabs[1]:
            render_competitors_tab(state)
        with tabs[2]:
            render_marketing_tab(state)
        with tabs[3]:
            render_lead_tab(state)
        with tabs[4]:
            render_swot_tab(state)
        with tabs[5]:
            render_sources_tab(state)
        with tabs[6]:
            render_batch_tab()

        # Full report expander
        with st.expander("📄 Full Markdown Report", expanded=False):
            report = state.get("final_report", "")
            st.markdown(report)
            st.download_button(
                "📥 Download Full Report (.md)",
                data=report,
                file_name=f"intel_{company.lower().replace(' ', '_')}.md",
                mime="text/markdown",
            )


if __name__ == "__main__":
    main()
