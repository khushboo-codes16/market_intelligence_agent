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

from workflow.graph import run_intelligence_pipeline
from reports.report_saver import save_report
from models.state import IntelligenceState


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

PIPELINE_STAGES = [
    ("🔎 Research Agent", "Scraping website & news, building RAG index..."),
    ("📢 Marketing Agent", "Analyzing positioning, messaging & value props..."),
    ("⚔️ Competitor Agent", "Identifying & profiling competitors..."),
    ("🎯 Lead Gen Agent", "Extracting signals & computing lead score..."),
    ("📄 Report Agent", "Compiling final intelligence report..."),
]


def run_with_progress(inputs: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """Run pipeline with live progress updates."""

    progress_container = st.container()
    with progress_container:
        st.markdown("### ⚙️ Running Intelligence Pipeline")
        progress_bar = st.progress(0)
        stage_placeholder = st.empty()
        log_placeholder = st.empty()

        # Animate through stages
        def update_ui(stage_idx: int, message: str):
            progress = int(((stage_idx + 1) / len(PIPELINE_STAGES)) * 90)
            progress_bar.progress(progress)
            stage_name, stage_desc = PIPELINE_STAGES[stage_idx]
            stage_placeholder.markdown(
                f"**{stage_name}** — {stage_desc}"
            )

        # Run pipeline in thread so UI can update
        result_container = {"result": None, "error": None}

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

        # Animate progress while running
        stage_idx = 0
        update_ui(0, "Starting...")
        while thread.is_alive():
            time.sleep(3)
            if stage_idx < len(PIPELINE_STAGES) - 1:
                stage_idx += 1
                update_ui(stage_idx, "Processing...")

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
    company = state.get("input", {}).get("company_name", "Unknown")

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

    st.markdown("---")
    st.markdown("### 📋 Executive Summary")
    st.markdown(state.get("executive_summary", "*Not available*"))

    st.markdown("---")
    st.markdown("### 🏢 Company Overview")
    st.markdown(state.get("company_overview", "*Not available*"))

    st.markdown("---")
    st.markdown("### 📰 Recent Activities")
    st.markdown(state.get("recent_activities", "*Not available*"))


def render_competitors_tab(state: Dict[str, Any]):
    st.markdown("### ⚔️ Competitor Analysis")

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
    st.markdown("### 📢 Product Marketing Intelligence")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🛍️ Products & Services",
        "🎯 Positioning",
        "💬 Messaging",
        "💰 Pricing",
        "🌟 Differentiation",
    ])

    with tab1:
        st.markdown(state.get("products_services", "*Not available*"))

    with tab2:
        st.markdown(state.get("product_positioning", "*Not available*"))

    with tab3:
        st.markdown(state.get("market_messaging", "*Not available*"))
        st.markdown("---")
        st.markdown("#### 💡 Value Propositions")
        st.markdown(state.get("value_propositions", "*Not available*"))

    with tab4:
        st.markdown(state.get("pricing_insights", "*Not available*"))

    with tab5:
        st.markdown(state.get("market_differentiation", "*Not available*"))


def render_lead_tab(state: Dict[str, Any]):
    st.markdown("### 🎯 Lead Generation Intelligence")

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
    st.markdown("### 📡 Detected Signals")

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
    st.markdown("### 📧 Outreach Recommendations")
    st.markdown(state.get("outreach_recommendations", "*Not available*"))


def render_swot_tab(state: Dict[str, Any]):
    st.markdown("### ⚡ SWOT Analysis")
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
    st.markdown("### 🔗 Sources & References")

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

            if st.button("💾 Download Report", use_container_width=True):
                state = st.session_state["result_state"]
                filepath = save_report(state)
                report_text = state.get("final_report", "")
                st.download_button(
                    "📥 Download .md",
                    data=report_text,
                    file_name=f"intel_{inputs['company_name'].lower().replace(' ', '_')}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

    # Run pipeline
    if generate_btn and inputs["company_name"].strip():
        with st.spinner(""):
            result = run_with_progress(inputs)
        if result:
            st.session_state["result_state"] = result
            st.success(f"✅ Intelligence report generated for **{inputs['company_name']}**!")
            st.rerun()

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
