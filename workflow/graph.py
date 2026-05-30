# workflow/graph.py
# ============================================================
# LangGraph multi-agent workflow with Supervisor orchestration
# ============================================================

from typing import Dict, Any, TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, END

from agents.research_agent import research_agent
from agents.product_marketing_agent import product_marketing_agent
from agents.competitor_analysis_agent import competitor_analysis_agent
from agents.lead_generation_agent import lead_generation_agent
from agents.report_generation_agent import report_generation_agent
from models.state import IntelligenceState, ResearchInput
from utils.logger import logger


# ── Supervisor Logic ─────────────────────────────────────────

def supervisor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Supervisor Agent: validates input, routes workflow, handles errors.
    This node runs at start and after each phase to log progress.
    """
    intel = IntelligenceState(**state)
    status = intel.status

    logger.info(f"[Supervisor] Status: {status} | "
                f"Docs: {len(intel.raw_documents)} | "
                f"RAG chunks: {intel.rag_chunks_count} | "
                f"Competitors: {len(intel.identified_competitors)}")

    # Validate we have minimum data to proceed
    if status == "research_complete" and len(intel.raw_documents) == 0:
        intel.errors.append("Warning: No documents collected. Analysis may be limited.")
        intel.agent_logs.append("[Supervisor] Warning: No documents scraped — proceeding with LLM knowledge only.")

    intel.agent_logs.append(f"[Supervisor] Routing from status: {status}")
    return intel.model_dump()


def error_handler(state: Dict[str, Any]) -> Dict[str, Any]:
    """Handle errors gracefully without crashing the workflow."""
    intel = IntelligenceState(**state)
    intel.agent_logs.append(f"[Supervisor] Error handled: {intel.errors[-1] if intel.errors else 'Unknown error'}")
    intel.status = "error_handled"
    return intel.model_dump()


def route_after_supervisor(state: Dict[str, Any]) -> str:
    """Determine next node based on current status."""
    status = state.get("status", "")
    routing = {
        "initializing": "research",
        "research_complete": "product_marketing",
        "marketing_complete": "competitor_analysis",
        "competitors_complete": "lead_generation",
        "leads_complete": "report_generation",
        "complete": END,
        "error_handled": "report_generation",
    }
    next_node = routing.get(status, END)
    logger.info(f"[Supervisor] Routing: {status} → {next_node}")
    return next_node


# ── Graph Builder ─────────────────────────────────────────────

def build_intelligence_graph() -> StateGraph:
    """Build and compile the LangGraph multi-agent workflow."""

    # Use dict as state type (Pydantic model serialized to dict)
    workflow = StateGraph(dict)

    # ── Add Nodes ──────────────────────────────────────────────
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("research", research_agent)
    workflow.add_node("product_marketing", product_marketing_agent)
    workflow.add_node("competitor_analysis", competitor_analysis_agent)
    workflow.add_node("lead_generation", lead_generation_agent)
    workflow.add_node("report_generation", report_generation_agent)
    workflow.add_node("error_handler", error_handler)

    # ── Define Flow ────────────────────────────────────────────
    workflow.set_entry_point("supervisor")

    # Supervisor routes to next agent based on status
    workflow.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "research": "research",
            "product_marketing": "product_marketing",
            "competitor_analysis": "competitor_analysis",
            "lead_generation": "lead_generation",
            "report_generation": "report_generation",
            END: END,
        },
    )

    # Each agent feeds back to supervisor for routing
    workflow.add_edge("research", "supervisor")
    workflow.add_edge("product_marketing", "supervisor")
    workflow.add_edge("competitor_analysis", "supervisor")
    workflow.add_edge("lead_generation", "supervisor")
    workflow.add_edge("report_generation", END)

    return workflow.compile()


# ── Main Entry Point ──────────────────────────────────────────

def run_intelligence_pipeline(
    company_name: str,
    company_website: str = "",
    product_name: str = "",
    product_scenario: str = "",
) -> Dict[str, Any]:
    """
    Run the full intelligence pipeline for a company.

    Args:
        company_name: Target company name
        company_website: Optional website URL
        product_name: Optional product to focus on
        product_scenario: Optional scenario description

    Returns:
        Final IntelligenceState dict with complete report
    """
    logger.info(f"Starting intelligence pipeline for: {company_name}")

    # Build initial state
    research_input = ResearchInput(
        company_name=company_name,
        company_website=company_website or None,
        product_name=product_name or None,
        product_scenario=product_scenario or None,
    )

    initial_state = IntelligenceState(
        input=research_input,
        status="initializing",
    ).model_dump()

    # Build and run graph
    graph = build_intelligence_graph()

    try:
        final_state = graph.invoke(initial_state)
        logger.info(f"Pipeline complete. Status: {final_state.get('status')}")
        return final_state
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        # Return partial state with error
        initial_state["errors"] = [str(e)]
        initial_state["status"] = "failed"
        initial_state["final_report"] = f"# Error\n\nPipeline failed: {str(e)}\n\nPlease check your API key and try again."
        return initial_state
