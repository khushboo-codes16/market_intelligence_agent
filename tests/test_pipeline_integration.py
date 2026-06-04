# tests/test_pipeline_integration.py
# ============================================================
# Integration tests: full pipeline with all external calls mocked
# Tests the pipeline flow, state passing, and error handling
# without hitting any real APIs or network
# ============================================================

import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, Any

from models.state import IntelligenceState, ResearchInput


# ── Mock Factories ────────────────────────────────────────────

def _mock_llm_response(text: str = "Mocked LLM output for testing."):
    """Return a mock LLMClient that always returns the given text."""
    mock = MagicMock()
    mock.complete.return_value = text
    mock.chat.return_value = text
    return mock


def _mock_scraper_pages():
    return [
        {"url": "https://testco.com", "title": "TestCo", "text": "TestCo builds AI software. " * 50},
        {"url": "https://testco.com/about", "title": "About", "text": "We are a fast growing startup. " * 50},
    ]


def _mock_scraper_articles():
    return [
        {
            "url": "https://news.com/testco-funding",
            "title": "TestCo Raises $20M",
            "text": "TestCo announced Series B funding of $20M. " * 30,
            "search_title": "TestCo Raises $20M",
            "search_snippet": "TestCo announced Series B funding",
        }
    ]


def _make_initial_state(company="TestCo", website="https://testco.com"):
    inp = ResearchInput(company_name=company, company_website=website)
    return IntelligenceState(input=inp, status="initializing").model_dump()


# ── Supervisor Node ───────────────────────────────────────────

class TestSupervisorNode:
    def test_passes_through_state(self):
        from workflow.graph import supervisor_node
        state = _make_initial_state()
        result = supervisor_node(state)
        assert result["status"] == "initializing"

    def test_appends_to_agent_logs(self):
        from workflow.graph import supervisor_node
        state = _make_initial_state()
        result = supervisor_node(state)
        assert any("[Supervisor]" in log for log in result["agent_logs"])

    def test_warns_when_no_docs_after_research(self):
        from workflow.graph import supervisor_node
        state = _make_initial_state()
        state["status"] = "research_complete"
        state["raw_documents"] = []
        result = supervisor_node(state)
        assert any("Warning" in e for e in result["errors"])


# ── Research Agent ────────────────────────────────────────────

class TestResearchAgent:
    @patch("agents.research_agent.scrape_company_website")
    @patch("agents.research_agent.scrape_news_articles")
    @patch("agents.research_agent.collect_public_intelligence_sources")
    @patch("agents.research_agent.get_rag_pipeline")
    def test_research_sets_status(self, mock_rag, mock_enrich, mock_news, mock_scrape):
        mock_scrape.return_value = _mock_scraper_pages()
        mock_news.return_value = _mock_scraper_articles()
        mock_enrich.return_value = []
        mock_rag_inst = MagicMock()
        mock_rag_inst.ingest_documents.return_value = 15
        mock_rag.return_value = mock_rag_inst

        from agents.research_agent import research_agent
        state = _make_initial_state()
        result = research_agent(state)

        assert result["status"] == "research_complete"

    @patch("agents.research_agent.scrape_company_website")
    @patch("agents.research_agent.scrape_news_articles")
    @patch("agents.research_agent.collect_public_intelligence_sources")
    @patch("agents.research_agent.get_rag_pipeline")
    def test_research_populates_raw_documents(self, mock_rag, mock_enrich, mock_news, mock_scrape):
        mock_scrape.return_value = _mock_scraper_pages()
        mock_news.return_value = []
        mock_enrich.return_value = []
        mock_rag_inst = MagicMock()
        mock_rag_inst.ingest_documents.return_value = 5
        mock_rag.return_value = mock_rag_inst

        from agents.research_agent import research_agent
        state = _make_initial_state()
        result = research_agent(state)

        assert len(result["raw_documents"]) == 2

    @patch("agents.research_agent.scrape_company_website")
    @patch("agents.research_agent.scrape_news_articles")
    @patch("agents.research_agent.collect_public_intelligence_sources")
    @patch("agents.research_agent.get_rag_pipeline")
    def test_research_sets_data_confidence(self, mock_rag, mock_enrich, mock_news, mock_scrape):
        # high = 30+ chunks AND 3+ pages scraped
        mock_scrape.return_value = _mock_scraper_pages() + [
            {"url": "https://testco.com/pricing", "title": "Pricing",
             "text": "Pricing page content " * 50}
        ]
        mock_news.return_value = _mock_scraper_articles() * 3
        mock_enrich.return_value = []
        mock_rag_inst = MagicMock()
        mock_rag_inst.ingest_documents.return_value = 35
        mock_rag.return_value = mock_rag_inst

        from agents.research_agent import research_agent
        state = _make_initial_state()
        result = research_agent(state)

        conf = result.get("data_confidence", {})
        assert conf.get("company_overview") == "high"

    @patch("agents.research_agent.scrape_company_website")
    @patch("agents.research_agent.scrape_news_articles")
    @patch("agents.research_agent.collect_public_intelligence_sources")
    @patch("agents.research_agent.get_rag_pipeline")
    def test_research_guesses_website_when_missing(self, mock_rag, mock_enrich, mock_news, mock_scrape):
        mock_scrape.return_value = []
        mock_news.return_value = []
        mock_enrich.return_value = []
        mock_rag_inst = MagicMock()
        mock_rag_inst.ingest_documents.return_value = 0
        mock_rag.return_value = mock_rag_inst

        from agents.research_agent import research_agent
        state = _make_initial_state(website="")  # no website provided
        result = research_agent(state)

        # Should have guessed a website URL
        assert result["input"]["company_website"] is not None
        assert "http" in result["input"]["company_website"]

    @patch("agents.research_agent.scrape_company_website")
    @patch("agents.research_agent.scrape_news_articles")
    @patch("agents.research_agent.collect_public_intelligence_sources")
    @patch("agents.research_agent.get_rag_pipeline")
    def test_research_ingests_public_enrichment(self, mock_rag, mock_enrich, mock_news, mock_scrape):
        mock_scrape.return_value = _mock_scraper_pages()
        mock_news.return_value = []
        mock_enrich.return_value = [
            {
                "url": "public-search:funding",
                "title": "TestCo funding snippets",
                "text": "TestCo raised Series B funding and is expanding sales hiring. " * 10,
                "source_type": "funding",
            }
        ]
        mock_rag_inst = MagicMock()
        mock_rag_inst.ingest_documents.return_value = 12
        mock_rag.return_value = mock_rag_inst

        from agents.research_agent import research_agent
        state = _make_initial_state()
        result = research_agent(state)

        assert any(doc.get("source_type") == "funding" for doc in result["raw_documents"])
        ingested_docs = mock_rag_inst.ingest_documents.call_args.args[0]
        assert any(doc["source"] == "public-search:funding" for doc in ingested_docs)


# ── Lead Generation Agent ─────────────────────────────────────

class TestLeadGenerationAgent:
    def _patched_lead_agent(self, signals_json: str = "[]"):
        mock_llm = _mock_llm_response()
        mock_llm.complete.side_effect = [
            signals_json,   # hiring signals
            signals_json,   # funding signals
            signals_json,   # launch signals
            signals_json,   # expansion signals
            signals_json,   # partnership signals
            "Recent activities summary",
            "Outreach recommendations",
            "Subject: Test Email\n\nHi [First Name],\n\nTest body\n\nBest,\n[Your Name]",
        ]
        return mock_llm

    @patch("agents.lead_generation_agent.get_rag_pipeline")
    @patch("agents.lead_generation_agent.get_llm_client")
    def test_lead_agent_sets_status(self, mock_llm_fn, mock_rag):
        mock_rag_inst = MagicMock()
        mock_rag_inst.retrieve_as_context.return_value = "Some context"
        mock_rag.return_value = mock_rag_inst
        mock_llm_fn.return_value = self._patched_lead_agent()

        from agents.lead_generation_agent import lead_generation_agent
        state = _make_initial_state()
        state["status"] = "research_complete"
        result = lead_generation_agent(state)

        assert result["status"] == "leads_complete"

    @patch("agents.lead_generation_agent.get_rag_pipeline")
    @patch("agents.lead_generation_agent.get_llm_client")
    def test_lead_agent_produces_email_draft(self, mock_llm_fn, mock_rag):
        mock_rag_inst = MagicMock()
        mock_rag_inst.retrieve_as_context.return_value = "context"
        mock_rag.return_value = mock_rag_inst
        mock_llm_fn.return_value = self._patched_lead_agent()

        from agents.lead_generation_agent import lead_generation_agent
        state = _make_initial_state()
        result = lead_generation_agent(state)

        assert "outreach_email_draft" in result
        assert result["outreach_email_draft"] != ""

    @patch("agents.lead_generation_agent.get_rag_pipeline")
    @patch("agents.lead_generation_agent.get_llm_client")
    def test_lead_score_always_present(self, mock_llm_fn, mock_rag):
        mock_rag_inst = MagicMock()
        mock_rag_inst.retrieve_as_context.return_value = "context"
        mock_rag.return_value = mock_rag_inst
        mock_llm_fn.return_value = self._patched_lead_agent()

        from agents.lead_generation_agent import lead_generation_agent
        state = _make_initial_state()
        result = lead_generation_agent(state)

        assert result["lead_score"] is not None
        assert "total_score" in result["lead_score"]
        assert 0 <= result["lead_score"]["total_score"] <= 100


# ── Parallel Agent Merge ─────────────────────────────────────

class TestParallelMerge:
    def test_merge_fields_from_all_agents(self):
        from workflow.graph import run_parallel_agents

        base_state = _make_initial_state()
        base_state["status"] = "research_complete"
        base_state["rag_chunks_count"] = 20

        marketing_result = dict(base_state)
        marketing_result["company_overview"] = "Great company overview"
        marketing_result["products_services"] = "Amazing products"
        marketing_result["product_positioning"] = "Positioned well"
        marketing_result["value_propositions"] = "Great value"
        marketing_result["market_messaging"] = "Strong messaging"
        marketing_result["pricing_insights"] = "Competitive pricing"
        marketing_result["market_differentiation"] = "Very different"

        competitor_result = dict(base_state)
        competitor_result["identified_competitors"] = ["CompA", "CompB"]
        competitor_result["competitor_table"] = "| Company | Notes |\n|---------|-------|"

        lead_result = dict(base_state)
        lead_result["lead_signals"] = [
            {"signal_type": "funding", "description": "Raised $10M",
             "confidence": "high", "source": "", "date_mentioned": ""}
        ]
        lead_result["lead_score"] = {"total_score": 70, "hiring_score": 10,
                                      "funding_score": 30, "launch_score": 15,
                                      "expansion_score": 10, "partnership_score": 5,
                                      "justification": "Good signals", "signals": []}
        lead_result["outreach_recommendations"] = "Reach out now"
        lead_result["recent_activities"] = "Recently launched product"
        lead_result["outreach_email_draft"] = "Subject: Hi\n\nTest email"

        with patch("workflow.graph.product_marketing_agent", return_value=marketing_result), \
             patch("workflow.graph.competitor_analysis_agent", return_value=competitor_result), \
             patch("workflow.graph.lead_generation_agent", return_value=lead_result):

            merged = run_parallel_agents(base_state)

        assert merged["company_overview"] == "Great company overview"
        assert merged["identified_competitors"] == ["CompA", "CompB"]
        assert merged["lead_score"]["total_score"] == 70
        assert merged["outreach_email_draft"] == "Subject: Hi\n\nTest email"
        assert merged["status"] == "parallel_complete"

    def test_merge_survives_one_agent_failure(self):
        from workflow.graph import run_parallel_agents

        base_state = _make_initial_state()
        base_state["status"] = "research_complete"

        good_result = dict(base_state)
        good_result["company_overview"] = "Still got overview"

        with patch("workflow.graph.product_marketing_agent", return_value=good_result), \
             patch("workflow.graph.competitor_analysis_agent", side_effect=Exception("Scrape failed")), \
             patch("workflow.graph.lead_generation_agent", return_value=dict(base_state)):

            merged = run_parallel_agents(base_state)

        # Pipeline continues despite one failure
        assert merged["status"] == "parallel_complete"
        assert merged["company_overview"] == "Still got overview"
        assert any("competitor_analysis failed" in e for e in merged["errors"])


# ── Pipeline Error Handling ───────────────────────────────────

class TestPipelineErrorHandling:
    @patch("workflow.graph.build_intelligence_graph")
    def test_pipeline_returns_error_state_on_crash(self, mock_build):
        mock_graph = MagicMock()
        mock_graph.invoke.side_effect = Exception("Simulated crash")
        mock_build.return_value = mock_graph

        from workflow.graph import run_intelligence_pipeline
        result = run_intelligence_pipeline("CrashCo")

        assert result["status"] == "failed"
        assert len(result["errors"]) > 0
        assert "# Error" in result["final_report"]

    @patch("workflow.graph.build_intelligence_graph")
    def test_pipeline_result_is_dict(self, mock_build):
        mock_graph = MagicMock()
        mock_graph.invoke.side_effect = Exception("boom")
        mock_build.return_value = mock_graph

        from workflow.graph import run_intelligence_pipeline
        result = run_intelligence_pipeline("TestCo")

        assert isinstance(result, dict)
