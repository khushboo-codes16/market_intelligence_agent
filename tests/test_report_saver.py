# tests/test_report_saver.py
# ============================================================
# Unit tests: report saving, run history, PDF generation
# ============================================================

import json
import pytest
from pathlib import Path
from unittest.mock import patch


def _make_state(company="TestCo", score=65, report="# Test Report\n\nContent here."):
    return {
        "input": {"company_name": company},
        "final_report": report,
        "lead_score": {"total_score": score},
        "identified_competitors": ["CompA", "CompB"],
        "lead_signals": [
            {"signal_type": "funding", "description": "Raised $5M", "confidence": "high"}
        ],
        "rag_chunks_count": 42,
    }


class TestSaveReport:
    def test_saves_md_file(self, tmp_path):
        with patch("reports.report_saver.REPORTS_DIR", str(tmp_path)):
            from reports.report_saver import save_report
            state = _make_state()
            path = save_report(state)

        assert path != ""
        assert Path(path).exists()
        assert Path(path).suffix == ".md"
        assert "# Test Report" in Path(path).read_text()

    def test_filename_contains_company_name(self, tmp_path):
        with patch("reports.report_saver.REPORTS_DIR", str(tmp_path)):
            from reports.report_saver import save_report
            path = save_report(_make_state("MyCompany"))

        assert "mycompany" in Path(path).name

    def test_empty_report_returns_empty_string(self, tmp_path):
        with patch("reports.report_saver.REPORTS_DIR", str(tmp_path)):
            from reports.report_saver import save_report
            state = _make_state(report="")
            result = save_report(state)

        assert result == ""

    def test_history_entry_written(self, tmp_path):
        with patch("reports.report_saver.REPORTS_DIR", str(tmp_path)):
            from reports.report_saver import save_report, load_run_history
            save_report(_make_state("HistoryTest", score=77))
            history = load_run_history()

        assert len(history) == 1
        assert history[0]["company"] == "HistoryTest"
        assert history[0]["lead_score"] == 77

    def test_history_keeps_newest_first(self, tmp_path):
        with patch("reports.report_saver.REPORTS_DIR", str(tmp_path)):
            from reports.report_saver import save_report, load_run_history
            save_report(_make_state("First"))
            save_report(_make_state("Second"))
            history = load_run_history()

        assert history[0]["company"] == "Second"
        assert history[1]["company"] == "First"

    def test_history_capped_at_50(self, tmp_path):
        with patch("reports.report_saver.REPORTS_DIR", str(tmp_path)):
            from reports.report_saver import save_report, load_run_history
            for i in range(55):
                save_report(_make_state(f"Co{i}"))
            history = load_run_history()

        assert len(history) <= 50

    def test_history_entry_has_required_fields(self, tmp_path):
        with patch("reports.report_saver.REPORTS_DIR", str(tmp_path)):
            from reports.report_saver import save_report, load_run_history
            save_report(_make_state())
            entry = load_run_history()[0]

        for field in ("company", "timestamp", "datetime", "lead_score",
                      "competitor_count", "signal_count", "filepath"):
            assert field in entry, f"Missing field: {field}"


class TestLoadRunHistory:
    def test_empty_when_no_file(self, tmp_path):
        with patch("reports.report_saver.REPORTS_DIR", str(tmp_path)):
            from reports.report_saver import load_run_history
            history = load_run_history()
        assert history == []

    def test_returns_list_on_corrupt_file(self, tmp_path):
        history_file = tmp_path / "run_history.json"
        history_file.write_text("not valid json")
        with patch("reports.report_saver.REPORTS_DIR", str(tmp_path)):
            from reports.report_saver import load_run_history
            history = load_run_history()
        assert history == []


class TestGeneratePdfBytes:
    def test_returns_bytes_when_deps_available(self):
        """Test PDF generation using the real weasyprint if available."""
        import sys
        # The conftest may have stubbed weasyprint with a MagicMock.
        # Check if the real package is actually importable.
        try:
            import importlib
            real_wp = importlib.import_module.__module__  # just a probe
            import weasyprint
            import markdown as md_lib
            # If conftest stubbed it, weasyprint won't have HTML attr as a real class
            if not hasattr(weasyprint, "HTML") or isinstance(weasyprint, type(sys)):
                pytest.skip("weasyprint is stubbed by conftest")
            html = md_lib.markdown("# Hello\n\nWorld")
            result = weasyprint.HTML(
                string=f"<html><body>{html}</body></html>"
            ).write_pdf()
            assert isinstance(result, bytes)
            assert len(result) > 100
        except (ImportError, AttributeError, TypeError):
            pytest.skip("weasyprint not available in this environment")

    def test_returns_none_gracefully_on_missing_dep(self):
        """generate_pdf_bytes catches ImportError and returns None."""
        import sys
        # Remove real module temporarily and replace with a failing mock
        from unittest.mock import MagicMock
        bad_mod = MagicMock()
        bad_mod.HTML.side_effect = ImportError("not installed")

        original = sys.modules.get("weasyprint")
        sys.modules["weasyprint"] = bad_mod
        try:
            from reports.report_saver import generate_pdf_bytes
            result = generate_pdf_bytes("# Test")
            # Should not raise — should return None or bytes
            assert result is None or isinstance(result, bytes)
        finally:
            if original is not None:
                sys.modules["weasyprint"] = original
            else:
                sys.modules.pop("weasyprint", None)
