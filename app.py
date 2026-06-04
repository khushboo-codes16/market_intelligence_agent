# ============================================================
# Main entry point for the Market Intelligence Agent
# ============================================================

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── CLI Mode ──────────────────────────────────────────────────

def run_cli():
    """Run pipeline from command line (non-Streamlit mode)."""
    import argparse
    from rich.console import Console
    from rich.markdown import Markdown
    from workflow.graph import run_intelligence_pipeline
    from reports.report_saver import save_report

    console = Console()

    parser = argparse.ArgumentParser(
        description="AI-Powered Business & Market Intelligence Agent"
    )
    parser.add_argument("company", help="Target company name")
    parser.add_argument("--website", "-w", default="", help="Company website URL")
    parser.add_argument("--product", "-p", default="", help="Product name")
    parser.add_argument("--scenario", "-s", default="", help="Product scenario")
    parser.add_argument("--output", "-o", default="", help="Output file path")

    args = parser.parse_args()

    console.print(f"\n[bold blue]🔍 AI Market Intelligence Agent[/bold blue]")
    console.print(f"[dim]Target: {args.company}[/dim]\n")

    with console.status(f"[bold green]Running intelligence pipeline for {args.company}...[/bold green]"):
        state = run_intelligence_pipeline(
            company_name=args.company,
            company_website=args.website,
            product_name=args.product,
            product_scenario=args.scenario,
        )

    if state.get("final_report"):
        console.print(Markdown(state["final_report"][:3000]))
        filepath = save_report(state)
        console.print(f"\n[bold green]✅ Report saved to: {filepath}[/bold green]")
    else:
        console.print("[bold red]Pipeline failed — check logs.[/bold red]")


# ── Streamlit Mode ────────────────────────────────────────────

def run_streamlit():
    """Launch Streamlit dashboard."""
    import subprocess
    ui_path = Path(__file__).parent / "ui" / "dashboard.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(ui_path)], check=True)


# ── Entry ─────────────────────────────────────────────────────

if __name__ == "__main__":
    # If run directly with arguments → CLI mode
    # If run via streamlit → UI mode
    if len(sys.argv) > 1 and sys.argv[1] not in ["run"]:
        run_cli()
    else:
        # Default: launch Streamlit
        run_streamlit()
