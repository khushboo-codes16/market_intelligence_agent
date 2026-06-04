# tests/conftest.py
# ============================================================
# Pytest configuration: mock heavy dependencies at collection
# time so unit tests don't need chromadb, sentence_transformers,
# groq, or streamlit installed to run.
# ============================================================

import sys
from unittest.mock import MagicMock

# ── Stub out heavy optional dependencies ─────────────────────
# These are only needed for integration/RAG tests, not unit tests.

_STUBS = [
    "chromadb",
    "chromadb.config",
    "sentence_transformers",
    "groq",
    "langgraph",
    "langgraph.graph",
    "langchain",
    "langchain_groq",
    "langchain_community",
    "streamlit",
    "streamlit_extras",
    "trafilatura",
    "plotly",
    "plotly.express",
    "plotly.graph_objects",
    "tiktoken",
]

for _mod in _STUBS:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# Ensure chromadb.config.Settings is a callable returning a mock
sys.modules["chromadb.config"].Settings = MagicMock(return_value=MagicMock())

# langgraph.graph needs END and StateGraph
_langgraph_graph = sys.modules["langgraph.graph"]
_langgraph_graph.END = "END"
_langgraph_graph.StateGraph = MagicMock()
