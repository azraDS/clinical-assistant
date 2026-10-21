"""Paths and settings shared across the package.

Data and outputs live under ``notebooks/`` (guideline PDF, docling/pypdf/LLM caches,
review sheets) so the package and the research notebook share the same on-disk caches:
running the app costs no extra LLM calls for anything the notebook has already computed.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
DATA_DIR = NOTEBOOK_DIR / "data"
OUTPUT_DIR = NOTEBOOK_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GUIDELINE_PATH = DATA_DIR / "guidelines" / "kdigo_2024_ckd.pdf"
DOCLING_CACHE = DATA_DIR / "cache" / "kdigo_2024_docling.json"
PYPDF_CACHE = DATA_DIR / "cache" / "kdigo_2024_pypdf.txt"
STATEMENTS_CACHE = DATA_DIR / "cache" / "statements_compiled.json"

LLM_MODEL = "gpt-4.1"
LLM_CACHE_PATH = DATA_DIR / "cache" / f"llm_cache_{LLM_MODEL}.json"

EMBEDDING_BACKEND = "local"
LOCAL_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

RERANKERS = {
    "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
    "medcpt": "ncbi/MedCPT-Cross-Encoder",
}
DEFAULT_RERANKER = "bge-reranker-v2-m3"

PIPELINE_STRATEGY = "llm_based_retrieval_plus_query_reformulation"
PIPELINE_K = 25

# Tags every Langfuse trace this pipeline produces, so later prompt/pipeline changes can be
# compared against this one in the Langfuse UI. Bump when the pipeline changes materially.
PIPELINE_VERSION = "v1"


def load_env() -> None:
    """Load the OpenAI key etc. from .env at the repo root, falling back to notebooks/.env.
    Real environment variables always win (dotenv never overrides them)."""
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(NOTEBOOK_DIR / ".env")


_langfuse_client = None
_langfuse_init_done = False


def get_langfuse():
    """Lazily build the Langfuse client, once per process. Returns None (never raises) if the
    SDK isn't installed or LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY aren't set — every call site
    that uses this must treat None as "tracing disabled" and run normally. Tracing must never be
    able to break the pipeline.

    Reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL (or the legacy
    LANGFUSE_HOST) from notebooks/.env. Traces are tagged environment="local" unless
    LANGFUSE_TRACING_ENVIRONMENT is already set in the environment.
    """
    global _langfuse_client, _langfuse_init_done
    if _langfuse_init_done:
        return _langfuse_client
    _langfuse_init_done = True
    load_env()
    os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", "local")
    if not os.environ.get("LANGFUSE_PUBLIC_KEY") or not os.environ.get("LANGFUSE_SECRET_KEY"):
        return None
    try:
        from langfuse import get_client

        _langfuse_client = get_client()
    except Exception:
        _langfuse_client = None
    return _langfuse_client


def add_trace_metadata(**kwargs) -> None:
    """Attach key-value metadata to whichever span/trace is currently active — a no-op if
    Langfuse isn't configured. Use this for things worth finding later (e.g. an invalid-citation
    count) that should show up as a clean, filterable field rather than buried in a function's
    auto-captured output."""
    langfuse = get_langfuse()
    if langfuse is not None:
        langfuse.update_current_span(metadata=kwargs)


def set_trace_input(value) -> None:
    """Replace the auto-captured input on whichever span is currently active — a no-op if
    Langfuse isn't configured. Use this (with @observe(..., capture_input=False)) on any
    function whose own arguments include something huge/static (e.g. the whole KnowledgeBase)
    that @observe would otherwise serialize in full on every call."""
    langfuse = get_langfuse()
    if langfuse is not None:
        langfuse.update_current_span(input=value)
