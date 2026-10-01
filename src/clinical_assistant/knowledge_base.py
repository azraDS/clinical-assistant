"""Build (or load) the compiled set of guideline statements: parsed + both applicability
opinions attached. This is the expensive one-time-per-guideline-version step (docling parse,
LLM applicability batch over ~160 statements); the result is cached to disk so the app and
tests start instantly after the first build. See notebooks/README.md §1-2.
"""

import json
from dataclasses import dataclass, field

from langfuse import observe

from . import config
from .applicability import annotate_statements_with_llm_metadata, apply_manual_overrides, extract_applicability
from .parsing import parse_statements


@dataclass
class KnowledgeBase:
    statements: list[dict]
    table_units: dict[str, dict]
    stmt_by_id: dict[str, dict] = field(init=False)
    corpus_ids: list[str] = field(init=False)

    def __post_init__(self):
        self.stmt_by_id = {s["statement_id"]: s for s in self.statements}
        self.corpus_ids = [s["statement_id"] for s in self.statements]


_RANGE_FIELDS = ("egfr_range", "acr_range", "age_range")


def _restore_range_tuples(statements: list[dict]) -> list[dict]:
    # JSON has no tuple type, so a cache round-trip turns (lo, hi) into [lo, hi]; restore tuples
    # so a cache-loaded KnowledgeBase has the exact same shapes as a freshly built one.
    for s in statements:
        for applic_key in ("applicability", "applicability_llm"):
            meta = s.get(applic_key)
            if not meta:
                continue
            for field_name in _RANGE_FIELDS:
                rng = meta.get(field_name)
                if rng is not None:
                    meta[field_name] = tuple(rng)
    return statements


@observe(name="build_knowledge_base", as_type="chain")
def build_knowledge_base(use_llm: bool) -> KnowledgeBase:
    """Parse the guideline PDF into statement units and attach applicability metadata to each —
    the offline document-processing pipeline (not per-patient inference). One Langfuse trace
    wraps this whole step, so the ~160 nested `extract_statement_applicability_metadata`
    generations (when use_llm=True and the cache doesn't already have them) show up as one
    navigable tree instead of ~160 unrelated top-level traces."""
    statements, table_units = parse_statements()
    for s in statements:
        s["applicability"] = extract_applicability(s)
    apply_manual_overrides(statements)
    if use_llm:
        annotate_statements_with_llm_metadata(statements)
    return KnowledgeBase(statements=statements, table_units=table_units)


def load_knowledge_base(use_llm: bool = True, rebuild: bool = False) -> KnowledgeBase:
    """Load the compiled knowledge base from disk, building (and caching) it if needed.

    use_llm=True attaches the LLM applicability second opinion (recommended — see
    notebooks/README.md §2 for why regex alone or LLM alone both hide statements patients need).
    Every LLM call in `build_knowledge_base` is itself cached (llm_calls.py's `cached_llm_call`),
    so a rebuild with an unchanged prompt is free.
    """
    if not rebuild and config.STATEMENTS_CACHE.exists():
        cached = json.loads(config.STATEMENTS_CACHE.read_text())
        if cached.get("use_llm") == use_llm:
            return KnowledgeBase(
                statements=_restore_range_tuples(cached["statements"]), table_units=cached["table_units"]
            )

    kb = build_knowledge_base(use_llm)
    config.STATEMENTS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    config.STATEMENTS_CACHE.write_text(
        json.dumps({"use_llm": use_llm, "statements": kb.statements, "table_units": kb.table_units}, ensure_ascii=False)
    )
    return kb
