"""Uses the on-disk docling/pypdf caches under notebooks/data/cache — no network, no LLM calls.
Skipped if the guideline hasn't been parsed yet (first run needs the PDF + docling installed)."""

import pytest

from clinical_assistant import config
from clinical_assistant.parsing import parse_statements

pytestmark = pytest.mark.skipif(
    not config.DOCLING_CACHE.exists(),
    reason="docling cache not built yet (run notebooks/02_retrieval_pipeline.ipynb once)",
)


def test_statement_count_and_types():
    statements, table_units = parse_statements()
    assert len(statements) == 169
    assert sum(s["type"] == "recommendation" for s in statements) == 28
    assert sum(s["type"] == "practice_point" for s in statements) == 141
    assert len(table_units) == 85


def test_heading_path_keeps_every_numbered_ancestor():
    statements, _ = parse_statements()
    by_id = {s["statement_id"]: s for s in statements}
    assert by_id["Rec 3.15.1.1"]["heading_path"] == (
        "Chapter 3: Delaying CKD progression and managing its complications › "
        "3.15 Cardiovascular disease (CVD) and additional specific interventions to modify risk › "
        "3.15.1 Lipid management"
    )
    # every statement has a chapter, and every statement's own leaf heading appears in the path
    assert all(s["chapter"] for s in statements)


def test_subheading_appends_after_numbered_ancestors():
    statements, _ = parse_statements()
    by_id = {s["statement_id"]: s for s in statements}
    s = by_id["PP 1.2.2.2"]
    assert s["heading_path"].endswith("1.2.2 Guidance to physicians and other healthcare providers › Key information")
