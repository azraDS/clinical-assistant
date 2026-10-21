"""Regex-only applicability + the eligibility filter — deterministic, no LLM calls."""

import pytest

from clinical_assistant import config
from clinical_assistant import sample_patients as sp
from clinical_assistant.eligibility import PREGNANCY_AVOID, label_statement_eligibility
from clinical_assistant.knowledge_base import load_knowledge_base
from clinical_assistant.patient import build_patient_state

pytestmark = pytest.mark.skipif(
    not config.DOCLING_CACHE.exists(),
    reason="docling cache not built yet (run notebooks/02_retrieval_pipeline.ipynb once)",
)


@pytest.fixture(scope="module")
def kb():
    return load_knowledge_base(use_llm=False)


def test_sglt2i_statement_population_and_egfr(kb):
    s = kb.stmt_by_id["Rec 3.7.2"]
    assert s["applicability"]["population"] == "adult"
    assert s["applicability"]["egfr_range"] == (20.0, float("inf"))
    # manual override: "ACR >=200, or heart failure" -> the ACR branch is not a hard requirement
    assert s["applicability"]["acr_range"] is None


def test_pediatric_statement_excluded_for_adult(kb):
    st = build_patient_state(sp.RECORDS["P001"], sp.NOTES["P001"], use_llm=False)
    filtered = label_statement_eligibility(kb, st)
    pediatric_only = [sid for sid, s in kb.stmt_by_id.items() if s["applicability"]["population"] == "child"]
    assert pediatric_only  # sanity: the guideline does have pediatric-only statements
    assert all(filtered[sid][0] == "excluded" for sid in pediatric_only)


def test_pregnant_patient_excludes_rasi(kb):
    st = build_patient_state(sp.RECORDS["P009"], sp.NOTES["P009"], use_llm=False)
    assert st.pregnant is True
    filtered = label_statement_eligibility(kb, st)
    rasi_statements = [
        sid for sid, s in kb.stmt_by_id.items() if set(s["applicability"]["drug_classes"]) & PREGNANCY_AVOID
    ]
    assert rasi_statements
    assert all(filtered[sid][0] == "excluded" for sid in rasi_statements)
