"""Full pipeline, end to end, on one sample patient. Needs OPENAI_API_KEY (notebooks/.env) and
hits the shared LLM cache — free if the notebook or app has already run this patient."""

import os

import pytest

from clinical_assistant import config
from clinical_assistant import sample_patients as sp
from clinical_assistant.pipeline import CarePipeline

_has_key = os.environ.get("OPENAI_API_KEY") or (config.NOTEBOOK_DIR / ".env").exists()
pytestmark = pytest.mark.skipif(
    not (config.DOCLING_CACHE.exists() and _has_key),
    reason="needs the docling cache and an OpenAI API key",
)


def test_run_pipeline_on_p001_has_no_invalid_citations():
    pipe = CarePipeline(use_llm=True)
    res = pipe.run(sp.RECORDS["P001"], sp.NOTES["P001"], patient_label="P001")
    assert res.result["invalid_citations"] == []
    assert len(res.result["recommendations"]) > 0
    assert all(r["sources"] for r in res.result["recommendations"] if not r["unsupported"])
