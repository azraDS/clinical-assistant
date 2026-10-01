"""The online pipeline, end to end: record + visit note -> PatientState -> filter -> retrieval
-> generation -> provenance-checked care review.

This is a fixed sequence of typed steps (deterministic code + two structured-output LLM calls:
sub-query/screening and final generation), not a tool-calling agent — nothing here needs the LLM
to decide *which* step runs next, so a dynamic agent loop would add cost and a new failure mode
for no benefit. See notebooks/README.md §6 for the diagram this mirrors.
"""

from dataclasses import dataclass

from langfuse import observe

from . import config
from .eligibility import label_statement_eligibility
from .generation import attach_provenance, build_evidence_package, generate_care_review, render_report
from .knowledge_base import KnowledgeBase, load_knowledge_base
from .patient import PatientState, build_patient_state
from .retrieval import RetrievalIndex


@dataclass
class CareReviewResult:
    patient_label: str
    patient_state: PatientState
    eligibility_labels: dict  # from label_statement_eligibility — reuse this, don't recompute
    evidence_package: dict
    result: dict  # summary, recommendations (with sources), data_to_obtain, not_covered, invalid_citations
    report_markdown: str


class CarePipeline:
    """Owns the (expensive-to-build) knowledge base and retrieval index. Build once, call
    .run() per patient. In Streamlit, construct this inside an @st.cache_resource function."""

    def __init__(self, use_llm: bool = True, rebuild_knowledge_base: bool = False):
        config.load_env()
        self.use_llm = use_llm
        self.kb: KnowledgeBase = load_knowledge_base(use_llm=use_llm, rebuild=rebuild_knowledge_base)
        self.index = RetrievalIndex(self.kb)

    @observe(name="care_pipeline.run", as_type="chain")
    def run(
        self,
        record: dict,
        note: str,
        patient_label: str = "patient",
        strategy: str = config.PIPELINE_STRATEGY,
        k: int = config.PIPELINE_K,
        save: bool = False,
    ) -> CareReviewResult:
        """Every per-patient LLM step (note extraction, sub-queries, screening, generation) is a
        live call traced automatically by langfuse.openai's client wrapper — nothing on this path
        is cached. Only the
        offline guideline-metadata build (`build_knowledge_base`) is cached, and that isn't part
        of this call at all (it already ran in __init__).

        The whole call is one Langfuse trace (root span "care_pipeline.run", as_type="chain");
        every step below creates a nested span/generation/retriever/evaluator inside it — input
        and output are auto-captured from this method's own arguments and return value (no
        manual span code needed), so the full trace shows patient state extraction, the
        eligibility filter, retrieval/reranking, LLM screening, generation and provenance
        checking as one tree with per-step latency and inputs/outputs. A no-op if Langfuse isn't
        configured (see config.get_langfuse). Not wrapped with an explicit flush() — this is
        meant to run inside a long-lived process (the Streamlit app), where Langfuse's normal
        background batching flushes on its own; a short-lived script should call
        `config.get_langfuse().flush()` itself before exiting.
        """
        st = build_patient_state(record, note, use_llm=self.use_llm)
        eligibility_labels = label_statement_eligibility(self.kb, st)
        pkg = build_evidence_package(
            self.kb, self.index, st, note, strategy=strategy, k=k, eligibility_labels=eligibility_labels
        )
        review = generate_care_review(note, pkg)
        result = attach_provenance(self.kb, review, pkg)
        # Also on the root trace (not just attach_provenance's own span), so it's visible without
        # drilling into a specific trace — e.g. filtering the Langfuse traces list for n_invalid_citations > 0.
        config.add_trace_metadata(
            invalid_citations=result["invalid_citations"], n_invalid_citations=len(result["invalid_citations"])
        )
        report = render_report(patient_label, note, result)

        if save:
            import json

            (config.OUTPUT_DIR / f"care_review_{patient_label}.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False)
            )
            (config.OUTPUT_DIR / f"care_review_{patient_label}.md").write_text(report)

        return CareReviewResult(
            patient_label=patient_label,
            patient_state=st,
            eligibility_labels=eligibility_labels,
            evidence_package=pkg,
            result=result,
            report_markdown=report,
        )
