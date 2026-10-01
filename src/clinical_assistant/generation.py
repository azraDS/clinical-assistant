"""Evidence package -> LLM care review -> code-attached provenance. The generator may only cite
statement_ids present in its evidence package; every citation is checked against the parsed
guideline in code, not trusted to the LLM. See notebooks/README.md §6.
Ported from notebooks/02_retrieval_pipeline.ipynb §13-14.
"""

import json
from collections import defaultdict
from typing import Literal, Optional

from langfuse import observe
from pydantic import BaseModel, Field

from . import config
from .eligibility import label_statement_eligibility
from .knowledge_base import KnowledgeBase
from .patient import PatientState, derive
from .retrieval import RetrievalIndex


@observe(name="build_evidence_package", as_type="span", capture_input=False)
def build_evidence_package(
    kb: KnowledgeBase,
    index: RetrievalIndex,
    st: PatientState,
    note: str,
    strategy: str = config.PIPELINE_STRATEGY,
    k: int = config.PIPELINE_K,
    rationale_chars: int = 600,
    eligibility_labels: dict | None = None,
) -> dict:
    # capture_input=False: kb and index would otherwise get auto-captured in full (the whole
    # knowledge base + BM25/embedding index) — set a small, useful input by hand instead.
    config.set_trace_input({"strategy": strategy, "k": k, "patient_state": st.model_dump(), "note": note})
    # Pass eligibility_labels if the caller (CarePipeline.run) already computed it — avoids a
    # second, redundant (and, worse, un-nested/orphan-traced if called outside an active span) call.
    eligibility_labels = eligibility_labels if eligibility_labels is not None else label_statement_eligibility(kb, st)
    retrieved = index.retrieve(st, note, strategy, k=k, eligibility_labels=eligibility_labels)
    by_domain = defaultdict(list)
    for sid, _score in retrieved:
        s = kb.stmt_by_id[sid]
        status, _, needs = eligibility_labels[sid]
        by_domain[s["domain"]].append(
            {
                "statement_id": sid,
                "grade": s["grade"],
                "page": s["page"],
                "status": status,
                "needs_data": needs,
                "heading_path": s["heading_path"],
                "text": s["text"],
                "rationale_excerpt": s["section_rationale"][:rationale_chars],
                "linked_tables": [kb.table_units[t] for t in s["table_refs"] if t in kb.table_units],
            }
        )
    return {
        "patient_state": st.model_dump(),
        "derived": derive(st),
        "evidence_by_domain": dict(by_domain),
    }


def format_evidence_package(pkg: dict) -> str:
    lines = [f"PATIENT: {json.dumps(pkg['patient_state'])}", f"DERIVED: {json.dumps(pkg['derived'])}", ""]
    for domain, items_ in pkg["evidence_by_domain"].items():
        lines.append(f"## {domain}")
        for e in items_:
            flag = f" [CONDITIONAL — needs {', '.join(e['needs_data'])}]" if e["needs_data"] else ""
            lines.append(f"- {e['statement_id']} (grade {e['grade'] or 'practice point'}, p.{e['page']}){flag}\n  {e['text']}")
            if e["rationale_excerpt"]:
                lines.append(f"  [section context] {e['rationale_excerpt']}")
            for t in e["linked_tables"]:
                lines.append(f"  [{t['table_id']}] {t['caption']}\n{t['markdown'][:1000]}")
        lines.append("")
    return "\n".join(lines)


class Recommendation(BaseModel):
    action: str = Field(description="What the clinician should consider for THIS patient, one concrete action")
    why_this_patient: str = Field(description="The patient facts (from record or note) that make it apply")
    statement_ids: list[str] = Field(description="IDs of the guideline statements it is based on — only IDs from the evidence package")
    applies: Literal["applies", "conditional"] = Field(description="'conditional' if it depends on data that is missing")
    missing_data: list[str] = Field(default_factory=list, description="Data needed before acting, if conditional")
    caution: Optional[str] = Field(None, description="Safety caveat stated in the evidence, if any")


class CareReview(BaseModel):
    summary: str = Field(description="2-3 sentences: the main issues for this visit")
    recommendations: list[Recommendation] = Field(description="Most important first")
    data_to_obtain: list[str] = Field(description="Tests or information to obtain")
    not_covered: str = Field(description="Questions from the note that the evidence package does not answer, or 'none'")


GENERATION_SYSTEM = """You are a clinical evidence assistant. You compare ONE synthetic patient with guideline statements
(KDIGO 2024 CKD) that were retrieved and filtered for this patient. Output is for clinician review and research only.

Rules:
1. Use only the patient information and the statements in the evidence package. Do not invent patient facts or guideline content.
2. Every recommendation must cite the statement_ids it is based on, exactly as written in the package.
3. Match the strength of the wording to the grade: grade 1 -> "the guideline recommends"; grade 2 -> "the guideline
   suggests / consider"; practice point -> "expert practice point".
4. Statements marked CONDITIONAL depend on missing data: say what data is needed; do not present them as definite.
5. Only include statements that matter for this visit. Merge statements that support the same action into one recommendation.
6. Put the clinician's own questions from the note first. Then routine CKD care that is missing (monitoring, CV risk,
   kidney failure risk), if the evidence supports it.
7. If the evidence does not answer something in the note, say so in not_covered instead of guessing."""


def generate_care_review(note: str, pkg: dict) -> CareReview:
    # langfuse.openai's client wrapper auto-traces this call (full prompt, output, token usage).
    from langfuse.openai import OpenAI

    prompt = (
        f"VISIT NOTE\n{note}\n\nEVIDENCE PACKAGE\n{format_evidence_package(pkg)}\n\n"
        "TASK\nWrite the care review for this visit."
    )
    resp = OpenAI(max_retries=10).responses.parse(
        model=config.LLM_MODEL,
        temperature=0,
        input=[{"role": "system", "content": GENERATION_SYSTEM}, {"role": "user", "content": prompt}],
        text_format=CareReview,
        name="generate_care_review",
    )
    return resp.output_parsed


STRENGTH = {"1": "strong recommendation", "2": "weak recommendation (suggestion)"}


def source_of(kb: KnowledgeBase, sid: str) -> dict:
    s = kb.stmt_by_id[sid]
    grade = s["grade"]
    return {
        "statement_id": sid,
        "grade": grade,
        "strength": STRENGTH[grade[0]] if grade else "practice point (expert advice, ungraded)",
        "evidence_certainty": {"A": "high", "B": "moderate", "C": "low", "D": "very low"}[grade[1]] if grade else None,
        "page": s["page"],
        "heading_path": s["heading_path"],
        "text": s["text"],
    }


@observe(name="attach_provenance", as_type="evaluator", capture_input=False)
def attach_provenance(kb: KnowledgeBase, review: CareReview, pkg: dict) -> dict:
    """Code, not the LLM: keep only citations that are in the evidence package and attach the verbatim source.

    capture_input=False: kb and pkg would otherwise get auto-captured in full (the whole
    knowledge base, and the evidence package already visible in build_evidence_package's output)
    — the genuinely useful thing to see here is the LLM's own raw output, before this function
    checks it."""
    config.set_trace_input({"review_before_provenance_check": review.model_dump()})
    in_package = {e["statement_id"] for items_ in pkg["evidence_by_domain"].values() for e in items_}
    out, invalid = [], []
    for rec in review.recommendations:
        valid = [sid for sid in rec.statement_ids if sid in in_package]
        invalid += [sid for sid in rec.statement_ids if sid not in in_package]
        out.append(
            {
                **rec.model_dump(),
                "statement_ids": valid,
                "sources": [source_of(kb, sid) for sid in valid],
                "unsupported": not valid,
            }
        )
    # Always set this metadata, even when invalid is empty — so a trace search can distinguish
    # "checked, 0 invalid" from a step that silently didn't run.
    config.add_trace_metadata(invalid_citations=invalid, n_invalid_citations=len(invalid))
    return {
        "summary": review.summary,
        "recommendations": out,
        "data_to_obtain": review.data_to_obtain,
        "not_covered": review.not_covered,
        "invalid_citations": invalid,
        "retrieved_ids": sorted(in_package),
    }


def render_report(patient_label: str, note: str, result: dict) -> str:
    lines = [
        f"# Care review — {patient_label} (synthetic, for clinician review only)",
        "",
        f"**Visit note:** {note}",
        "",
        result["summary"],
        "",
    ]
    for n, r in enumerate(result["recommendations"], 1):
        tag = "" if r["applies"] == "applies" else f" — *conditional: needs {', '.join(r['missing_data']) or 'more data'}*"
        lines.append(f"### {n}. {r['action']}{tag}")
        lines.append(f"*Why this patient:* {r['why_this_patient']}")
        if r["caution"]:
            lines.append(f"*Caution:* {r['caution']}")
        if r["unsupported"]:
            lines.append("⚠️ **No valid citation — not supported by the retrieved guideline text.**")
        for src in r["sources"]:
            lines.append(
                f"> **{src['statement_id']}** ({src['strength']}"
                + (f", {src['evidence_certainty']}-certainty evidence" if src["evidence_certainty"] else "")
                + f"; KDIGO 2024 p.{src['page']}; {src['heading_path']})  \n> {src['text']}"
            )
        lines.append("")
    if result["data_to_obtain"]:
        lines += ["**Data to obtain:** " + "; ".join(result["data_to_obtain"]), ""]
    lines += [f"**Not covered by the retrieved evidence:** {result['not_covered']}", ""]
    if result["invalid_citations"]:
        lines.append(f"⚠️ Removed citations not in the evidence package: {', '.join(result['invalid_citations'])}")
    return "\n".join(lines)
