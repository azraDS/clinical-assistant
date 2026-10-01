"""Eligibility filter: for one patient, is each statement excluded, conditional (depends on
missing data) or eligible? Consensus of regex + LLM applicability metadata — a statement is
excluded only if BOTH opinions exclude it. See notebooks/README.md §2.
Ported from notebooks/02_retrieval_pipeline.ipynb §8.
"""

from langfuse import observe

from . import config
from .knowledge_base import KnowledgeBase
from .patient import PatientState, derive

# Patient-side safety rule: the guideline text never says these drug classes are contraindicated
# in pregnancy, so no metadata extraction (regex or LLM) can find it from the statement text alone.
# Uses the drug_classes already extracted from each statement. (Drug list to be confirmed by a
# clinician — see notebooks/README.md gap #13.)
PREGNANCY_AVOID = {"RASi", "SGLT2i", "MRA", "statin", "GLP-1 RA"}


def in_range(value, rng):
    lo, hi = rng
    return lo <= value < hi


def check_applicability(meta: dict, st: PatientState, d: dict) -> tuple[str, list[str], list[str]]:
    reasons, needs = [], []

    if meta["population"] == "child" and d["is_adult"]:
        reasons.append("pediatric-only statement; patient is adult")
    if meta["population"] == "adult" and not d["is_adult"]:
        reasons.append("adult-only statement; patient is a child")

    if meta["egfr_range"]:
        if st.egfr is None:
            needs.append("eGFR")
        elif not in_range(st.egfr, meta["egfr_range"]):
            reasons.append(f"eGFR {st.egfr} outside {meta['egfr_range']}")

    if meta["acr_range"]:
        if st.acr_mg_g is None:
            needs.append("urine ACR")
        elif not in_range(st.acr_mg_g, meta["acr_range"]):
            reasons.append(f"ACR {st.acr_mg_g} outside {meta['acr_range']}")

    if meta["age_range"] and st.age is not None and not in_range(st.age, meta["age_range"]):
        reasons.append(f"age {st.age} outside {meta['age_range']}")

    if meta["diabetes"] == "required":
        if st.diabetes == "none":
            reasons.append("requires diabetes")
        elif st.diabetes == "unknown":
            needs.append("diabetes status")
    if meta["diabetes"] == "excluded" and st.diabetes in ("type_1", "type_2"):
        reasons.append("statement is for people without diabetes")

    on_krt = bool(st.on_dialysis or st.kidney_transplant)
    if meta["excludes_krt"] and on_krt:
        reasons.append("not for dialysis/transplant")
    if meta["requires_krt"] and not on_krt:
        reasons.append("only for dialysis/transplant")

    if meta["mentions_pregnancy"] and st.sex == "M":
        reasons.append("pregnancy-specific")

    blocked = set(meta.get("drug_classes", [])) & PREGNANCY_AVOID
    if st.pregnant and blocked:
        reasons.append(f"pregnant: {', '.join(sorted(blocked))} generally avoided in pregnancy (see PP 4.1.4)")

    status = "excluded" if reasons else "conditional" if needs else "eligible"
    return status, reasons, needs


@observe(name="label_statement_eligibility", as_type="evaluator", capture_input=False)
def label_statement_eligibility(kb: KnowledgeBase, st: PatientState) -> dict[str, tuple[str, list[str], list[str]]]:
    """Labels every statement excluded/conditional/eligible for this patient — it does not itself
    remove anything; callers drop "excluded" statements from their own candidate set. Consensus of
    regex and LLM metadata: excluded only if both exclude; otherwise kept, with the union of
    missing data from the opinions that keep it.

    capture_input=False: `kb` would otherwise get auto-captured in full on every call (the whole
    169-statement knowledge base, megabytes of static text nobody needs to see per-patient) — set
    a small, useful input by hand instead."""
    config.set_trace_input({"patient_state": st.model_dump()})
    d = derive(st)
    out = {}
    for s in kb.statements:
        opinions = [check_applicability(s["applicability"], st, d)]
        if "applicability_llm" in s:
            opinions.append(check_applicability(s["applicability_llm"], st, d))
        kept = [o for o in opinions if o[0] != "excluded"]
        if not kept:
            reasons = list(dict.fromkeys(r for o in opinions for r in o[1]))
            out[s["statement_id"]] = ("excluded", reasons, [])
        else:
            needs = list(dict.fromkeys(n for o in kept for n in o[2]))
            out[s["statement_id"]] = ("conditional" if needs else "eligible", [], needs)
    return out
