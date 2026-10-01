"""Statement-level applicability metadata: conditions a patient must meet for a statement to
apply (population, eGFR/ACR/age range, diabetes, dialysis/transplant, drug classes).

Two independent extractors (regex + LLM) are combined by consensus in eligibility.py: a
statement is excluded for a patient only if BOTH sources exclude it. See notebooks/README.md §2.
Ported from notebooks/02_retrieval_pipeline.ipynb §5.
"""

import hashlib
import re
from typing import Literal, Optional

from pydantic import BaseModel, Field

from . import config
from .kdigo_lookups import A_RANGES, G_RANGES, INF
from .llm_calls import LLMResult, cached_llm_call, extract_token_usage

# ---------------------------------------------------------------------------
# Regex extractor
# ---------------------------------------------------------------------------

CHILD_RE = re.compile(
    r"\b(child(?![- ]bearing)|children|pediatric|paediatric|infant|neonat\w*|adolescen\w*|young people)\b", re.I
)
ADULT_RE = re.compile(r"\badults?\b(?!-oriented| care| provider| clinic)", re.I)
G_RANGE_RE = re.compile(r"\bG(1|2|3a|3b|3|4|5)\s*[-–]\s*G?(1|2|3a|3b|3|4|5)\b")
G_SINGLE_RE = re.compile(r"\b(?:CKD|categor(?:y|ies))\s+G(1|2|3a|3b|3|4|5)\b(?!\s*[-–])")
EGFR_OP_RE = re.compile(r"\beGFR\w*\s*(?:of\s*|level\s*)?([<>≥≤])\s*(\d+)")
EGFR_BETWEEN_RE = re.compile(r"\beGFR\w*\s*(?:of\s*)?(\d+)\s*[-–]\s*(\d+)")
ACR_OP_RE = re.compile(r"([<>≥≤])\s*(\d+)\s*mg/g")
A_RANGE_RE = re.compile(r"\bA([123])\s*[-–]\s*A?([123])\b|\b[G\d]\w*\s*[-–]?\s*\w*,\s*A([123])\b")
AGE_OP_RE = re.compile(r"\baged?\s*([<>≥≤])\s*(\d+)")
AGE_BETWEEN_RE = re.compile(r"\baged?\s*(\d+)\s*[-–]\s*(\d+)")
DIABETES_REQ_RE = re.compile(r"\b(?:with|and)\s+(?:type 2 diabetes|T2D|diabetes)\b", re.I)
DIABETES_EXCL_RE = re.compile(r"\bwithout diabetes\b", re.I)
EXCLUDES_KRT_RE = re.compile(r"not (?:treated with|receiving|on) (?:chronic )?(?:dialysis|KRT)", re.I)
REQUIRES_KRT_RE = re.compile(
    r"\b(?:receiving|on|treated with) (?:chronic |maintenance )?(?:dialysis|KRT)\b|\bkidney transplant recipients\b", re.I
)

DRUG_CLASS_PATTERNS = {
    "RASi": r"\bRASi\b|\bACEi\b|\bARBs?\b|renin[- ]angiotensin|angiotensin",
    "SGLT2i": r"SGLT2",
    "MRA": r"mineralocorticoid|finerenone|\bMRA\b",
    "GLP-1 RA": r"GLP-1",
    "statin": r"statin",
    "metformin": r"metformin",
    "NSAID": r"NSAID|nonsteroidal anti-inflammatory",
    "potassium binder": r"potassium[- ]binder|potassium exchange",
    "urate-lowering": r"urate-lowering|allopurinol|febuxostat",
    "anticoagulant": r"anticoagula|DOAC|warfarin|NOAC",
    "bicarbonate": r"bicarbonate",
}


def _interval_from_op(op, value):
    value = float(value)
    return {"<": (0, value), "≤": (0, value + 1e-9), ">": (value + 1e-9, INF), "≥": (value, INF)}[op]


def _hull(intervals):
    if not intervals:
        return None
    return (min(lo for lo, _ in intervals), max(hi for _, hi in intervals))


def extract_applicability(s: dict) -> dict:
    text = s["text"]
    context = f"{s['heading_path']} {text}"
    meta = {}

    child = bool(CHILD_RE.search(context))
    adult = bool(ADULT_RE.search(text))
    meta["population"] = "both" if child and adult else "child" if child else "adult" if adult else "any"

    egfr = []
    for a, b in G_RANGE_RE.findall(text):
        lo_a, hi_a = G_RANGES[f"G{a}"]
        lo_b, hi_b = G_RANGES[f"G{b}"]
        egfr.append((min(lo_a, lo_b), max(hi_a, hi_b)))
    for g in G_SINGLE_RE.findall(text):
        egfr.append(G_RANGES[f"G{g}"])
    egfr += [_interval_from_op(op, v) for op, v in EGFR_OP_RE.findall(text)]
    egfr += [(float(a), float(b) + 1) for a, b in EGFR_BETWEEN_RE.findall(text)]
    meta["egfr_range"] = _hull(egfr)

    acr = [_interval_from_op(op, v) for op, v in ACR_OP_RE.findall(text)]
    for a, b, single in A_RANGE_RE.findall(text):
        if single:
            acr.append(A_RANGES[f"A{single}"])
        else:
            acr.append((A_RANGES[f"A{a}"][0], A_RANGES[f"A{b}"][1]))
    if re.search(r"severely increased albuminuria", text, re.I):
        acr.append(A_RANGES["A3"])
    if re.search(r"moderately increased albuminuria", text, re.I):
        acr.append(A_RANGES["A2"])
    meta["acr_range"] = _hull(acr)

    age = [_interval_from_op(op, v) for op, v in AGE_OP_RE.findall(text)]
    age += [(float(a), float(b) + 1) for a, b in AGE_BETWEEN_RE.findall(text)]
    meta["age_range"] = _hull(age)

    meta["diabetes"] = (
        "excluded" if DIABETES_EXCL_RE.search(text) else "required" if DIABETES_REQ_RE.search(text) else None
    )
    meta["excludes_krt"] = bool(EXCLUDES_KRT_RE.search(text))
    meta["requires_krt"] = bool(REQUIRES_KRT_RE.search(text)) and not meta["excludes_krt"]
    meta["mentions_heart_failure"] = bool(re.search(r"heart failure", text, re.I))
    meta["mentions_pregnancy"] = bool(re.search(r"pregnan", text, re.I))
    meta["drug_classes"] = [k for k, pat in DRUG_CLASS_PATTERNS.items() if re.search(pat, text, re.I)]
    meta["metadata_source"] = "regex"
    return meta


# Statements where the regex/LLM consensus still gets it wrong (found by the §11 filter-error
# check): "A or B" criteria that the filter can only express as AND, so the constraint is
# dropped rather than wrongly excluding people who qualify through the other branch.
MANUAL_OVERRIDES = {
    # Transition-of-care statements are about adolescents / young adults, not all adults.
    # (Illustrative — to be confirmed by clinical review.)
    **{sid: {"population": "child"} for sid in ["PP 5.3.1.1.1", "PP 5.3.1.1.2", "PP 5.3.1.1.3"]},
    **{sid: {"age_range": (0, 26)} for sid in ["PP 5.3.1.2.1", "PP 5.3.1.2.2", "PP 5.3.1.2.3"]},
    "Rec 3.7.2": {"acr_range": None},  # "... ACR >=200, or heart failure irrespective of albuminuria"
    "PP 5.2.3.2": {"egfr_range": None, "age_range": None},  # "G4-G5, aged >65, poor growth, or symptoms..."
}


def apply_manual_overrides(statements: list[dict]) -> None:
    for s in statements:
        if s["statement_id"] in MANUAL_OVERRIDES:
            s["applicability"].update(MANUAL_OVERRIDES[s["statement_id"]])
            s["applicability"]["metadata_source"] = "manual"


# ---------------------------------------------------------------------------
# LLM extractor (second opinion) + consensus merge
# ---------------------------------------------------------------------------


class LLMApplicability(BaseModel):
    # analysis comes first so the model reasons before filling fields
    analysis: str = Field(
        description="1-3 sentences: who must the patient be for this statement to apply? "
        "Note any 'or' alternatives and any thresholds that are only about timing/action."
    )
    has_or_alternatives: bool = Field(
        description="True if the statement lists alternative qualifying criteria (any one is enough)"
    )
    alternative_only_criteria: list[
        Literal["egfr", "acr", "age", "diabetes", "heart_failure", "dialysis", "kidney_transplant"]
    ] = Field([], description="Criteria that appear inside only ONE of several 'or' alternatives, so a patient can qualify without them")
    population: Literal["adult", "child", "both", "any"]
    population_quote: Optional[str] = Field(None, description="Exact span from the text supporting population, or null")
    egfr_min: Optional[float] = Field(None, description="Inclusive lower eGFR bound, ml/min/1.73m2")
    egfr_max: Optional[float] = Field(None, description="Exclusive upper eGFR bound, ml/min/1.73m2")
    egfr_quote: Optional[str] = None
    acr_min: Optional[float] = Field(None, description="Inclusive lower urine ACR bound, mg/g")
    acr_max: Optional[float] = Field(None, description="Exclusive upper urine ACR bound, mg/g")
    acr_quote: Optional[str] = None
    age_min: Optional[float] = Field(None, description="Inclusive lower age bound, years")
    age_max: Optional[float] = Field(None, description="Exclusive upper age bound, years")
    age_quote: Optional[str] = None
    requires_conditions: list[
        Literal["type_2_diabetes", "diabetes", "heart_failure", "hypertension", "dialysis", "kidney_transplant", "pregnancy", "albuminuria"]
    ] = []
    excludes_conditions: list[Literal["diabetes", "dialysis", "kidney_transplant", "pregnancy"]] = []
    conditions_quote: Optional[str] = None
    intent: Literal[
        "diagnosis_evaluation", "monitoring", "treatment_start", "treatment_continue_or_stop",
        "referral", "education_or_care_model", "laboratory_practice",
    ]


LLM_META_SYSTEM = (
    "You extract applicability criteria from ONE clinical guideline statement (KDIGO 2024 CKD). "
    "The criteria are used as a HARD FILTER: a patient who does not meet them will never see the statement. "
    "So set a criterion only when EVERY patient the statement is about must meet it. When unsure, leave it null.\n"
    "Use only what is explicitly stated in the statement text or its heading path. "
    "If a criterion is not stated, leave it null / empty / 'any'. Quotes must be copied verbatim from the text.\n"
    "- population: 'adult' or 'child' only if the statement is restricted to that group; "
    "'both' if it explicitly addresses adults and children; otherwise 'any'.\n"
    "- Ranges: min is inclusive, max is exclusive. 'eGFR <60' -> egfr_max=60; 'eGFR >=20' -> egfr_min=20; "
    "'eGFR 20 to 45' -> egfr_min=20, egfr_max=46. GFR categories: G1 >=90, G2 60-89, G3a 45-59, G3b 30-44, "
    "G4 15-29, G5 <15. A category range spans from the lowest to the highest bound: 'G3-G5' -> egfr_min=null, egfr_max=60; "
    "'G1-G4' -> egfr_min=15, egfr_max=null; 'G4-G5' -> egfr_max=30. Albuminuria: A1 <30, A2 30-299, A3 >=300 mg/g.\n"
    "- Alternatives ('or', 'either', 'one or more of the following', bullet lists of situations; any ONE qualifies): "
    "a criterion that belongs to only one alternative is NOT a requirement — leave it null AND list it in "
    "alternative_only_criteria. "
    "E.g. 'eGFR >=20 with ACR >=200, or heart failure' -> no eGFR, no ACR criterion; "
    "'known coronary disease, diabetes, or prior stroke' -> diabetes is not required.\n"
    "- A number that describes a result to act on, WHEN to do something, or a caution for people already covered "
    "(e.g. 'a change in eGFR of >20%', 'serum bicarbonate <18', 'plan dialysis access when GFR <15-20', "
    "'caution at G4-G5', 'continue even when eGFR falls below 30', 'the eGFR at which transplantation is undertaken') "
    "is NOT an eligibility range.\n"
    "- requires_conditions: only conditions the patient MUST have, stated in the text. A statement about people "
    "with CKD in general does not require diabetes. 'with or without diabetes' -> neither required nor excluded.\n"
    "- dialysis / kidney_transplant go in requires_conditions only if the statement is about people who are ALREADY on "
    "dialysis or have a transplant — not when it is about informing, planning or deciding to start KRT or conservative care. "
    "They go in excludes_conditions only if the text says the statement is not for people on dialysis / with a transplant."
)
LLM_META_VERSION = hashlib.sha1((LLM_META_SYSTEM + str(LLMApplicability.model_json_schema())).encode()).hexdigest()[:8]


def _call_llm_for_applicability(s: dict):
    from openai import OpenAI

    client = OpenAI(max_retries=10)
    return client.responses.parse(
        model=config.LLM_MODEL,
        temperature=0,
        input=[
            {"role": "system", "content": LLM_META_SYSTEM},
            {"role": "user", "content": f"Heading path: {s['heading_path']}\n\nStatement:\n{s['text']}"},
        ],
        text_format=LLMApplicability,
    )


def extract_applicability_with_llm_cached(s: dict) -> dict:
    """The LLM second opinion for one statement (paired with the regex-based `extract_applicability`
    above), cached on disk — this is the offline document-processing step, not per-patient inference."""

    def call():
        resp = _call_llm_for_applicability(s)
        return LLMResult(resp.output_parsed.model_dump(), extract_token_usage(resp))

    return cached_llm_call(
        f"applicability|{LLM_META_VERSION}|{s['statement_id']}|{s['text']}",
        call,
        trace_name="extract_statement_applicability_metadata",
        trace_input={"statement_id": s["statement_id"], "heading_path": s["heading_path"], "text": s["text"]},
        trace_model=config.LLM_MODEL,
    )


def _words(t):
    return re.findall(r"[a-z0-9]+", (t or "").lower())


def quote_found(quote, s, min_share=0.9):
    q = _words(quote)
    if not q:
        return False
    source = set(_words(s["heading_path"] + " " + s["text"]))
    return sum(w in source for w in q) / len(q) >= min_share


def _llm_range(llm, name):
    lo, hi = llm[f"{name}_min"], llm[f"{name}_max"]
    return None if lo is None and hi is None else (lo if lo is not None else 0, hi if hi is not None else INF)


def merge_llm_applicability(rx: dict, llm: dict, s: dict) -> dict:
    """LLM answer -> applicability record. Constraints whose quote is not in the text are dropped."""
    dropped = []
    cond_ok = quote_found(llm["conditions_quote"], s)
    cond_q = " ".join(_words(llm["conditions_quote"]))
    req, excl = set(llm["requires_conditions"]), set(llm["excludes_conditions"])
    if (req or excl) and not cond_ok:
        dropped.append("conditions (quote not in text)")
        req, excl = set(), set()
    if req & {"diabetes", "type_2_diabetes"} and not re.search(r"diabet|t2d", cond_q):
        dropped.append("diabetes (quote does not mention diabetes)")
        req -= {"diabetes", "type_2_diabetes"}
    if (req | excl) & {"dialysis", "kidney_transplant"} and not re.search(r"dialysis|transplant|krt", cond_q):
        dropped.append("dialysis/transplant (quote does not mention it)")
        req -= {"dialysis", "kidney_transplant"}
        excl -= {"dialysis", "kidney_transplant"}

    merged = dict(rx)
    merged["population"] = llm["population"]
    if llm["population"] != "any" and not quote_found(llm["population_quote"], s):
        dropped.append(f"population={llm['population']} (quote not in text)")
        merged["population"] = "any"
    merged["diabetes"] = "excluded" if "diabetes" in excl else "required" if req & {"diabetes", "type_2_diabetes"} else None
    merged["excludes_krt"] = bool(excl & {"dialysis", "kidney_transplant"})
    merged["requires_krt"] = bool(req & {"dialysis", "kidney_transplant"})
    for name in ("egfr", "acr", "age"):
        rng = _llm_range(llm, name)
        if rng is not None and not quote_found(llm[f"{name}_quote"], s):
            dropped.append(f"{name}_range {rng} (quote not in text)")
            rng = None
        merged[f"{name}_range"] = rng
    # 2. Criteria the model itself marked as belonging to one "or" alternative are never hard requirements.
    for crit in llm.get("alternative_only_criteria", []):
        if crit in ("egfr", "acr", "age") and merged[f"{crit}_range"] is not None:
            dropped.append(f"{crit}_range {merged[f'{crit}_range']} (only one 'or' alternative)")
            merged[f"{crit}_range"] = None
        if crit == "diabetes" and merged["diabetes"] == "required":
            dropped.append("diabetes (only one 'or' alternative)")
            merged["diabetes"] = None
        if crit in ("dialysis", "kidney_transplant") and merged["requires_krt"]:
            dropped.append("requires_krt (only one 'or' alternative)")
            merged["requires_krt"] = False
    # 3. G1 (>=90) is open-ended: a range that includes G1, or an upper bound above 90, has no upper limit.
    g1_quoted = re.search(r"\bg1", " ".join(_words(llm["egfr_quote"] or "")))
    if merged["egfr_range"] and merged["egfr_range"][1] != INF and (
        merged["egfr_range"][1] > 90 or (g1_quoted and merged["egfr_range"][1] >= 90)
    ):
        merged["egfr_range"] = (merged["egfr_range"][0], INF)
    merged["metadata_source"] = "llm"
    merged["dropped_by_quote_check"] = dropped
    return merged


def annotate_statements_with_llm_metadata(statements: list[dict], max_workers: int = 2) -> list[dict]:
    """Attach the LLM second opinion (`extract_applicability_with_llm_cached`) to every statement
    without a manual override, in place, as `s["llm_applicability"]`. Returns the rows that
    disagree with the regex extraction, for review (mirrors outputs/applicability_llm_disagreements.csv)."""
    from concurrent.futures import ThreadPoolExecutor

    todo = [s for s in statements if s["applicability"]["metadata_source"] != "manual"]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        llm_answers = list(pool.map(extract_applicability_with_llm_cached, todo))

    disagreements = []
    for s, llm in zip(todo, llm_answers):
        s["llm_applicability"] = llm
        rx = s["applicability"]
        merged = merge_llm_applicability(rx, llm, s)
        changed = {
            k: (rx[k], merged[k])
            for k in ("population", "diabetes", "excludes_krt", "requires_krt", "egfr_range", "acr_range", "age_range")
            if rx[k] != merged[k]
        }
        if changed:
            disagreements.append(
                {
                    "statement_id": s["statement_id"],
                    "regex -> llm": "; ".join(f"{k}: {a} -> {b}" for k, (a, b) in changed.items()),
                    "text": s["text"][:200],
                }
            )
        s["applicability_llm"] = merged
    return disagreements
