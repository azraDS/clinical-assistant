"""Patient state: EHR record (deterministic mapping) merged with the visit note (LLM extraction
into the same schema). Derived fields (G/A category, risk, drug classes, missing data) are
computed in code, never by the LLM. See notebooks/README.md §3.
Ported from notebooks/02_retrieval_pipeline.ipynb §7.
"""

import re
from typing import Literal, Optional

from langfuse import observe
from pydantic import BaseModel, Field

from . import config
from .kdigo_lookups import RISK_HEATMAP, acr_category, gfr_category


class PatientState(BaseModel):
    age: Optional[int] = None
    sex: Optional[Literal["M", "F"]] = None
    egfr: Optional[float] = None
    egfr_date: Optional[str] = None
    acr_mg_g: Optional[float] = None
    potassium: Optional[float] = None
    sbp: Optional[int] = None
    known_ckd: Optional[bool] = None
    hypertension: Optional[bool] = None
    diabetes: Literal["type_1", "type_2", "none", "unknown"] = "unknown"
    heart_failure: Optional[bool] = None
    on_dialysis: Optional[bool] = None
    kidney_transplant: Optional[bool] = None
    pregnant: Optional[bool] = None
    medications: list[str] = []
    symptoms: list[str] = []
    concerns: list[str] = Field(default_factory=list, description="Clinician's stated concerns / questions this visit")


DRUG_TO_CLASS = {
    r"pril\b": "RASi",
    r"sartan\b": "RASi",
    r"gliflozin\b": "SGLT2i",
    r"finerenone|spironolactone|eplerenone": "MRA",
    r"glutide\b|tirzepatide": "GLP-1 RA",
    r"statin\b": "statin",
    r"amlodipine|nifedipine|felodipine": "CCB",
    r"metformin": "metformin",
    r"ibuprofen|naproxen|diclofenac": "NSAID",
}


def drug_classes(medications: list[str]) -> list[str]:
    return sorted({cls for med in medications for pat, cls in DRUG_TO_CLASS.items() if re.search(pat, med, re.I)})


def derive(state: PatientState) -> dict:
    missing = []
    if state.acr_mg_g is None:
        missing.append("urine ACR")
    if state.diabetes == "unknown":
        missing.append("diabetes status")
    if state.egfr is None:
        missing.append("eGFR")
    g, a = gfr_category(state.egfr), acr_category(state.acr_mg_g)
    return {
        "is_adult": state.age is None or state.age >= 18,
        "gfr_category": g,
        "acr_category": a,
        "risk_category": RISK_HEATMAP.get(g, {}).get(a) if g and a else None,
        "drug_classes": drug_classes(state.medications),
        "missing_data": missing,
    }


def state_from_record(record: dict) -> PatientState:
    """record: {"demographics": {"age", "sex"}, "diagnoses": [...], "medications": [...],
    "labs": [{"date", "test", "value", "unit"}, ...]}."""
    labs = {}
    for lab in sorted(record["labs"], key=lambda entry: entry["date"]):
        labs[lab["test"].lower()] = lab
    dx = " ".join(record["diagnoses"]).lower()
    return PatientState(
        age=record["demographics"]["age"],
        sex=record["demographics"]["sex"],
        egfr=labs.get("egfr", {}).get("value"),
        egfr_date=labs.get("egfr", {}).get("date"),
        acr_mg_g=labs.get("acr", {}).get("value"),
        potassium=labs.get("potassium", {}).get("value"),
        known_ckd="chronic kidney disease" in dx or "ckd" in dx,
        hypertension="hypertension" in dx,
        diabetes="type_2" if "type 2 diabetes" in dx else "type_1" if "type 1 diabetes" in dx else "unknown",
        heart_failure="heart failure" in dx or None,
        on_dialysis="dialysis" in dx,
        kidney_transplant="transplant" in dx,
        pregnant=True if "pregnan" in dx else None,
        medications=record["medications"],
    )


NOTE_EXTRACTION_SYSTEM = (
    "Extract patient facts from a clinician's visit note into the schema. "
    "Only record facts explicitly stated. Unknown -> null (or 'unknown' for diabetes). "
    "Convert BP 'a/b' to sbp=a. Do not infer diagnoses from lab values.\n"
    "- A condition that is simply not mentioned is UNKNOWN, not absent: diabetes='none' only if the note says so "
    "(e.g. 'no diabetes', 'non-diabetic'); heart_failure / on_dialysis / kidney_transplant / pregnant / hypertension = false "
    "only if the note explicitly rules it out, otherwise null.\n"
    "- 'no PMH' / 'no past medical history' means no known diagnoses: set diabetes='none' and the booleans false.\n"
    "- concerns: the clinician's questions or worries this visit, short phrases."
)


def _call_llm_for_note_extraction(note: str):
    # langfuse.openai's client wrapper auto-traces this call (full prompt, output, token usage).
    from langfuse.openai import OpenAI

    client = OpenAI(max_retries=10)
    return client.responses.parse(
        model=config.LLM_MODEL,
        input=[{"role": "system", "content": NOTE_EXTRACTION_SYSTEM}, {"role": "user", "content": note}],
        text_format=PatientState,
        name="note_extraction",
    )


def extract_patient_state_from_note(note: str) -> PatientState:
    """Per-patient inference — always a live call, never cached."""
    return _call_llm_for_note_extraction(note).output_parsed


def state_from_note_offline(note: str) -> PatientState:
    # Minimal regex stand-in so the pipeline runs without an API key. Not a replacement for the
    # LLM extraction above (see notebooks/README.md §3's note-extraction prompt tests).
    bp = re.search(r"BP\s*(\d{2,3})/(\d{2,3})", note)
    symptoms = [
        s
        for s in ["fatigue", "nausea", "itch", "oedema", "edema", "breathless", "weight loss", "poor appetite"]
        if s in note.lower()
    ]
    return PatientState(
        sbp=int(bp.group(1)) if bp else None,
        symptoms=symptoms,
        concerns=["blood pressure uncontrolled"] if re.search(r"elevated despite|uncontrolled", note, re.I) else [],
    )


def merge_states(record_state: PatientState, note_state: PatientState) -> PatientState:
    merged = record_state.model_dump()
    for key, value in note_state.model_dump().items():
        if key in ("medications", "symptoms", "concerns"):
            merged[key] = sorted(set(merged[key]) | set(value))
        elif key == "diabetes":
            if merged[key] == "unknown":
                merged[key] = value
        elif merged.get(key) is None and value is not None:
            merged[key] = value
    return PatientState(**merged)


@observe(name="build_patient_state", as_type="span")
def build_patient_state(record: dict, note: str, use_llm: bool = True) -> PatientState:
    """The online entry point: EHR record + free-text visit note -> merged PatientState."""
    record_state = state_from_record(record)
    note_state = extract_patient_state_from_note(note) if use_llm else state_from_note_offline(note)
    return merge_states(record_state, note_state)
