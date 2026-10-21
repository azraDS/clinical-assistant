"""Interactive tester for the CKD guideline pipeline: pick (or type) a patient, run the full
pipeline (filter -> retrieval -> LLM generation -> provenance check), and inspect every stage —
not just the final report. Research prototype on synthetic data; not a clinical tool.

Run with: streamlit run app.py
"""

import pandas as pd
import streamlit as st

from clinical_assistant import config
from clinical_assistant import sample_patients as sp
from clinical_assistant.patient import derive
from clinical_assistant.pipeline import CarePipeline

st.set_page_config(page_title="CKD guideline assistant", layout="wide")

# Every strategy below still runs note_extraction and generate_care_review (always live LLM
# calls, regardless of choice — there's no fully LLM-free pipeline run). What varies is whether
# the RETRIEVAL stage itself calls an LLM. Labelled explicitly so a strategy comparison can
# isolate retrieval quality without confusing it with "no LLM at all."
#
# recall@15 / inappropriate@15 cited below are from the notebook's 12-patient gold-set evaluation
# (notebooks/README.md §11) — measured BEFORE the sub-query gap-filling redesign, so treat them as
# directional, not exact current numbers (see this repo's top-level README, "Notebook vs. package").
STRATEGY_LABELS = {
    "llm_based_retrieval_plus_query_reformulation": (
        "LLM-based retrieval + query-reformulation fill (pipeline default) — uses LLM"
    ),
    "llm_based_retrieval": (
        "LLM-based retrieval only, no query reformulation — uses LLM — recall@15≈0.67, inappropriate@15≈0.00"
    ),
    "code_and_llm_query_reformulation_rerank": (
        "Retrieval via code + LLM query reformulation (combined) — uses LLM — recall@15≈0.47, inappropriate@15≈0.17"
    ),
    "llm_query_reformulation_rerank": (
        "Retrieval via LLM-only query reformulation, no checklist — uses LLM — recall@15≈0.42, inappropriate@15≈0.50"
    ),
    "code_query_reformulation_rerank": (
        "Retrieval via code-only query reformulation — no LLM in retrieval — recall@15≈0.39, inappropriate@15≈0.17"
    ),
    "filtered_hybrid_raw": "Retrieval: raw visit note, no query reformulation at all — no LLM in retrieval",
}


@st.cache_resource(show_spinner="Loading knowledge base and retrieval index (first run only)…")
def get_pipeline() -> CarePipeline:
    return CarePipeline(use_llm=True)


def record_form(defaults: dict, note_default: str, key_prefix: str) -> tuple[dict, str]:
    demo = defaults["demographics"]
    c1, c2 = st.columns(2)
    age = c1.number_input("Age", 0, 120, demo["age"], key=f"{key_prefix}_age")
    sex = c2.selectbox("Sex", ["M", "F"], index=0 if demo["sex"] == "M" else 1, key=f"{key_prefix}_sex")

    diagnoses = st.text_input("Diagnoses (comma-separated)", ", ".join(defaults["diagnoses"]), key=f"{key_prefix}_dx")
    medications = st.text_input(
        "Medications (comma-separated)", ", ".join(defaults["medications"]), key=f"{key_prefix}_meds"
    )

    st.caption(
        "Labs — leave value blank to omit. Type any test name (eGFR, ACR, Potassium, LDL, HDL, "
        "Bicarbonate, ...). ⚠️ Only eGFR / ACR / Potassium are mapped into the structured patient "
        "state (state_from_record) — anything else is carried in the record but not used by the "
        "deterministic filter; mention it in the visit note too if you want the LLM to see it."
    )
    lab_rows = defaults["labs"] or [{"date": "", "test": "eGFR", "value": None, "unit": "mL/min/1.73m2"}]
    labs_df = st.data_editor(
        pd.DataFrame(lab_rows),
        num_rows="dynamic",
        width="stretch",
        key=f"{key_prefix}_labs",
        column_config={
            "test": st.column_config.TextColumn("Test", help="Free text — e.g. eGFR, ACR, Potassium, LDL, HDL"),
        },
    )

    note = st.text_area("Visit note (free text)", note_default, height=120, key=f"{key_prefix}_note")

    record = {
        "demographics": {"age": int(age), "sex": sex},
        "diagnoses": [d.strip() for d in diagnoses.split(",") if d.strip()],
        "medications": [m.strip() for m in medications.split(",") if m.strip()],
        "labs": [
            {"date": row["date"], "test": row["test"], "value": row["value"], "unit": row.get("unit", "")}
            for row in labs_df.to_dict("records")
            if row.get("value") not in (None, "")
        ],
    }
    return record, note


def render_patient_state(st_state) -> None:
    d = derive(st_state)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Merged patient state**")
        st.json(st_state.model_dump(exclude_defaults=True))
    with c2:
        st.markdown("**Derived**")
        st.json(d)
        if d["missing_data"]:
            st.info("Missing data: " + ", ".join(d["missing_data"]))


def render_filter(eligibility_labels: dict):
    counts = pd.Series([v[0] for v in eligibility_labels.values()]).value_counts()
    st.markdown("**Eligibility filter**")
    st.write({k: int(v) for k, v in counts.items()})
    excluded_rows = [
        {"statement_id": sid, "reason": "; ".join(reasons)}
        for sid, (status, reasons, _needs) in eligibility_labels.items()
        if status == "excluded"
    ]
    with st.expander(f"Excluded statements ({len(excluded_rows)})"):
        st.dataframe(pd.DataFrame(excluded_rows), width="stretch", height=300)


def render_evidence(pkg: dict):
    st.markdown("**Evidence package, by care domain**")
    for domain, items_ in pkg["evidence_by_domain"].items():
        with st.expander(f"{domain} ({len(items_)})"):
            for e in items_:
                flag = f" — needs {', '.join(e['needs_data'])}" if e["needs_data"] else ""
                st.markdown(f"**{e['statement_id']}** (grade {e['grade'] or 'PP'}, p.{e['page']}){flag}  \n{e['text']}")


def render_report(result: dict):
    invalid = result["invalid_citations"]
    if invalid:
        st.error(
            f"🚨 Provenance check FAILED — the LLM cited {len(invalid)} statement_id(s) that were not in the "
            f"evidence package (hallucinated), stripped before display: {', '.join(invalid)}"
        )
    else:
        st.success(
            "✅ Provenance check passed — every cited statement_id was verified against the "
            "evidence package (0 hallucinated citations)"
        )
    st.markdown("### Summary")
    st.write(result["summary"])
    for n, r in enumerate(result["recommendations"], 1):
        tag = (
            "" if r["applies"] == "applies" else f" — conditional: needs {', '.join(r['missing_data']) or 'more data'}"
        )
        st.markdown(f"**{n}. {r['action']}**{tag}")
        st.caption(f"Why this patient: {r['why_this_patient']}")
        if r["caution"]:
            st.warning(r["caution"])
        if r["unsupported"]:
            st.error("No valid citation — not supported by the retrieved guideline text.")
        for src in r["sources"]:
            certainty = f", {src['evidence_certainty']}-certainty evidence" if src["evidence_certainty"] else ""
            with st.expander(f"{src['statement_id']} — {src['strength']}{certainty} (p.{src['page']})"):
                st.write(src["heading_path"])
                st.write(src["text"])
    if result["data_to_obtain"]:
        st.markdown("**Data to obtain:** " + "; ".join(result["data_to_obtain"]))
    st.markdown(f"**Not covered by the retrieved evidence:** {result['not_covered']}")


def main():
    st.title("CKD guideline retrieval + care review")
    st.caption(
        "Research prototype on synthetic data, not a clinical decision-support system. "
        "See notebooks/README.md for how the pipeline works."
    )

    with st.sidebar:
        st.header("Settings")
        if config.get_langfuse() is not None:
            st.caption(f"🔎 Tracing to Langfuse (environment=local, version={config.PIPELINE_VERSION})")
        else:
            st.caption("Tracing off — add LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY to .env to enable")
        strategy = st.selectbox(
            "Retrieval strategy",
            list(STRATEGY_LABELS.keys()),
            format_func=lambda s: STRATEGY_LABELS[s],
            index=0,
            help="llm_based_retrieval_plus_query_reformulation is the pipeline default, chosen for highest "
            "recall in the notebook's gold-set evaluation (notebooks/README.md §11). Note extraction and the "
            "final care-review generation always use the LLM regardless of this choice — only the retrieval "
            "stage itself varies.",
        )
        k = st.slider("k (statements passed to the generator)", 10, 40, config.PIPELINE_K)
        st.caption(
            "Patient-level inference (note extraction, sub-queries/screening, generation) always "
            "runs live — no caching, even for a repeated sample patient. Only the one-time "
            "169-statement guideline applicability metadata stays cached."
        )

        st.divider()
        source = st.radio("Patient", ["Sample patient", "Custom patient"])
        if source == "Sample patient":
            pid = st.selectbox("Pick a sample patient (P001-P012)", sp.PATIENT_IDS)
            defaults, note_default = sp.RECORDS[pid], sp.NOTES[pid]
            patient_label = pid
        else:
            pid = "custom"
            defaults = {"demographics": {"age": 60, "sex": "M"}, "diagnoses": [], "medications": [], "labs": []}
            note_default = ""
            patient_label = "custom"

    # keyed by the specific patient (not just "Sample patient" vs "Custom patient"), so switching
    # the sample-patient dropdown creates fresh widgets instead of Streamlit reusing the previous
    # patient's typed-in values for every field except the ones that happen to differ in type.
    record, note = record_form(defaults, note_default, key_prefix=f"{source}:{pid}")

    run = st.button("Run pipeline", type="primary", disabled=not note.strip())
    if not run:
        st.info("Fill in the patient (or pick a sample) and click **Run pipeline**.")
        return

    pipe = get_pipeline()

    tab_state, tab_filter, tab_evidence, tab_report = st.tabs(
        ["Patient state", "Eligibility filter", "Evidence package", "Care review"]
    )

    # One pipeline run powers all four tabs — patient state is extracted exactly once (not once
    # for a "preview" and again inside the pipeline). Every per-patient LLM step is always a live,
    # traced call (langfuse.openai's client wrapper) — a real, billable call every time, by design.
    with st.spinner("Running the pipeline…"):
        res = pipe.run(record, note, patient_label=patient_label, strategy=strategy, k=k)

    with tab_state:
        render_patient_state(res.patient_state)
    with tab_filter:
        render_filter(res.eligibility_labels)
    with tab_evidence:
        render_evidence(res.evidence_package)
    with tab_report:
        render_report(res.result)
        st.download_button("Download report (.md)", res.report_markdown, file_name=f"care_review_{patient_label}.md")


if __name__ == "__main__":
    main()
