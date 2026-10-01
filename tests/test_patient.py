from clinical_assistant import sample_patients as sp
from clinical_assistant.patient import build_patient_state, derive


def test_p001_offline_state():
    st = build_patient_state(sp.RECORDS["P001"], sp.NOTES["P001"], use_llm=False)
    assert st.age == 67
    assert st.egfr == 48
    assert st.sbp == 158  # parsed from "BP 158/94" by the offline regex fallback
    d = derive(st)
    assert d["gfr_category"] == "G3a"
    assert d["is_adult"] is True


def test_derived_drug_classes_from_record_medications():
    st = build_patient_state(sp.RECORDS["P002"], sp.NOTES["P002"], use_llm=False)
    assert "RASi" in derive(st)["drug_classes"]  # ramipril -> RASi
