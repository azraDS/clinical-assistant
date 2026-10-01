"""The 12 synthetic test patients from the research notebook (P001-P003 from
initial_experiment.ipynb; P004-P012 each exercise one guideline area or filter rule), for the
Streamlit app's example picker and for smoke-testing the package against a known baseline.
See notebooks/README.md §5 ("Test set") for what each patient is meant to exercise.
"""

RECORDS: dict[str, dict] = {
    "P001": {
        "demographics": {"age": 67, "sex": "M"},
        "diagnoses": ["Hypertension"],
        "medications": ["Amlodipine 10 mg daily"],
        "labs": [
            {"date": "2026-04-12", "test": "eGFR", "value": 48, "unit": "mL/min/1.73m2"},
            {"date": "2026-04-12", "test": "Creatinine", "value": 1.6, "unit": "mg/dL"},
        ],
    },
    "P002": {
        "demographics": {"age": 54, "sex": "F"},
        "diagnoses": ["Hypertension", "Chronic kidney disease"],
        "medications": ["Ramipril 10 mg daily"],
        "labs": [
            {"date": "2026-03-10", "test": "eGFR", "value": 38, "unit": "mL/min/1.73m2"},
            {"date": "2026-03-10", "test": "Creatinine", "value": 1.8, "unit": "mg/dL"},
        ],
    },
    "P003": {
        "demographics": {"age": 42, "sex": "M"},
        "diagnoses": [],
        "medications": [],
        "labs": [
            {"date": "2026-05-02", "test": "eGFR", "value": 94, "unit": "mL/min/1.73m2"},
            {"date": "2026-05-02", "test": "Creatinine", "value": 0.9, "unit": "mg/dL"},
        ],
    },
    # T2D + severely increased albuminuria on ACEi + metformin: SGLT2i, finerenone, GLP-1 RA, referral (ACR >=300)
    "P004": {
        "demographics": {"age": 61, "sex": "F"},
        "diagnoses": ["Type 2 diabetes", "Hypertension", "Chronic kidney disease"],
        "medications": ["Metformin 1 g twice daily", "Lisinopril 40 mg daily"],
        "labs": [
            {"date": "2026-02-20", "test": "eGFR", "value": 55, "unit": "mL/min/1.73m2"},
            {"date": "2026-05-18", "test": "eGFR", "value": 52, "unit": "mL/min/1.73m2"},
            {"date": "2026-05-18", "test": "ACR", "value": 420, "unit": "mg/g"},
            {"date": "2026-05-18", "test": "Potassium", "value": 4.6, "unit": "mmol/L"},
            {"date": "2026-05-18", "test": "HbA1c", "value": 8.2, "unit": "%"},
        ],
    },
    # Advanced CKD G4 A3 with hyperkalemia on ARB and uremic symptoms: KRT planning, referral, potassium
    "P005": {
        "demographics": {"age": 72, "sex": "M"},
        "diagnoses": ["Hypertension", "Chronic kidney disease"],
        "medications": ["Losartan 100 mg daily", "Amlodipine 10 mg daily"],
        "labs": [
            {"date": "2025-11-02", "test": "eGFR", "value": 27, "unit": "mL/min/1.73m2"},
            {"date": "2026-05-30", "test": "eGFR", "value": 22, "unit": "mL/min/1.73m2"},
            {"date": "2026-05-30", "test": "ACR", "value": 850, "unit": "mg/g"},
            {"date": "2026-05-30", "test": "Potassium", "value": 5.6, "unit": "mmol/L"},
        ],
    },
    # Young woman, new A2 albuminuria + microscopic hematuria, no diabetes, untreated high BP: cause, RASi, teratogenicity
    "P006": {
        "demographics": {"age": 45, "sex": "F"},
        "diagnoses": [],
        "medications": [],
        "labs": [
            {"date": "2026-06-01", "test": "eGFR", "value": 68, "unit": "mL/min/1.73m2"},
            {"date": "2026-06-01", "test": "ACR", "value": 150, "unit": "mg/g"},
        ],
    },
    # Child: CAKUT, CKD G3a, high BP — pediatric statements must appear, adult-only must not
    "P007": {
        "demographics": {"age": 12, "sex": "M"},
        "diagnoses": ["Chronic kidney disease", "Congenital anomalies of the kidney and urinary tract"],
        "medications": [],
        "labs": [
            {"date": "2026-04-22", "test": "eGFR", "value": 55, "unit": "mL/min/1.73m2"},
            {"date": "2026-04-22", "test": "ACR", "value": 90, "unit": "mg/g"},
        ],
    },
    # T2D + HFrEF + atrial fibrillation on warfarin, A1: SGLT2i for HF, NOAC over warfarin, MRA hyperkalemia
    "P008": {
        "demographics": {"age": 58, "sex": "M"},
        "diagnoses": [
            "Type 2 diabetes",
            "Heart failure with reduced ejection fraction",
            "Atrial fibrillation",
            "Chronic kidney disease",
        ],
        "medications": [
            "Metformin 1 g twice daily",
            "Ramipril 5 mg daily",
            "Bisoprolol 5 mg daily",
            "Spironolactone 25 mg daily",
            "Warfarin 5 mg daily",
        ],
        "labs": [
            {"date": "2026-05-05", "test": "eGFR", "value": 41, "unit": "mL/min/1.73m2"},
            {"date": "2026-05-05", "test": "ACR", "value": 25, "unit": "mg/g"},
            {"date": "2026-05-05", "test": "Potassium", "value": 5.2, "unit": "mmol/L"},
        ],
    },
    # Pregnant, CKD A3 on ACEi: drug safety in pregnancy (RASi/SGLT2i/statin would be harmful)
    "P009": {
        "demographics": {"age": 34, "sex": "F"},
        "diagnoses": ["Chronic kidney disease", "Pregnancy"],
        "medications": ["Lisinopril 10 mg daily"],
        "labs": [
            {"date": "2026-06-10", "test": "eGFR", "value": 75, "unit": "mL/min/1.73m2"},
            {"date": "2026-06-10", "test": "ACR", "value": 320, "unit": "mg/g"},
        ],
    },
    # Frail 80-year-old, falls, orthostatic dizziness, OTC NSAID, weight loss: less intensive BP, deprescribing, nutrition
    "P010": {
        "demographics": {"age": 80, "sex": "F"},
        "diagnoses": ["Hypertension", "Chronic kidney disease", "Osteoarthritis"],
        "medications": ["Amlodipine 5 mg daily", "Ramipril 5 mg daily", "Ibuprofen 400 mg as needed"],
        "labs": [
            {"date": "2026-05-12", "test": "eGFR", "value": 33, "unit": "mL/min/1.73m2"},
            {"date": "2026-05-12", "test": "ACR", "value": 45, "unit": "mg/g"},
        ],
    },
    # Gout flare treated with naproxen, CKD G3b A1, on ARB: urate-lowering, acute gout treatment, NSAID harm
    "P011": {
        "demographics": {"age": 50, "sex": "M"},
        "diagnoses": ["Hypertension", "Chronic kidney disease", "Gout"],
        "medications": ["Losartan 50 mg daily", "Naproxen 500 mg twice daily"],
        "labs": [
            {"date": "2026-06-03", "test": "eGFR", "value": 40, "unit": "mL/min/1.73m2"},
            {"date": "2026-06-03", "test": "ACR", "value": 12, "unit": "mg/g"},
            {"date": "2026-06-03", "test": "Uric acid", "value": 10.1, "unit": "mg/dL"},
        ],
    },
    # On hemodialysis: most of this guideline is out of scope; statements restricted to non-KRT must not appear
    "P012": {
        "demographics": {"age": 63, "sex": "M"},
        "diagnoses": ["Type 2 diabetes", "Kidney failure on hemodialysis"],
        "medications": ["Insulin glargine 20 units nightly", "Sevelamer 800 mg with meals"],
        "labs": [
            {"date": "2026-06-15", "test": "eGFR", "value": 8, "unit": "mL/min/1.73m2"},
            {"date": "2026-06-15", "test": "Potassium", "value": 5.3, "unit": "mmol/L"},
        ],
    },
}

# Current-visit notes as a clinician would type them
NOTES: dict[str, str] = {
    "P001": "67M, HTN on amlodipine 10. Increasing fatigue over recent months. BP 158/94 today, "
    "still elevated despite meds. Recent bloods: eGFR 48, creat 1.6. No urine ACR on file. What should we do?",
    "P002": "54F known CKD + HTN on ramipril 10mg. Repeat eGFR 38. No new symptoms. BP 138/84. Review management.",
    "P003": "42M routine bloods, eGFR 94, creat 0.9. Asymptomatic, no PMH.",
    "P004": "61F T2D and CKD, on metformin and lisinopril 40 (max dose). eGFR 52, ACR 420, K 4.6, HbA1c 8.2 - "
    "glucose not at target. BP 136/80. Not on SGLT2i. How to optimise kidney protection?",
    "P005": "72M CKD G4, eGFR down from 27 to 22 in 6 months, ACR 850. On losartan 100 + amlodipine. K 5.6. "
    "Reports poor appetite and nausea for 2 months. BP 142/78. Not yet seen by nephrology.",
    "P006": "45F, incidental ACR 150 and microscopic hematuria on dipstick. eGFR 68. No diabetes. BP 142/90 on two readings, "
    "no treatment. No FHx of kidney disease. Uses no contraception, may want another child. Next steps?",
    "P007": "12M with CAKUT and CKD G3a (eGFR 55), ACR 90. Office BP above 95th percentile for age/sex/height on repeat. "
    "Growth tracking along 10th centile. Not on medication. Management plan?",
    "P008": "58M T2D, HFrEF (EF 30%), AF on warfarin (labile INR), CKD eGFR 41, ACR 25. On metformin 1g bd, ramipril 5, "
    "bisoprolol, spironolactone 25. K 5.2. BP 118/70. Review meds.",
    "P009": "34F, 14 weeks pregnant, known CKD, ACR 320, eGFR 75. Still taking lisinopril 10mg. BP 128/82. First visit since conception.",
    "P010": "80F with CKD eGFR 33, ACR 45, HTN on amlodipine + ramipril. BP 150/70 seated, dizzy on standing. "
    "Two falls this year. Takes ibuprofen from the pharmacy for knee pain. Lost 4 kg in 6 months, eats little. Lives alone.",
    "P011": "50M CKD G3b (eGFR 40, ACR 12), HTN on losartan 50. Second gout attack this year (big toe), started naproxen 500 bd. "
    "Urate 10.1. BP 134/84. Should we start urate-lowering?",
    "P012": "63M on hemodialysis 3x/week for 2 years, T2D on insulin. Pre-dialysis K 5.3. No statin. "
    "Asks whether he should start a statin or one of the 'new kidney drugs'.",
}

PATIENT_IDS = list(RECORDS)
