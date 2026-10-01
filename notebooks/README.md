# CKD Guideline Retrieval Pipeline

Research prototype on **synthetic data**, not a clinical decision-support system.
Code: [`02_retrieval_pipeline.ipynb`](02_retrieval_pipeline.ipynb). Section numbers (§) below refer to that notebook.

Goal: given a patient (structured record + the clinician's visit note), retrieve the KDIGO 2024 CKD guideline
statements a clinician should see for that visit, and **never hide a statement the patient needs**.

```text
 OFFLINE (once per guideline version)                 ONLINE (per visit)                        EVALUATION
 ────────────────────────────────────                 ──────────────────                        ──────────
 KDIGO PDF                                            EHR record ──┐                            12 synthetic patients
   │ 1. parse + clean                                 visit note ──┤ 3. patient state           × 169 statements
   ▼                                                               ▼                            labelled must / should /
 169 statement units ──► 2. applicability metadata ──► 4a. eligibility filter                   inappropriate
   │                        (regex + LLM, consensus)       │                                        │
   │                                                       ▼                                        ▼
   └──► BM25 + dense index ───────────────────────────► 4b. retrieval + rerank ──► top-k ──► 5. recall@k,
                                                           │                                   inappropriate@k,
                                                           ▼                                   filter errors
                                                     4c. evidence package ──► generation LLM (initial_experiment.ipynb)
```

---

## Step 1 — Parse, clean, and split into statement units (§1–§4)

1. **Parse**: docling turns the PDF into typed items (section headers, text, list items, tables, pictures). Cached as JSON.
2. **Clean**: the PDF's symbol font decodes wrongly (`‡`→`≥`, `¼`→`=`, `$`→`≥` in tables …), and docling drops spaces
   in some list items (`eGFR≥20ml/minper1.73m2`). Spacing is copied back from an independent pypdf extraction.
3. **Split**: walk the body in reading order, keep a heading stack, and cut one unit per
   `Recommendation x.y.z:` / `Practice Point x.y.z:` line, plus the bullet list items that follow it.
   The summary chapter at the front of the PDF has the same statements; it is used only when a body copy is
   truncated by a page break. Result: **169 units** (28 recommendations, 141 practice points), checked against pypdf
   (same 169 IDs).

### The unit hierarchy — what is a "chunk"?

The chunk is **one statement**, never a fixed-size slice of text. The headings above it and the prose around it are
attached as fields, not as separate chunks:

```text
Chapter 3: Delaying CKD progression and managing its complications          ← chapter        ┐
└── 3.7 Sodium-glucose cotransporter-2 inhibitors (SGLT2i)                   ← section        ├─ heading_path
    ├── (subheading, e.g. "Pediatric considerations", if any)                ← subheading     ┘
    │
    ├── Recommendation 3.7.1: We recommend treating patients with T2D ...    ← unit (chunk)
    ├── Recommendation 3.7.2: We recommend treating adults with CKD ...      ← unit (chunk)
    │     • eGFR ≥20 ml/min per 1.73 m2 with urine ACR ≥200 mg/g, or         ← list items = part of the unit
    │     • heart failure, irrespective of level of albuminuria.
    ├── Practice Point 3.7.1: ...                                            ← unit (chunk)
    │
    ├── "The Work Group concurs with ... " (all other prose in 3.7)          ← section_rationale (NOT indexed)
    └── Figure 23 | ... (table / figure)                                     ← table unit, linked via table_refs
```

One unit as stored:

```python
{
  "statement_id": "Rec 3.7.2",  "type": "recommendation",  "grade": "1A",  "page": 99,
  "heading_path": "Chapter 3: Delaying CKD progression ... › 3.7 Sodium-glucose cotransporter-2 inhibitors (SGLT2i)",
  "domain": "3.7 Sodium-glucose cotransporter-2 inhibitors (SGLT2i)",   # care domain = 2-level section
  "text": "Recommendation 3.7.2: We recommend treating adults with CKD with an SGLT2i for ...",
  "section_rationale": "The Work Group concurs with ...",               # parent context
  "table_refs": [],
  "applicability": {...},        # regex metadata      (step 2)
  "applicability_llm": {...},    # LLM metadata        (step 2)
}
```

### Which text is used where

| Stage | Text used | Why |
|---|---|---|
| BM25 (sparse) | `statement_id \| heading_path` + `text` | heading path adds domain words the statement may lack ("Lipid management") |
| Dense embedding | same string (`bge-base-en-v1.5`) | |
| Cross-encoder rerank | same string, paired with each sub-query | |
| LLM screening | `[id] (domain) text`, all statements the filter kept | |
| Applicability metadata | regex: `text` (child detection also reads `heading_path`); LLM: `heading_path` + `text` | |
| Evidence package (to the generator) | `text` + grade + page + first 1,200 chars of `section_rationale` + linked tables | parent context is added **after** retrieval |

The rationale is **not** embedded: it is long and general, and it would blur what the statement is about.
The 85 captioned tables and figures are **not** in the retrieval index either. They reach the generator only when a
retrieved statement cites them (e.g. PP 5.1.1 → Figure 48, referral criteria).

### Reference: what the `grade` field means (KDIGO / GRADE)

Recommendations end with a grade like "(1A)", extracted into `grade`. **Number = strength, letter = certainty of evidence.**

| Strength | Wording in the text | Meaning |
|---|---|---|
| **1** | "We **recommend** …" | Strong: most patients should receive it |
| **2** | "We **suggest** …" | Weak: many would want it, but choices vary, so decide with the patient |

| Evidence | Certainty | Meaning |
|---|---|---|
| **A** | High | Further research is very unlikely to change the conclusion |
| **B** | Moderate | Further research may change it |
| **C** | Low | Further research is likely to change it |
| **D** | Very low | The estimate is very uncertain |

Examples: Rec 3.7.2 **(1A)** SGLT2i = strong recommendation, high-certainty evidence; Rec 1.1.4.1 **(2D)** kidney
biopsy = weak suggestion, very low certainty. **Practice Points have no grade** (`null`): they are the expert panel's
advice, not graded against a formal review of the evidence.

The grade is **not** used for filtering or ranking (a 2D statement can still be exactly what a patient needs). It is
passed to the generator so the wording can match: "recommends" (1), "suggests / consider" (2), "expert advice" (practice point).

---

## Step 2 — Applicability metadata (§5)

Hybrid retrieval ranks by **topic similarity**. It cannot tell that "statin for adults ≥50 with eGFR <60" is wrong
for a 42-year-old with eGFR 94, or that a pediatric BP target is wrong for an adult. So every statement gets
structured **applicability** fields: conditions the patient must meet for the statement to apply.

| Field | Example | Filter rule |
|---|---|---|
| `population` | `adult` / `child` / `both` / `any` | child-only statements excluded for adults, and vice versa |
| `egfr_range` | `(0, 60)` from "eGFR <60" or "G3a–G5" | excluded if the patient's eGFR is known and outside |
| `acr_range` (mg/g) | `(200, ∞)` | excluded if known and outside; **"needs ACR"** if unknown |
| `age_range` | `(50, ∞)` | excluded if known and outside |
| `diabetes` | `required` / `excluded` | excluded on contradiction; "needs diabetes status" if unknown |
| `excludes_krt` / `requires_krt` | "not on dialysis or transplant" | excluded on contradiction |
| `drug_classes` | `["SGLT2i"]` | not used by the filter yet (see next steps) |

Metadata is a property of the **statement**, the same for every patient. It is extracted once, offline.

```text
                 statement text (+ heading path)
                 │                              │
                 ▼                              ▼
     regex extractor (§5)             LLM extractor (§5b, gpt-4.1, JSON schema)
     thresholds, G/A categories,      analysis → fields + verbatim quote per field
     "adults", "without diabetes"             │
                 │                   deterministic checks (code, not LLM):
                 │                    • quote words must be in the text, else drop the field
                 │                    • criteria the LLM marks as "only one 'or' alternative" → drop
                 │                    • a range including G1 has no upper bound (G1 = eGFR ≥90, open-ended)
                 │                              │
                 ▼                              ▼
          applicability               applicability_llm
                 │                              │
                 └──────────► manual overrides (§5a) ◄── known errors, versioned in code
                                    │
                                    ▼
                  clinician review sheet (§5c) — rows marked Y replace BOTH machine answers
                                    │
                                    ▼
             filter (§8): a statement is excluded only if BOTH records exclude it  (consensus)
```

Why consensus: each source alone hid statements that patients needed (see *Gaps* below). Regex and the LLM make
different mistakes, so excluding a statement needs both to be wrong at once.

### Testing the LLM prompt before any batch run (§5b tests)

Prompt changes are tested on **single statements** first. Each is one cached LLM call:

- **tuning set (30 statements)**: known traps ("or" lists, timing thresholds, G-category ranges, invented diabetes /
  dialysis constraints). The prompt is changed until these pass.
- **held-out set (18 statements)**: never used to change the prompt, so it gives the honest accuracy estimate.

Only then is the batch run over all 161 statements (the 8 with manual overrides are skipped).

---

## Step 3 — Patient state (§7)

Two inputs, merged into one `PatientState`:

```text
 EHR record (structured JSON)  ── deterministic mapping ──┐
                                                          ├──► merge ──► PatientState ──► derive() in code
 visit note (free text)        ── LLM extraction ─────────┘            (facts)            G/A category, risk,
                                  (schema, "only explicit facts")                          drug classes,
                                                                                           missing data
```

Sample record (P005):

```python
{
  "demographics": {"age": 72, "sex": "M"},
  "diagnoses": ["Hypertension", "Chronic kidney disease"],
  "medications": ["Losartan 100 mg daily", "Amlodipine 10 mg daily"],
  "labs": [
    {"date": "2025-11-02", "test": "eGFR", "value": 27, "unit": "mL/min/1.73m2"},
    {"date": "2026-05-30", "test": "eGFR", "value": 22, "unit": "mL/min/1.73m2"},
    {"date": "2026-05-30", "test": "ACR", "value": 850, "unit": "mg/g"},
    {"date": "2026-05-30", "test": "Potassium", "value": 5.6, "unit": "mmol/L"},
  ],
}
```

Visit note: *"72M CKD G4, eGFR down from 27 to 22 in 6 months, ACR 850. On losartan 100 + amlodipine. K 5.6.
Reports poor appetite and nausea for 2 months. BP 142/78. Not yet seen by nephrology."*

Resulting state:

```python
PatientState(age=72, sex="M", egfr=22.0, acr_mg_g=850.0, potassium=5.6, sbp=142, known_ckd=True,
             hypertension=True, diabetes="unknown", heart_failure=None, on_dialysis=False,
             kidney_transplant=False, pregnant=None, medications=[...], symptoms=["nausea", "poor appetite"],
             concerns=[...])
derive() → {"is_adult": True, "gfr_category": "G4", "acr_category": "A3", "risk_category": "very high",
            "drug_classes": ["CCB", "RASi"], "missing_data": ["diabetes status"]}
```

**Is clinical history used?** Partly:

- **Diagnoses and medication lists: yes**, as current facts.
- **Labs: only the latest value of each test.** Trends (27 → 22) are *not* computed from the record. P005's decline
  reaches the pipeline only because the note mentions it.
- **Past notes: no.** Only the current visit note is used.

**Long notes**: today the whole note goes to the extraction LLM. The schema extraction acts as the summary, so the
LLM never needs to write a free-text summary. For very long histories, extract per note and merge by date (newest
wins) rather than summarising, because a summary can silently drop a fact the filter depends on.

Merge policy: structured values win and the note fills gaps. Lists (meds, symptoms) are unioned.
Derived fields are computed in code, never by the LLM.

**Note-extraction prompt tests (§7)**: 8 single-note traps, one cached LLM call each (facts not stated must stay
`unknown` / `null`, "no PMH" → `none`, stated T2D + HF, pregnancy, hemodialysis). Current prompt: **8/8 passed**.

⚠️ Known bug: the medication union does not dedupe near-duplicates, so P001 ends up with
`["Amlodipine 10 mg daily", "amlodipine 10"]` (the record entry plus the note's short form). It does not change
`drug_classes`, but it is noise in the evidence package (gap #17).

---

## Step 4 — Online retrieval (§8–§10, §12)

```text
PatientState + note
      │
      ▼
4a. ELIGIBILITY FILTER (per statement, consensus of regex + LLM metadata)
      ├── excluded     a known fact contradicts it (child-only for an adult, eGFR outside range …) → removed
      ├── conditional  depends on data we don't have (e.g. ACR unknown) → KEPT, flagged "needs ACR"
      └── eligible     nothing contradicts it
      │   plus a patient-side pregnancy rule: pregnant → statements whose drug_classes ∩ {RASi, SGLT2i, MRA,
      │   statin, GLP-1 RA} are excluded (the guideline text does not say this, so it is not metadata)
      │   (eligible + conditional go on — typically ~135–145 of 169)
      ▼
4b. QUERIES — one per care domain
      ├── template sub-queries from the state (default): "statin treatment in adults aged 72 with eGFR 22",
      │   "SGLT2 inhibitor for adults with CKD eGFR 22", "management of hyperkalemia in CKD", …
      └── or LLM sub-queries written from state + note        (strategy: filtered_llmquery_rerank)
      │
      ▼   for each sub-query:
    BM25 ranking ─┐
                  ├─ RRF fusion (k=60) → top 16 ─► cross-encoder rerank (bge-reranker-v2-m3) → top 8
    dense ranking ┘
      │
      ▼
    round-robin interleave across sub-queries (every care domain gets a slot) → top k
      │
      ▼
4c. EVIDENCE PACKAGE, grouped by care domain: text, grade, page, status, "needs data", rationale excerpt, linked tables
```

**Template sub-queries** (`build_subqueries`, §10) are written by **code, not an LLM**: a fixed checklist of care
domains, each with an if-rule on the patient state, with the patient's numbers filled in. For example, "if eGFR < 60 →
add the statin, SGLT2i, monitoring, risk/referral queries"; "if K > 5.0 → add a hyperkalemia query"; one query per
symptom. The phrasing mimics guideline wording so it matches the statements. Deterministic, free, and every
checklist domain is always asked. Weakness: it only covers domains someone wrote a rule for, and it ignores the
note's free-text concerns. The alternative, **LLM sub-queries** (common in RAG, "query decomposition"), writes the
queries from state + note. It scores better here (recall@15 0.45 vs 0.38) because it picks up the note (gout,
pregnancy, falls), but it costs a call and can skip a routine domain.

**Round-robin interleave**: each sub-query produces its own ranked list, and scores from different queries can't be
compared. So the final list takes the **#1 of every sub-query, then the #2 of every sub-query, …** (skipping
duplicates) until k are collected:

```text
Q1 monitoring:    PP 2.1.1      PP 2.1.2     …
Q2 statin:        Rec 3.15.1.1  PP 3.15.1.1  …
Q3 hyperkalemia:  PP 3.11.5.1   PP 3.6.3     …
                  ── round 1 ──  ── round 2 ──
final: PP 2.1.1, Rec 3.15.1.1, PP 3.11.5.1, PP 2.1.2, PP 3.15.1.1, PP 3.6.3, …
```

This stops one big topic (e.g. RASi, with many statements) from filling the whole top k, and every care domain
gets at least one slot. Weakness: the order means "rank within its query", not overall importance, and with ~12
sub-queries and k=10, some domains get no slot at all.

Alternative, currently the best (`llm_screen`, §12): after the filter, give the LLM **all** remaining statements
(~8k tokens; this guideline is small) with the patient state and note, and let it pick and order up to 25.
With a guideline this size, no retrieval is needed.

`llm_screen_plus_retrieval` (the pipeline default, §13) takes the LLM's picks first, then fills the remaining slots
from `filtered_combined_rerank` (code checklist sub-queries **plus** LLM sub-queries). The retrieval fill catches
routine care the LLM skips. It has the best recall at k=35 (0.93) but shows a few more inappropriate statements
there (0.58 vs 0.00), because the fill is not screened.

Strategies compared (§10):

| Strategy | Query | Candidates | Ranking |
|---|---|---|---|
| `dense_raw` | raw patient text | all 169 | cosine |
| `hybrid_raw` | raw patient text | all 169 | BM25 + dense, RRF |
| `filtered_hybrid_raw` | raw patient text | filter output | BM25 + dense, RRF |
| `filtered_multiquery` | template sub-queries | filter output | RRF per sub-query, interleaved |
| `filtered_multiquery_rerank` | template sub-queries | filter output | + cross-encoder per sub-query |
| `filtered_llmquery_rerank` | LLM sub-queries | filter output | + cross-encoder per sub-query |
| `filtered_combined_rerank` | template + LLM sub-queries | filter output | + cross-encoder per sub-query |
| `llm_screen` | state + note | filter output | LLM selects and orders |
| `llm_screen_plus_retrieval` | state + note, then combined sub-queries | filter output | LLM picks first, retrieval fills the rest |

---

## Step 5 — Evaluation (§11)

### Test set = synthetic patients × gold labels

For each patient, someone reads **every** statement and labels only those that matter:

| Label | Meaning | Used for |
|---|---|---|
| `must` | a clinician would expect to see it for this visit | recall@k, filter errors |
| `should` | useful, not essential | not scored yet |
| `inappropriate` | showing it would be wrong or misleading for this patient | inappropriate@k |
| *(blank)* | not relevant | treated as not relevant |

`eligible` / `conditional` / `excluded` are **not** labels. They are the filter's output, shown in the review sheet
so the reviewer can spot a `must` the filter removed.

The labels answer a different question from step 2. Metadata is a property of the statement ("is this for adults?").
A gold label depends on the patient ("does *this* patient need to see it?"). Both are defined over the same
169 statements.

Test patients (P001–P003 from the first notebook; P004–P012 each exercise one guideline area or filter rule):

| | Scenario | | Scenario |
|---|---|---|---|
| P001 | 67M, HTN, single low eGFR, no ACR | P007 | 12-year-old, CAKUT, high BP |
| P002 | 54F, known CKD on ramipril | P008 | T2D + heart failure + AF on warfarin |
| P003 | healthy control (no `must`) | P009 | pregnant, still on lisinopril |
| P004 | T2D, ACR 420, max ACEi, HbA1c high | P010 | frail 80F, falls, OTC ibuprofen |
| P005 | G4 A3, fast decline, K 5.6, uremic symptoms | P011 | gout treated with naproxen |
| P006 | new albuminuria + hematuria, may want pregnancy | P012 | on hemodialysis (mostly out of scope) |

⚠️ The current labels are a **draft written by Claude** (an LLM), not a clinician. They are fine for
comparing strategies. They do not establish clinical accuracy.

### Metrics

- **recall@k**: of the `must` statements, the fraction that appear in the top k. recall@10 = found in the top 10.
- **inappropriate@k**: how many `inappropriate` statements appear in the top k. This stands in for precision, and it
  measures the safety-relevant part of it.
- **filter errors**: `must` statements the filter removed. Target **0**: nothing later in the pipeline can recover them.

Recall comes first because a missing statement is invisible to the clinician and to the generator. An extra,
harmless statement costs a few tokens and the generator can ignore it. Precision does matter where the extra
statement is *wrong* for the patient, which is why inappropriate@k is tracked separately rather than ignored.

### How many patients? What about a big guideline?

- 12 patients × ~11 `must` each ≈ 120 relevance judgements. That is enough to find failure *types* (every gap below
  was found this way), but too few for precise numbers: one statement moves a patient's recall by 6–17 points.
  For a report, aim for 30–50 cases covering each care domain and each filter rule at least twice.
- With 1,000+ statements, nobody labels every row. Use **pooling**: for each patient, label only the union of the
  top-k from every strategy (plus anything the filter excluded that looks relevant). Unlabelled statements count as
  not relevant. This is the standard approach in IR evaluation (TREC).

---

## Results (2026-09-25, draft gold set)

| Strategy | recall@10 | recall@15 | recall@25 | recall@35 | inappropriate@15 |
|---|---|---|---|---|---|
| `dense_raw` (original approach) | 0.27 | 0.34 | 0.40 | 0.47 | 3.58 |
| `hybrid_raw` | 0.27 | 0.32 | 0.40 | 0.49 | 3.75 |
| `filtered_hybrid_raw` | 0.29 | 0.36 | 0.46 | 0.52 | 0.58 |
| `filtered_multiquery` | 0.22 | 0.34 | 0.43 | 0.52 | 0.25 |
| `filtered_multiquery_rerank` | 0.27 | 0.39 | 0.45 | 0.52 | 0.17 |
| `filtered_llmquery_rerank` | 0.35 | 0.42 | 0.58 | 0.66 | 0.50 |
| `filtered_combined_rerank` | 0.32 | 0.47 | 0.58 | 0.68 | 0.17 |
| `llm_screen` | 0.56 | 0.67 | 0.80 | 0.80 | 0.00 |
| **`llm_screen_plus_retrieval`** (pipeline default) | **0.56** | **0.67** | **0.82** | **0.93** | 0.00 |

Filter errors: **0**. Applicability prompt: 30/30 tuning, **16/18 held-out**. Note-extraction prompt: **8/8**
(§7 trap set). Reranker comparison (`bge-reranker-v2-m3` vs `medcpt`, §10): near-identical, kept `bge-reranker-v2-m3`.

Numbers moved since the first draft (e.g. `llm_screen` recall@15 was 0.60, now 0.67) because the applicability
filter and sub-queries were tuned further; re-run §11 after any filter or query change to keep this table current.

---

## Gaps and failure points found, and how they were handled

| # | Failure | Found by | Fix |
|---|---|---|---|
| 1 | PDF symbol font decoded wrongly (`‡50` = `≥50`); docling drops spaces | reading output | glyph map + respacing from pypdf (§2) |
| 2 | Some statements truncated by page breaks or mislabelled by docling | pypdf completeness check | fall back to the summary copy; match statement lines before item type |
| 3 | Raw-note dense retrieval surfaced pediatric and wrong-threshold statements (3.7 inappropriate in top 15) | gold set | eligibility filter (§8) |
| 4 | One query can't cover all care domains | gold set | care-domain sub-queries + interleave (§10) |
| 5 | **Regex reads "A or B" as "A required"** (Rec 3.7.2: "ACR ≥200 **or** heart failure" hid it from a heart-failure patient; PP 5.2.3.2 hid malnutrition screening from an 80-year-old) | filter-error check (5 errors) | consensus filter + manual overrides for the two statements |
| 6 | **LLM alone invented constraints** ("diabetes required" on generic statements, "requires dialysis" on "inform about KRT options", G3–G5 read as eGFR 30–60) | filter-error check (9 errors) | LLM made a second opinion rather than the only source; consensus rule |
| 7 | LLM prompt was changed and re-run as a whole batch each time (slow, costly, and the numbers mixed several changes) | — | single-statement prompt tests (tuning + held-out) before any batch; all answers cached |
| 8 | LLM "or" reasoning was correct in its explanation but ignored in the fields | prompt tests | schema field `alternative_only_criteria`, enforced in code |
| 9 | LLM quotes didn't match the text exactly (`≥` garbled, "…" shortening) | prompt tests | word-level quote check (≥90% of quote words in the text) |
| 10 | OpenAI rate limit (30k tokens/min) broke batch runs | — | 2 workers, retry with backoff, on-disk cache |
| 11 | Note extraction recorded P001 as "no diabetes" although the note doesn't say it | reading patient states | **open**: add to prompt tests for note extraction |
| 12 | LLM screening misses routine care the note doesn't ask about (statin, risk equation, annual monitoring) | per-patient misses (§11) | addressed: `filtered_combined_rerank`'s always-on code checklist, used as the fill in `llm_screen_plus_retrieval` (§10); see gap #18 for what still slips through |
| 13 | **Pregnancy**: RASi statements still shown to a pregnant patient (P009) | gold set | fixed: patient-side rule in §8 (`PREGNANCY_AVOID` on `drug_classes`) |
| 14 | Lab trends from the record are not computed (eGFR decline visible only via the note) | P005 | **open** |
| 15 | Gold labels are LLM-written | — | **open**: clinician review of `outputs/gold_review_P*.csv` |
| 16 | `heading_path` keeps only the deepest numbered heading: Rec 3.15.1.1 gets `Chapter 3 › 3.15.1 Lipid management`, and the parent `3.15 Cardiovascular disease…` is lost, so its words are not searchable | inspecting units | **open**: keep the full numbered-heading stack in `walk_statements` (§3) |
| 17 | Medication union doesn't dedupe near-duplicate strings across record and note (P001: `"Amlodipine 10 mg daily"` + `"amlodipine 10"`) | reading patient states | **open**: normalize before `st.medications` union, or dedupe case/dose-insensitively |
| 18 | `llm_screen` alone still misses some routine care (statin, risk equation, annual monitoring) and NOAC-over-warfarin for P008 | per-patient misses (§11 cell 64) | partially addressed: `llm_screen_plus_retrieval` fills from `filtered_combined_rerank`, which runs the always-on code checklist; **still open**: fill can itself miss (P008: `Rec 3.16.1`/`PP 3.16.2` NOAC-warfarin still absent at k=25) and raises inappropriate@35 slightly (0.58) since the fill isn't screened |
| 19 | Full end-to-end run (§14, all 12 patients + P013) shows one inappropriate citation at k=25 — `Rec 3.7.3` in P012's report (on hemodialysis, a population mostly out of scope for this guideline) | end-to-end check, 2026-09-27 | **open**: resolves at k=35 in this run, but worth a dialysis-population rule alongside the existing pregnancy rule (§8) rather than relying on a larger k |

### Lessons

- **Test prompt changes on single known cases first**, then batch. A domain expert who can say up front
  "these are the traps" (the "or" lists, "with or without diabetes", pregnancy) saves whole batch runs.
- **Check a metadata change against real patients, not just against the statements.** The disagreement list
  (regex vs LLM) doesn't catch errors both sources share. Rec 3.7.2 was wrong in both, and only the patient test found it.
- **Don't always add a new metadata field.** The pregnancy gap is not in the guideline text: KDIGO doesn't say
  "ACEi contraindicated in pregnancy". So an LLM batch over the statements can't extract it. It needs a
  **patient-side rule** using metadata we already have (`drug_classes`), e.g. `pregnant → flag statements whose
  drug_classes ∩ {RASi, SGLT2i, MRA, statin}`. That costs no LLM calls.

---

## Step 6 — Evidence package and generation (§13–§14)

```text
patient → build_evidence_package() (§13) ──► format_evidence_package() ──► generate_care_review() (§14, CareReview)
             (filter + llm_screen_plus_retrieval,                              │
              grouped by care domain, rationale + tables attached)             ▼
                                                              provenance check (code): every cited statement_id
                                                              must be in the package; invalid citations are
                                                              dropped and reported, not trusted to the LLM
                                                                              │
                                                                              ▼
                                                          outputs/care_review_<patient>.md
```

`PIPELINE_STRATEGY = "llm_screen_plus_retrieval"`, `PIPELINE_K = 25`. The generation system prompt (§14) matches
wording to grade ("recommends" / "suggests" / "expert practice point"), flags `CONDITIONAL` statements as
depending on missing data instead of presenting them as settled, and puts the clinician's own questions before
routine care the note didn't ask about.

**End-to-end check, all 12 patients + a fresh P013 added from raw inputs (§14 cells 70–71)**, at k=25 and k=35:
`invalid_citations` and `unsupported_recs` are checked automatically (target 0 both); `must_cited` and
`inappropriate_cited` are compared against the draft gold set. This is the pipeline's own acceptance check, not
just a retrieval metric.

Result (2026-09-27 run): **0/12 invalid citations, 0/12 unsupported recommendations** at both k. `must_cited`
0.64 at k=25 → 0.73 at k=35 (mean over patients with a `must` label; P003 is the healthy control, no `must`).
One inappropriate citation at k=25 — `Rec 3.7.3` in P012's report (on hemodialysis, mostly out of scope for this
guideline) — which the k=35 fill resolves to 0. P013 (new patient, nothing precomputed) ran end to end with no
errors, confirming `add_patient` → state → pipeline works outside the 12 fixed test patients.

---

## Next steps

1. ~~Pregnancy rule on existing `drug_classes`~~ — done (§8), see gap #13.
2. ~~Routine-care coverage for `llm_screen`~~ — partially done via `llm_screen_plus_retrieval` (§10); still misses
   NOAC-over-warfarin for P008 at k=25 (gap #18) — worth a targeted sub-query or checklist rule.
3. ~~Note-extraction prompt tests~~ — done (§7), 8/8 on the trap set.
4. ~~Full heading path (gap #16)~~ — done: `walk_statements` (§3) keeps every numbered ancestor plus a trailing
   subheading (e.g. `1.2 Evaluation of GFR › 1.2.2 Guidance to physicians … › Key information`), verified down to
   4-level statement numbers (80 of 169) and 0 statements missing a chapter. The full pipeline (§3–§14) has been
   re-run on this heading_path with 0 cell errors, so the results table above already reflects it.
5. Medication dedupe (gap #17): normalize the record/note medication strings before unioning.
6. Lab trends (gap #14): compute eGFR/ACR slope from the record's lab history rather than relying on the note to
   mention a decline.
7. Clinician review of the gold labels (`outputs/gold_review_P*.csv`) — the one thing no automated check can
   replace, and everything in the results table is only as good as this.
8. ~~Move §2–§10 into `src/` modules~~ — in progress: `src/clinical_assistant/` (parsing, applicability, patient,
   retrieval, evaluation, generation) plus `app.py` (Streamlit pipeline tester). See the top-level README for the
   package layout; the notebook stays as the research/eval workbench, the package is what the app imports.

## Output files

| File | What |
|---|---|
| `outputs/statement_applicability_review.csv` | metadata review sheet; `llm_differs` marks rows to check first |
| `outputs/applicability_llm_disagreements.csv` | regex vs LLM differences + what the quote check dropped |
| `outputs/gold_review_P001…P012.csv` | gold-label review sheets (save as `gold_reviewed_P00X.csv`) |
| `outputs/evidence_package_P001.json` | example evidence package, input to the generation step |
| `outputs/care_review_<patient>.md` / `.json` | generated care review + provenance, one per patient (§14) |
| `data/cache/llm_cache_gpt-4.1.json` | every LLM answer, keyed by prompt hash; delete to force fresh calls |
