# Clinical Assistant — CKD guideline retrieval + care review

![python](https://img.shields.io/badge/python-3.13-blue) ![status](https://img.shields.io/badge/status-research%20prototype-orange) ![data](https://img.shields.io/badge/data-synthetic%20only-green)

> Research prototype on **synthetic patients**. Not a clinical decision-support system.

## What it does

- **Input:** a patient's structured record + the clinician's free-text visit note
- **Output:** the KDIGO 2024 CKD guideline statements relevant to *this* patient, and a draft care review where every recommendation cites its source statements
- **Goal:** never hide a statement the patient needs; never show one that doesn't apply (e.g. pediatric-only rules for an adult)

## Why it is interesting

- **Hybrid retrieval, measured:** BM25 + dense search + cross-encoder rerank + LLM screening, compared on a 12-patient gold set (recall@k, "inappropriate@k") — see [results](#retrieval-strategy-comparison)
- **Structured guideline parsing:** a 199-page PDF is turned into 169 statement units with heading paths, GRADE strength and linked tables
- **Deterministic safety layer:** eligibility filtering (regex + LLM consensus) and a pregnancy safety rule run in code, not in the prompt
- **Citations verified in code:** every cited statement ID is checked against the retrieved evidence; hallucinated citations are stripped and reported
- **Fixed pipeline, not an agent:** reproducible, testable, cheaper — rationale in `src/clinical_assistant/pipeline.py`
- **Observability:** every run is one Langfuse trace (prompts, outputs, tokens, latency); opt-in, fails safe

## Pipeline

```text
OFFLINE (once per guideline version)        ONLINE (per patient)
KDIGO PDF -> parse -> 169 statements        record + note -> patient state
          -> applicability metadata                       -> eligibility filter
          -> BM25 + dense index                           -> retrieval + rerank + LLM screening
                                                          -> evidence package
                                                          -> care review + citation check
```

Full write-up (method, experiments, open problems): [`notebooks/README.md`](notebooks/README.md) ·
Talk track: [`PRESENTATION.md`](PRESENTATION.md)

## Quick start

The KDIGO guideline PDF is copyrighted and is **not** in this repo. Download the
[KDIGO 2024 CKD Guideline](https://kdigo.org/wp-content/uploads/2024/03/KDIGO-2024-CKD-Guideline.pdf)
and save it as `notebooks/data/guidelines/kdigo_2024_ckd.pdf`:

```bash
mkdir -p notebooks/data/guidelines
curl -L -o notebooks/data/guidelines/kdigo_2024_ckd.pdf \
  https://kdigo.org/wp-content/uploads/2024/03/KDIGO-2024-CKD-Guideline.pdf
```

```bash
cp .env.example .env          # add OPENAI_API_KEY (Langfuse keys optional)
```

**With Docker**

```bash
docker compose up --build     # http://localhost:8501
```

**Without Docker** (Python 3.13)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[app,dev]"
pre-commit install            # secret scanning + lint on every commit
streamlit run app.py
```

First run parses the PDF and downloads the embedding/reranker models, so it takes a few minutes;
results are cached under `notebooks/data/cache`. Everything except the LLM calls runs locally.

## Using the app

- Pick a sample patient (P001–P012) or type your own
- Run the pipeline, then inspect each stage in its own tab:
  - merged patient state
  - eligibility filter (excluded / conditional statements)
  - retrieved evidence by care domain
  - final care review with per-recommendation citations and a pass/fail provenance banner

## Development

```bash
pytest                        # most tests are offline; PDF-dependent and LLM tests skip themselves
ruff check . && ruff format --check .
```

- CI (`.github/workflows/ci.yml`) runs gitleaks, ruff and pytest on every push / PR
- Pre-commit blocks `.env` files, private keys and hard-coded secrets (gitleaks) before they reach git history
- `.env` is gitignored; only `.env.example` is tracked

## Repository layout

```text
notebooks/                       Research notebook (parsing, applicability, retrieval, evaluation,
                                  generation) + notebooks/README.md, the full write-up
                                  notebooks/data/, notebooks/outputs/  guideline PDF, caches, review sheets
src/clinical_assistant/          The package the app imports — the notebook's online pipeline,
                                  ported to reusable modules (see below)
app.py                           Streamlit tester for the full pipeline
tests/                           pytest — deterministic checks free/offline, one LLM smoke test
```

### `src/clinical_assistant/` modules

| Module | What |
|---|---|
| `config.py` | Paths (shared with `notebooks/data`, `notebooks/outputs` so caches aren't duplicated), model names, Langfuse setup |
| `llm_calls.py` | `cached_llm_call` — document-processing only. Per-patient calls don't use this file at all (see Observability below) |
| `text_clean.py` | PDF glyph/spacing cleanup |
| `parsing.py` | Docling parse → 169 statement units with full heading paths + captioned tables |
| `applicability.py` | Regex + LLM extraction of per-statement applicability metadata (population, eGFR/ACR/age range, ...) |
| `knowledge_base.py` | `build_knowledge_base` (offline: parse + attach applicability) + `load_knowledge_base` (on-disk compiled cache) |
| `kdigo_lookups.py` | Deterministic GFR/ACR category + risk heat map lookups |
| `patient.py` | `PatientState`, record mapping, LLM note extraction, `derive()` |
| `eligibility.py` | `label_statement_eligibility` — labels each statement excluded/conditional/eligible for one patient (regex+LLM consensus) + the pregnancy safety rule |
| `retrieval.py` | `RetrievalIndex` — BM25+dense hybrid search, cross-encoder rerank, sub-queries, LLM screening |
| `generation.py` | Evidence package → LLM care review → code-checked provenance/citations |
| `pipeline.py` | `CarePipeline` — the single orchestrator: record + note → care review |
| `sample_patients.py` | The 12 synthetic test patients (P001–P012), for the app's picker and for tests |

### Caching: document-processing only, never per-patient

The on-disk LLM cache (`llm_calls.cached_llm_call`, shared with the notebook's cache file) is
used **only** by `applicability.py`'s guideline-metadata extraction — the ~160-statement batch
that runs once per guideline version, wrapped in one `build_knowledge_base` trace. Every
per-patient step (note extraction, sub-queries, screening, generation) calls the OpenAI API
directly (via `langfuse.openai`'s client wrapper, see below) with no cache in front of it at all —
every request is a real, billable API call, even for a repeated sample patient. This is
deliberate: patient-facing inference should never silently replay a stale answer. There's no
toggle to re-enable caching for that path (yet) — noted as a possible later addition, not a
priority.

### Observability (Langfuse)

Two different mechanisms, by design:

- **Per-patient LLM calls** (`note_extraction`, `llm_gap_subqueries`, `llm_based_retrieval.generate`,
  `generate_care_review`) use `from langfuse.openai import OpenAI` instead of the plain `openai`
  client — this auto-traces the full prompt (system + user messages), the full output and token
  usage (Langfuse computes cost from that) with zero extra code per call site; pass `name=` as an
  extra kwarg to `.responses.parse(...)` for a custom span name.
- **Deterministic steps** (`build_patient_state`, `label_statement_eligibility`,
  `RetrievalIndex.retrieve`/`.llm_based_retrieval`, `build_evidence_package`, `attach_provenance`,
  `CarePipeline.run`, `build_knowledge_base`) are decorated directly with Langfuse's own
  `@observe(name=..., as_type=...)` — no custom wrapper in between. (An earlier version routed
  these through a `config.traced()` indirection meant to no-op safely without keys configured;
  removed, since `@observe` already degrades safely on its own and resolves its client lazily per
  call, not at decoration/import time — one less layer for no real benefit.)
- The offline applicability-metadata extraction (`applicability.py`, via `cached_llm_call`) still
  needs its own manual generation wrapper, because a cache *hit* has no real OpenAI call for the
  native wrapper to auto-trace.

One call to `CarePipeline.run()` is one Langfuse trace (root span `care_pipeline.run`,
`as_type="chain"`), with every step above nested inside it — inputs, outputs, latency, and (for
LLM calls) the full prompt and token usage. The offline `build_knowledge_base` step is its own
separate trace (or ~160 nested traces if the applicability cache needs a fresh build) — it never
nests inside a patient's `care_pipeline.run` trace, since it always finishes before any patient
request exists.

Observation types are picked to match what each step actually is, not left as generic spans:

| Step | `as_type` |
|---|---|
| `note_extraction`, `llm_gap_subqueries`, `llm_based_retrieval.generate`, `generate_care_review`, `extract_statement_applicability_metadata` | `generation` |
| `label_statement_eligibility`, `attach_provenance` | `evaluator` |
| `retrieve`, `llm_based_retrieval`, `template_subqueries` | `retriever`/`span` (template_subqueries is a plain `span` — it's code, not a retrieval call itself) |
| `build_patient_state`, `build_evidence_package` | `span` |
| `care_pipeline.run`, `build_knowledge_base` | `chain` |

`label_statement_eligibility` is named for what it actually does — it labels every statement
`excluded`/`conditional`/`eligible` for one patient; it doesn't remove anything itself (callers
drop the `"excluded"` ones from their own candidate set). Don't confuse this per-patient *result*
with `s["applicability"]`/`s["applicability_llm"]`, the static per-statement *metadata* (population,
eGFR/ACR/age range, ...) computed once offline — two different things that used to share the name
"applicability," which was confusing; the per-patient one is now called `eligibility_labels`
everywhere it's passed around.

It's opt-in and fails safe: `config.get_langfuse()` returns `None` unless `LANGFUSE_PUBLIC_KEY`
and `LANGFUSE_SECRET_KEY` are set in `.env` (used by `add_trace_metadata` and the
cached/document-processing path); the `@observe`-decorated functions and `langfuse.openai`'s
wrapper degrade safely on their own even without keys. Add `LANGFUSE_BASE_URL` too if your project
isn't on Langfuse's default US cloud region (EU cloud or self-hosted). Traces are tagged
`environment=local` (set `LANGFUSE_TRACING_ENVIRONMENT` to override) and
`version=config.PIPELINE_VERSION` ("v1" — bump this when the pipeline changes materially, so old
and new runs are easy to tell apart in Langfuse).

Retrieval granularity is deliberately at the `retrieve`/`llm_based_retrieval` level, not every
`hybrid`/`rerank`/BM25/dense call inside them — enough to see the real structure (LLM screening
vs. the sub-query retrieval fallback) without flooding the trace with one node per sub-query.

**This is a fixed pipeline, not a tool-calling agent.** Two steps use an LLM with structured
output (sub-query/screening, final generation); a third (note extraction) uses one more. All
three run at fixed points in a deterministic sequence — nothing here needs an LLM to decide which
step runs next, so a dynamic agent loop would add cost and a new failure mode for no benefit that
the current pipeline needs. See `pipeline.py`'s docstring.

## Notebook vs. package

The notebook (`notebooks/02_retrieval_pipeline.ipynb`) stays the research workbench: strategy
comparisons, prompt tests, the evaluation harness against the draft gold set, and the write-up in
its README. The package is the online path only — what a real request needs — factored out so the
app (and any future caller) can import it instead of re-running notebook cells.

**One known divergence**: the package's sub-query generation (`retrieval.py`'s
`build_template_subqueries` + `build_llm_gap_subqueries`) has since been redesigned past what
`notebooks/README.md` §10 describes. The notebook's `filtered_combined_rerank` still independently
generates LLM sub-queries and dedups them against the template by exact string match after the
fact (`[q for q in llm_queries if q not in template]`); the package instead gives the LLM the
template queries up front and asks it to return only genuine gaps, so no post-hoc dedup is
needed. The notebook hasn't been re-run with this change — if the two are ever reconciled, update
notebooks/README.md's §10 description and re-run §11 first.

## Retrieval strategy comparison

Measured once, on the notebook's 12-patient draft gold set (`notebooks/README.md` §11, dated
2026-09-25, **Claude-labeled, not clinician-reviewed** — a POC-stage relevance signal, not a
validated benchmark). Strategy names below are the current package names (`retrieval.py`); the
raw numbers are unchanged from the notebook, just relabeled.

| Strategy | recall@10 | recall@15 | recall@25 | recall@35 | inappropriate@15 | Status |
|---|---|---|---|---|---|---|
| `dense_raw` (internal only, not in UI) | 0.27 | 0.34 | 0.40 | 0.47 | 3.58 | current |
| `hybrid_raw` (internal only, not in UI) | 0.27 | 0.32 | 0.40 | 0.49 | 3.75 | current |
| `filtered_hybrid_raw` | 0.29 | 0.36 | 0.46 | 0.52 | 0.58 | current |
| `code_query_reformulation_rerank` | 0.27 | 0.39 | 0.45 | 0.52 | 0.17 | current |
| `llm_query_reformulation_rerank` | 0.35 | 0.42 | 0.58 | 0.66 | 0.50 | **stale — pre-redesign** |
| `code_and_llm_query_reformulation_rerank` | 0.32 | 0.47 | 0.58 | 0.68 | 0.17 | **stale — pre-redesign** |
| `llm_based_retrieval` | 0.56 | 0.67 | 0.80 | 0.80 | 0.00 | current |
| `llm_based_retrieval_plus_query_reformulation` (pipeline default) | 0.56 | 0.67 | 0.82 | 0.93 | 0.00 | **stale — pre-redesign** |

"Current" rows don't touch LLM sub-query generation at all (no reformulation, code-only template
queries, or pure LLM screening with no query rewriting), so the sub-query redesign couldn't have
changed their numbers. "Stale" rows route through `build_llm_gap_subqueries`, the function that
redesign changed (independent generation + post-hoc string-dedup → template-aware prompt, no
dedup needed) — including the pipeline's own default strategy. Those three rows, especially the
default, are last known good, not current — re-running `notebooks/README.md` §11 against today's
code is the only way to get real post-redesign numbers (tracked as open work, not yet done).

## License

[MIT](LICENSE) for the code. The KDIGO guideline remains © KDIGO and is not redistributed here.
