# Redrob Intelligent Candidate Ranker — India.RUNS Track 1

Ranks the top 100 candidates from a 100,000-candidate pool for the released
**Senior AI Engineer — Founding Team** job description. Built for the India.RUNS
Data & AI Challenge (Track 1).

## Running the ranker locally

**Prerequisites:** Python 3.10+ and the official `candidates.jsonl` (or
`candidates.jsonl.gz`) from the challenge bundle, placed in this folder.

> **Why `candidates.jsonl` is not in this repository:** it is the organisers'
> 100,000-row challenge dataset (~465 MB) and is intentionally git-ignored — it is
> the organisers' data to distribute, and committing it would bloat every clone.
> Reviewers who already have the dataset can place it in this folder and run
> Step 2 to reproduce the exact submission. The bundled `sample.jsonl` (Step 2b)
> lets the ranker run without the full dataset.

### Step 1 — Set up (one time)

```bash
git clone https://github.com/Tonumoy/redrob-candidate-ranker.git
cd redrob-candidate-ranker
git lfs install && git lfs pull          # fetch the precomputed embeddings (~146 MB)
pip install -r requirements.txt
```

`pip install -r requirements.txt` installs only NumPy and scikit-learn; the
ranking step needs nothing else (no PyTorch, no network).

### Step 2 — Generate the submission file

```bash
python src/rank.py --candidates ./candidates.jsonl --out ./tonumoy_mukherjee.csv
python validate_submission.py ./tonumoy_mukherjee.csv      # -> "Submission is valid."
```

This produces `tonumoy_mukherjee.csv` — the submission file — containing a header
plus 100 rows of `candidate_id,rank,score,reasoning`. With no `--backend` flag,
`rank.py` uses `auto`, which selects the **hybrid** ranking when the Git LFS
embeddings are present. The ranking step is **CPU-only, runs offline, and stays
well within the 5-minute / 16 GB limits**: it loads the precomputed embeddings
with NumPy and computes TF-IDF with scikit-learn. Measured runtime on the full
100,000-candidate pool: **~100–140 s using under 8 GB of RAM** on a 2-core Intel
i7-7500U laptop. Both `candidates.jsonl` and `candidates.jsonl.gz` are accepted.

**Windows (PowerShell), using the bundled virtual environment** — the same two
commands with explicit paths:

```powershell
cd "D:\India Runs Hackathon\redrob-ranker\redrob-ranker"
.\.venv\Scripts\python.exe src\rank.py --candidates ".\candidates.jsonl" --out tonumoy_mukherjee.csv
.\.venv\Scripts\python.exe validate_submission.py tonumoy_mukherjee.csv
```

### Step 2b — Quick check on the bundled sample (no dataset required)

The repository ships `sample.jsonl` (50 records) so a fresh clone runs without the
full dataset:

```bash
python src/rank.py --candidates ./sample.jsonl --out ./sample_out.csv
```

This writes a ranked CSV for the 50 sample records and confirms the pipeline runs
end to end. `validate_submission.py` requires exactly 100 rows (the real
submission size), so it applies only to the Step 2 output, not to this sample.

### Step 3 — Selecting a ranking mode (`--backend`)

The same command supports three semantic backends; add `--backend` to choose one.
Each writes a valid submission CSV. `hybrid` is the backend used for the
submission.

```bash
# hybrid (submission) — 0.3*dense + 0.7*TF-IDF: keyword precision plus semantic recall
python src/rank.py --candidates ./candidates.jsonl --out out_hybrid.csv --backend hybrid

# tfidf — keyword matching only; fully offline, needs no embeddings
python src/rank.py --candidates ./candidates.jsonl --out out_tfidf.csv  --backend tfidf

# dense — bge-small embeddings only (requires the Git LFS artifacts)
python src/rank.py --candidates ./candidates.jsonl --out out_dense.csv  --backend dense

# auto (default) — hybrid if the embeddings are present, otherwise tfidf
python src/rank.py --candidates ./candidates.jsonl --out out_auto.csv   --backend auto
```

The **Ranking modes** table below describes each backend and when to use it.

> **No Git LFS?** The `.npy` files arrive as small pointer stubs, and `rank.py`
> automatically falls back to the TF-IDF backend (still valid and fully offline).
> To restore the exact hybrid result, regenerate the artifacts with
> `python src/precompute_embeddings.py --candidates ./candidates.jsonl`, or force
> the no-artifact path with `--backend tfidf`.

## Live demo

A hosted sandbox is available at
**https://redrob-candidate-ranker-tonumoy.streamlit.app** (Streamlit Community
Cloud, deployed from this repository). It accepts up to 100 candidate records
(JSON array or JSONL) or a built-in demo sample, runs the pipeline on CPU in
seconds, and returns the ranked CSV in the official submission format. A mirror is
hosted on Hugging Face Spaces
(**https://bodhi108-redrob-candidate-ranker.hf.space**; Docker configuration in
[`space/`](space/)). Both hosted demos use the TF-IDF backend for a lightweight,
reliable footprint; the full hybrid backend runs from this repository with
`--backend hybrid`.

### Ranking modes (`--backend`)

| Mode | What it does | When to use it |
|---|---|---|
| **`tfidf`** | Pure scikit-learn TF-IDF over each candidate's work evidence. Exact keyword and phrase precision; fully offline, no downloads. | Air-gapped environments; the reliable fallback. |
| **`dense`** | Cosine similarity over precomputed **bge-small** embeddings. Captures meaning and paraphrase and finds buzzword-free fits, but flattens the ranking and can demote keyword-strong top profiles. | Recall-focused experiments. |
| **`hybrid`** (submission) | `0.3·dense + 0.7·TF-IDF`, both min-max normalised. TF-IDF precision at the top combined with dense recall at the margin — keeps the strongest profile at #1 while surfacing strong plain-language candidates. | The submitted ranking. |
| **`auto`** (default) | Hybrid when `artifacts/` is present, otherwise TF-IDF. | The single-command reproduction. |

The dense embeddings are committed via **Git LFS** (`artifacts/*.npy`) so a clean
clone reproduces the hybrid result offline. They are generated offline (GPU and
network are allowed here, outside the 5-minute ranking budget) with:

```bash
python src/precompute_embeddings.py --candidates ./candidates.jsonl   # bge-small, ~3 min on a Colab T4
```

(or via the Colab notebook in [`notebooks/`](notebooks/)). The measured comparison
of the three modes is included in the submission deck.

## How it works

The job description is parsed into an auditable specification (`src/jd_spec.py`)
of must-haves, disqualifiers, dataset traps, and weights. Each candidate is scored
on six structured features in the [0, 1] range:

- **Semantic match** between the JD and the candidate's *work evidence* (career
  history and summary, not the raw skill list) — the recall engine for strong
  candidates who describe their work in plain language.
- **Demonstrated domain evidence** — information-retrieval, ranking, search, or
  recommender work shown in the career text.
- **Experience-band fit** for the target seniority.
- **Product-vs-services** background.
- **Trust-gated skills** — skills with no endorsements and no supporting duration
  earn nothing (the keyword-stuffing guard).
- **Location** alignment.

A **behavioural availability multiplier**, built from the 23 Redrob signals,
down-weights profiles that look strong on paper but are unlikely to be available.
Three guards protect the top of the ranking: a **non-fit-title cap** sinks keyword
stuffers (for example HR, Sales, or Marketing titles carrying AI skills), a
**high-precision impossibility check** (`src/validation.py`) zeroes honeypot
profiles, and an **abroad-and-won't-relocate penalty** removes profiles that
cannot be hired under the JD's no-sponsorship constraint.

Reasoning is **deterministic and fact-grounded** (no LLM): each explanation cites
only fields present on the profile, names the JD criterion matched, and surfaces a
genuine concern — so it contains no hallucinated skills or employers.

## Repository layout

```
src/jd_spec.py                JD decoded as data: weights, skills, traps, locations
src/features.py               structured features + penalties + availability
src/semantic.py               dense (precomputed) or TF-IDF semantic backend
src/validation.py             high-precision honeypot / impossibility detector
src/scoring.py                score combiner -> final score
src/reasoning.py              deterministic, fact-grounded reasoning (no LLM)
src/rank.py                   main reproduction command
src/precompute_embeddings.py  offline dense-artifact generation (optional)
app.py                        Streamlit sandbox app (CPU, TF-IDF)
sample.jsonl                  50 records for the local quick check (Step 2b)
demo_sample.jsonl             24 curated records for the sandbox "Load demo sample"
eval/local_eval.py            honeypot / non-fit-rate + top-10 inspection harness
notebooks/                    Colab GPU notebook to regenerate the dense artifacts
space/                        Hugging Face Space Docker deployment config
tests/test_format.py          submission-format invariant test
validate_submission.py        organisers' format validator (from the bundle)
submission_metadata.yaml      portal metadata mirror
```

## Compute and constraints

CPU-only · no network during ranking · ≤ 5 minutes · ≤ 16 GB RAM · top 100 only.
Measured on the full 100,000-candidate pool: **~100–140 s, under 8 GB of RAM, no
network**. AI tools used during development are declared in
`submission_metadata.yaml`.
