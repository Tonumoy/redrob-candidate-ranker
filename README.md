# Redrob Intelligent Candidate Ranker — India.RUNS Track 1

Ranks the top-100 candidates from the 100,000-candidate pool for the released
**Senior AI Engineer — Founding Team** JD. Built for the Data & AI Challenge.

## Run it locally — exact commands (anyone)

**Prerequisites:** Python 3.10+ and the `candidates.jsonl` (or
`candidates.jsonl.gz`) file from the challenge bundle, placed in this folder.

### Step 1 — Clone and set up (one time)

```bash
git clone https://github.com/Tonumoy/redrob-candidate-ranker.git
cd redrob-candidate-ranker
git lfs install && git lfs pull          # fetch the precomputed embeddings (~146 MB)
pip install -r requirements.txt
```

### Step 2 — Generate the submission file (one command)

```bash
python src/rank.py --candidates ./candidates.jsonl --out ./tonumoy_mukherjee.csv
python validate_submission.py ./tonumoy_mukherjee.csv      # -> "Submission is valid."
```

`tonumoy_mukherjee.csv` (a header + 100 rows of
`candidate_id,rank,score,reasoning`) is the file you upload. With no `--backend`
flag, `rank.py` defaults to **`auto`**, which selects the shipped **hybrid**
ranking when the LFS embeddings are present. The ranking step is **CPU-only, no
network, well under the 5-min / 16 GB budget** — it just *loads* the precomputed
embeddings (numpy) and computes TF-IDF (scikit-learn); measured **~95–140 s in
under 8 GB RAM** on a 2-core Intel i7-7500U laptop. Both `candidates.jsonl` and
`candidates.jsonl.gz` work.

> **Windows / VS Code (PowerShell), using the bundled venv** — identical two
> steps, with explicit paths:
> ```powershell
> cd "D:\India Runs Hackathon\redrob-ranker\redrob-ranker"
> .\.venv\Scripts\python.exe src\rank.py --candidates ".\candidates.jsonl" --out tonumoy_mukherjee.csv
> .\.venv\Scripts\python.exe validate_submission.py tonumoy_mukherjee.csv
> ```

### Step 3 — Run any of the three modes (`--backend`)

The same command runs all three measured semantic backends — just add
`--backend`. Each writes a valid submission CSV; **`hybrid` is what we ship.**

```bash
# SHIPPED — 0.3*dense + 0.7*TF-IDF: keyword precision at the top + dense recall
python src/rank.py --candidates ./candidates.jsonl --out out_hybrid.csv --backend hybrid

# keywords only — pure scikit-learn TF-IDF, fully offline, needs no embeddings
python src/rank.py --candidates ./candidates.jsonl --out out_tfidf.csv  --backend tfidf

# embeddings only — cosine over bge-small (needs the LFS artifacts)
python src/rank.py --candidates ./candidates.jsonl --out out_dense.csv  --backend dense

# default — hybrid if artifacts present, else automatically falls back to tfidf
python src/rank.py --candidates ./candidates.jsonl --out out_auto.csv   --backend auto
```

(See the **Ranking modes** table below for what each does and when to use it.)

> No `git-lfs`? The `.npy` files arrive as pointer stubs and `rank.py`
> automatically falls back to the **TF-IDF** backend (still valid, fully offline).
> To restore the exact hybrid, run `python src/precompute_embeddings.py
> --candidates ./candidates.jsonl` to regenerate the artifacts, or use
> `--backend tfidf` to force the no-artifact mode.

## Live sandbox (demo)

**https://redrob-candidate-ranker-tonumoy.streamlit.app** (Streamlit Community
Cloud, deployed from this repo) — upload up to 100 candidate records (JSON array
or JSONL) or click **Load demo sample**; it runs the pipeline on CPU in seconds
and returns the ranked CSV in the official submission format. A mirror is on
HuggingFace Spaces (`https://bodhi108-redrob-candidate-ranker.hf.space`; Docker
config in [`space/`](space/)). The hosted demos use the **TF-IDF** backend for a
light, reliable footprint; the full **hybrid** runs from this repo
(`--backend hybrid`).

### Ranking modes (`--backend`)

We built and measured three semantic backends; pick any with `--backend`:

| Mode | What it does | When |
|---|---|---|
| **`tfidf`** | Pure scikit-learn TF-IDF over work-evidence. Exact keyword/phrase precision; **fully offline, no downloads**. | Air-gapped sandbox; the bulletproof fallback. |
| **`dense`** | Cosine over precomputed **bge-small** embeddings. Understands meaning/paraphrase, catches buzzword-free fits — but flattens, demoting keyword-strong elites. | Recall-first experiments. |
| **`hybrid`** (shipped) | `0.3·dense + 0.7·TF-IDF` (both min-max normalised). **TF-IDF precision at the top + dense recall at the margin** — keeps the strongest profile #1 *and* surfaces plain-language Tier-5s. | The submitted ranking; the JD's "hybrid retrieval" ideal. |
| **`auto`** (default) | hybrid if `artifacts/` present, else tfidf. | The single reproduce command. |

The dense embeddings are committed via **Git LFS** (`artifacts/*.npy`) so a clean
clone reproduces the hybrid offline. They are produced offline (GPU/network
allowed, outside the 5-min ranking budget) by:

```bash
python src/precompute_embeddings.py --candidates ./candidates.jsonl   # bge-small, ~3 min on a Colab T4
```

(or the Colab notebook in [`notebooks/`](notebooks/)). The live demo Space runs
the **TF-IDF** backend (kept lightweight so the sandbox stays reliably up); the
full **hybrid** is reproduced from this repo (`--backend hybrid`), and the
measured 3-mode comparison is in the deck.

## How it works (one paragraph)

We parse the JD into an auditable spec (`src/jd_spec.py`) of must-haves,
disqualifiers, traps and weights. Each candidate gets six structured features
in [0,1] — **semantic** JD↔work-evidence match (the recall engine for
"plain-language Tier-5" candidates who don't use buzzwords), **demonstrated
domain evidence** (IR / ranking / search / recsys work in their *career text*,
not their skill list), **experience-band fit**, **product-vs-services**,
**trust-gated skills** (stuffed skills with 0 endorsements/0 duration earn
nothing — the keyword-stuffer guard), and **location**. A **behavioural
availability multiplier** built from the 23 Redrob signals down-weights
perfect-on-paper-but-unavailable profiles. A **non-fit-title hard cap** keeps
keyword stuffers (HR/Sales/Marketing/etc. with AI skills) out of the top, a
**high-precision impossibility check** (`src/validation.py`) hard-zeros
honeypots, and an **abroad-and-won't-relocate penalty** drops profiles the JD
can't actually hire (no visa sponsorship). Reasoning is **fact-grounded and
deterministic** (no LLM) — it only
ever cites fields that exist on the profile, names the JD axis matched, and
surfaces a real concern, so it passes the Stage-4 no-hallucination checks.

## Layout

```
src/jd_spec.py              JD decoded as data: weights, skills, traps, locations
src/features.py             structured features + penalties + availability
src/semantic.py             dense (precomputed) OR TF-IDF semantic backend
src/validation.py           high-precision honeypot / impossibility detector
src/scoring.py              combiner -> final score
src/reasoning.py            fact-grounded, varied, no-LLM reasoning
src/rank.py                 THE reproduce command
src/precompute_embeddings.py  offline dense artifacts (optional)
app.py                      Streamlit sandbox app (CPU, TF-IDF)
demo_sample.jsonl           24 curated records for the sandbox "Load demo sample"
eval/local_eval.py          honeypot/non-fit-rate + top-10 inspection harness
notebooks/                  Colab GPU notebook to regenerate the dense artifacts
space/                      HF Space Docker deployment config (Dockerfile, deps)
tests/test_format.py        validator-invariant test
validate_submission.py      organiser's format validator (copied from bundle)
submission_metadata.yaml    portal metadata mirror
```

## Compute & constraints

CPU-only · no network during ranking · ≤ 5 min · ≤ 16 GB · top-100 only.
Measured: **~100 s, < 8 GB RAM, no network** on the full 100k pool (TF-IDF
backend). AI tools used in development are declared honestly in
`submission_metadata.yaml`.
