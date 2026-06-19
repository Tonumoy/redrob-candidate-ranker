# Redrob Intelligent Candidate Ranker — India.RUNS Track 1

Ranks the top-100 candidates from the 100,000-candidate pool for the released
**Senior AI Engineer — Founding Team** JD. Built for the Data & AI Challenge.

## Reproduce the submission (single command)

```bash
pip install -r requirements.txt
python src/rank.py --candidates ./candidates.jsonl --out ./submission.csv
python validate_submission.py ./submission.csv   # -> "Submission is valid."
```

Runs **CPU-only, no network, well under the 5-min / 16 GB budget** — measured
**~100 s in under 8 GB RAM** on a 2-core Intel i7-7500U laptop. `candidates.jsonl`
or `candidates.jsonl.gz` both work.

## Live sandbox (demo)

**https://huggingface.co/spaces/Bodhi108/redrob-candidate-ranker** — upload up to
100 candidate records (JSON array or JSONL) or click **Load demo sample**; it
runs the full pipeline on CPU in seconds and returns the ranked CSV in the
official submission format. The HF Space Docker config lives in [`space/`](space/).

### Optional dense upgrade (offline pre-computation)

```bash
python src/precompute_embeddings.py --candidates ./candidates.jsonl
```

Writes dense artifacts to `artifacts/`. `rank.py` uses them automatically if
present; otherwise it falls back to a pure scikit-learn **TF-IDF** semantic
backend (no downloads), so the ranking step always reproduces inside the
Stage-3 sandbox.

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
