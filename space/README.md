---
title: Redrob Candidate Ranker
emoji: 🎯
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 8501
pinned: false
license: mit
short_description: Interpretable hybrid candidate ranker for Track-1
---

# Redrob Intelligent Candidate Ranker — India.RUNS Track 1 (sandbox)

Live sandbox for the **Senior AI Engineer — Founding Team** ranking system.
Upload up to 100 candidate records (JSON array or JSONL), or click **Load demo
sample**, and the app runs the full ranking pipeline on **CPU, no network**, in
seconds, returning the ranked CSV in the official submission format.

## How it ranks (interpretable hybrid)

Each candidate is scored on six structured features — a **TF-IDF semantic match**
of the JD against *work evidence* (career history + summary, not the raw skill
list), **demonstrated domain work** (IR / ranking / search / recsys), **experience
band** (6–8 yr), **product-vs-services**, **trust-gated skills**, and **location**
— combined with a **behavioural availability** multiplier from the 23 Redrob
signals. Guards: a **non-fit-title hard cap** (keyword stuffers), a **high-precision
honeypot check** (impossible profiles), and an **abroad + won't-relocate** penalty.
Reasoning is **fact-grounded** (deterministic, no LLM, hallucination-proof).

The same code path produces the full 100k → top-100 submission via
`python src/rank.py --candidates candidates.jsonl --out submission.csv`.

Full project + methodology: see the GitHub repository.
