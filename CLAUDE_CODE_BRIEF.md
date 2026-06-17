# CLAUDE CODE BRIEF — India.RUNS Track 1 (Data & AI Challenge)
### Intelligent Candidate Discovery & Ranking — Redrob AI × Hack2skill

You (the local Claude Code session) are taking over a build that already has a
**validated, spec-compliant reference implementation** in `redrob-ranker/`.
Your job is to run it on the **full 100,000-candidate pool**, add the dense
embedding upgrade, tune, deploy the sandbox, generate the deck PDF, and ship a
clean public GitHub repo. Do **not** rewrite from scratch — extend and verify.

> **Anti-hallucination rule:** every fact below was read directly from the
> hackathon bundle (`submission_spec.docx`, `validate_submission.py`,
> `candidate_schema.json`, `job_description.docx`, `redrob_signals_doc.docx`,
> `sample_submission.csv`) and the live event page. Do not invent rules,
> deadlines, columns, or metrics. If something here conflicts with a file on
> disk, **the file on disk wins** — re-read it and flag the conflict.

---

## 0. Where things are

- Hackathon bundle (read-only reference): the unzipped folder you have open,
  containing `candidates.jsonl` (~465 MB, 100,000 lines) or
  `candidates.jsonl.gz`, plus `job_description.*`, `submission_spec.*`,
  `candidate_schema.json`, `redrob_signals_doc.*`, `sample_candidates.json`,
  `sample_submission.csv`, `validate_submission.py`,
  `submission_metadata_template.yaml`.
- Reference implementation: `redrob-ranker/` (this is the code you extend).
- Deck template (MANDATORY): `Idea_Submission_Template___Redrob.pptx`
  (11 slides — you must use this exact template, then export to PDF).

First action: `ls` the bundle, confirm `wc -l candidates.jsonl` prints
`100000`, and open 2–3 random records to confirm the schema matches
`candidate_schema.json`.

---

## 1. The task in one sentence

From 100,000 candidate profiles, output a CSV ranking the **top 100** for the
JD *Senior AI Engineer — Founding Team*, best-fit first, with a fact-grounded
1–2 sentence reasoning per candidate.

## 2. The output contract (from `validate_submission.py` — non-negotiable)

- Filename = your **registered Team ID** + `.csv` (e.g. `team_xxx.csv`).
- UTF-8. Header row **exactly**: `candidate_id,rank,score,reasoning`.
- **Exactly 100 data rows** (rows 2–101). Not 99, not 101.
- `candidate_id` matches `^CAND_[0-9]{7}$`, exists in `candidates.jsonl`, unique.
- `rank` = each integer 1..100 exactly once.
- `score` = float, **non-increasing** as rank increases (rank 1 highest).
- Score ties allowed, but ties must be broken by **candidate_id ascending**.
- `reasoning` = optional column but **strongly** scored at Stage 4 — keep it.

Run `python validate_submission.py <file>.csv` before every upload. The
reference `rank.py` already satisfies all of this (verified on the sample).

## 3. How it's scored (from `submission_spec` §4)

`composite = 0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10`
against a **hidden** ground truth with relevance tiers (tier 3+ = "relevant").
**NDCG@10 dominates (0.50) → the top 10 matter most.** Optimise top-of-list
precision above all. Scoring runs once, after close; **no live leaderboard**,
**3 submissions max**, last valid submission counts.

## 4. Constraints (from `submission_spec` §3) — design around these

| Limit | Value |
|---|---|
| Ranking-step runtime | ≤ 5 min wall-clock |
| RAM | ≤ 16 GB |
| Compute | **CPU only** during ranking |
| Network | **OFF** during ranking — no OpenAI/Anthropic/Gemini/any hosted LLM |
| Disk | ≤ 5 GB intermediate |

Pre-computation (embeddings) **may** exceed 5 min and use GPU — but the
`rank.py` step that emits the CSV must obey the table. Stage 3 reproduces your
ranking step in a sandbox matching these exactly; if it can't reproduce, you're
disqualified regardless of score. **The reference design already separates
offline precompute from the fast CPU ranking step — preserve that boundary.**

## 5. The matching philosophy (decoded from `job_description.docx`)

The JD *tells participants the traps*. Encode this, don't fight it.

**Strong positives (Tier-5 shape):**
- Shipped **embeddings retrieval / vector or hybrid search / ranking / search /
  recommendation / personalization** systems to real users.
- **Product-company** applied ML (not pure services, not pure research).
- **6–8 yrs** ideal (5–9 acceptable); strong **Python**; designed ranking
  **evaluation** (NDCG/MRR/MAP, A/B testing).
- Located in / will relocate to **Pune, Noida, Hyderabad, Mumbai, Delhi NCR,
  Bangalore**.
- **Available** (recently active, responsive, reasonable notice).

**Explicit traps in the dataset:**
1. **Keyword stuffers** — all AI skills listed but title is HR/Sales/Marketing/
   Content Writer/etc. → **NOT a fit** (hard-cap them).
2. **Plain-language Tier-5s** — *don't* use "RAG"/"Pinecone" words but career
   history shows they built a recommendation/search system at a product company
   → **ARE a fit** (semantic match must catch these — pure keyword match fails).
3. **Behavioural twins** — near-identical profiles separated only by signals.
4. **~80 honeypots** — impossible profiles (e.g. 8 yrs at a 3-yr-old company;
   "expert" in 10 skills with 0 months used) → forced to tier 0. **Honeypot
   rate > 10% in top-100 = DQ.**

**Disqualifiers/negatives:** research/academia-only (no production); recent
(<12 mo) LangChain+OpenAI only with no pre-LLM ML; "architect/tech-lead" with no
hands-on code in 18 mo; title-chasers (job-hop every ~1.5 yr); **entirely**
consulting-firm career (TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini/
Mindtree); CV/speech/robotics primary without NLP/IR; closed-source-only 5+ yr
with no external validation.

The real candidate distribution (sampled 3,000): titles are dominated by
**non-fit decoys** (HR Manager, Business Analyst, Mechanical/Civil Engineer,
Graphic Designer, Sales, Marketing, Content Writer, Accountant). Genuine fits
are **rare** — the JD says it expects "10 great matches, not 1000 maybes". So
be **conservative and evidence-demanding at the top 10**, looser in the tail.

## 6. Architecture (already implemented — extend it)

```
candidates.jsonl
   │
   ├─ (offline) precompute_embeddings.py ─► artifacts/{candidate_embeddings,jd_embedding}.npy
   │
rank.py (CPU, no network, <5min)
   ├─ semantic.py     dense cosine (if artifacts) else TF-IDF cosine   [plain-language recall]
   ├─ features.py     6 structured features + penalties + availability  [JD rules]
   ├─ validation.py   high-precision honeypot/impossibility net          [tier-0 guard]
   ├─ scoring.py      base·(1−penalty)·availability, title cap, hp zero  [combine]
   └─ reasoning.py    deterministic, fact-grounded, varied               [Stage-4 proof]
        └─► submission.csv (top 100)
```

Component weights live in `src/jd_spec.py` (`WEIGHTS`). **Tune there**, never
bury constants in logic.

## 7. Your execution plan (in order)

**Step 1 — Smoke test on the full pool with the TF-IDF backend.**
```bash
cd redrob-ranker
pip install -r requirements.txt
python src/rank.py --candidates /path/to/candidates.jsonl --out ./submission.csv
python validate_submission.py ./submission.csv      # must say "Submission is valid."
python eval/local_eval.py --candidates /path/to/candidates.jsonl --submission ./submission.csv
```
Confirm: 100 rows, honeypot rate 0%, non-fit-title rate ~0%, runtime < 5 min,
and **manually eyeball the top-10 dump** — they must be obviously-strong
product-company IR/ranking/ML engineers in-band and in-location. If a stuffer
or honeypot is in the top 10, fix the rule before proceeding.

**Step 2 — Add the dense upgrade (quality, especially for plain-language fits).**
```bash
python src/precompute_embeddings.py --candidates /path/to/candidates.jsonl
# model: BAAI/bge-small-en-v1.5 (384-d). GPU optional; this step may exceed 5 min.
python src/rank.py --candidates /path/to/candidates.jsonl --out ./submission.csv
```
`rank.py` auto-detects `artifacts/` and switches to dense cosine. Re-validate +
re-eyeball top-10. Compare dense vs TF-IDF top-100 overlap; keep whichever gives
the cleaner, more defensible top-10. **Ship dense if it clearly helps**, but
keep the TF-IDF fallback intact for sandbox reproducibility.

**Step 3 — Tune weights + thresholds (no leaderboard, so tune by judgment).**
- Inspect 30–50 top candidates and 30–50 just-below-cutoff candidates by hand.
- Adjust `WEIGHTS`, `NONFIT_TITLE_CAP`, availability bounds, honeypot thresholds
  in `jd_spec.py` / `validation.py` to make the top-10 unimpeachable and to keep
  honeypots + stuffers out. Re-run `local_eval.py` after each change.
- Because NDCG@10 carries half the score, spend most effort on positions 1–10.
- **Commit after each meaningful change** — Stage 4 checks for real git history,
  not a single dump. Make 8–20 small, message-ful commits across your session.

**Step 4 — Reasoning quality pass (Stage-4 manual review).**
Sample 10 random rows; verify each reasoning: cites real facts, names a JD axis,
admits a real concern where one exists, no skill/employer that isn't in the
profile, all 10 read differently, tone matches rank. Improve `reasoning.py`
templates if any check is weak. **Never** let it state a fact not present in the
candidate record.

**Step 5 — Deploy the sandbox (mandatory, `submission_spec` §10.5).**
- Easiest: HuggingFace Space (SDK: Streamlit) using `app.py`. Your HF account is
  `Bodhi108`. Create a public Space, push `app.py` + `src/` + `requirements.txt`,
  set it to accept ≤100-candidate uploads. Confirm it runs in < 5 min CPU.
- Alternative: a Google Colab notebook that runs end-to-end (you use Colab
  often) or Streamlit Cloud. Put the working URL in `submission_metadata.yaml`.

**Step 6 — Public GitHub repo.**
- Push the whole `redrob-ranker/` tree (README, src, tests, app, eval,
  `submission_metadata.yaml`, `validate_submission.py`, `requirements.txt`).
- **Do not commit** the 465 MB `candidates.jsonl` (gitignore it). Document the
  single reproduce command in README (already written).
- Fill `submission_metadata.yaml`: real Team ID, contact email/phone, repo URL,
  sandbox URL, compute summary, AI-tools = Claude (honest), precompute timing.
- Confirm the repo is **public** (the portal field requires public access).

**Step 7 — Build the deck PDF (mandatory template).**
Use `Idea_Submission_Template___Redrob.pptx` **exactly** (it's required). Fill the
11 slides — they map 1:1 to this architecture:
1. Team Name / Problem Statement (Track 1 – Intelligent Candidate Discovery) / Team Leader.
2. **Solution Overview** — interpretable hybrid ranker; differentiator = reads
   *work evidence + behaviour*, not keywords; explicitly beats the 4 traps.
3. **JD Understanding & Candidate Evaluation** — the decoded must-haves,
   disqualifiers, and the 23 signals; how we go beyond keywords (semantic +
   trust-gated skills + title cap).
4. **Ranking Methodology** — 6 features + availability multiplier + honeypot net;
   the combine formula; dense-or-TF-IDF backend.
5. **Explainability & Data Validation** — deterministic fact-grounded reasoning
   (no hallucination by construction); honeypot/impossibility checks.
6. **End-to-End Workflow** — JD → precompute → features+semantic → score → rank →
   reasoning → CSV.
7. **System Architecture** — the diagram in §6 above (draw it cleanly).
8. **Results & Performance** — top-10 examples + honeypot rate 0% + runtime/RAM
   vs the constraint table (CPU, <5 min, no network).
9. **Technologies Used** — Python, numpy, scikit-learn (TF-IDF), sentence-
   transformers (bge-small) offline, Streamlit/HF Spaces; *why each*.
10. **Submission Assets** — GitHub URL, sandbox URL, (optional) a short Loom/
    YouTube walkthrough video.
11. Closing.
Then **export to PDF** (the portal upload is a PDF, ≤ 5 MB). If you have the
`pptx` skill, use python-pptx to fill text frames in place without disturbing the
template's layout; then LibreOffice/`soffice --headless --convert-to pdf`.

**Step 8 — Final pre-flight before uploading (≤ 3 submissions total!).**
- [ ] `validate_submission.py` says valid; filename = Team ID `.csv`.
- [ ] top-10 eyeballed, defensible; honeypot rate 0%, no stuffers up top.
- [ ] `rank.py` reproduces in < 5 min on CPU with network off
      (`unshare -n` or disconnect to truly test the no-network path).
- [ ] repo public, README single-command works on a clean clone, metadata filled.
- [ ] sandbox link live and runs a small sample.
- [ ] deck PDF uses the mandatory template, ≤ 5 MB.
- Upload via the dashboard: select the Track-1 challenge in the dropdown, paste
  the **public GitHub URL**, upload the **deck PDF**, upload the **ranked CSV**.

## 8. Deadlines (verify on YOUR dashboard — it is authoritative)

- The submission portal showed **Ending Date 02 Jul 2026, 23:59 IST** for the
  Data & AI Challenge (countdown ~14 days from 18 Jun). The public marketing
  timeline said 28 Jun. **Treat the earlier date (28 Jun) as your safe target**
  and confirm the live countdown on your dashboard. Finish with days to spare —
  you only get 3 submissions and there's no leaderboard to iterate against.

## 9. Defendability (Stages 4–5 — you must be able to defend this live)

Everything is interpretable on purpose. Be ready to explain: why semantic +
structured + behavioural (not one big model); why the title hard-cap is the
decisive anti-stuffer lever; why honeypot detection is high-precision not
high-recall; why NDCG@10 drove your top-of-list conservatism; the dense-vs-
TF-IDF tradeoff; and how reasoning is hallucination-proof by construction. Keep
the git history real (incremental commits). Declare Claude usage honestly — it's
permitted and not penalised; contradicting your declaration is what's penalised.

---

### TL;DR for the session
Run `rank.py` on the full pool → validate → eyeball top-10 → add dense
embeddings → tune weights in `jd_spec.py` → polish reasoning → deploy HF Space →
push public repo → fill the mandatory deck → export PDF → pre-flight checklist →
submit on the dashboard. Extend the reference code; don't rebuild it. Re-read the
bundle files if anything here is unclear — the files on disk are the source of truth.
