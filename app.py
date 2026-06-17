"""
app.py — Sandbox / demo (HuggingFace Spaces or Streamlit Cloud).

Satisfies submission_spec Section 10.5: accepts a small candidate sample
(≤100), runs the ranking system end-to-end on CPU within the budget, and
shows the ranked CSV. Uses the TF-IDF backend so it needs no model download.

Deploy on HF Spaces (SDK: streamlit) or Streamlit Cloud. Put this file at repo
root or adjust the sys.path import below.
"""
import os, sys, json, io
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import features as F, semantic as S, scoring as SC, reasoning as R, jd_spec as J

st.set_page_config(page_title="Redrob Candidate Ranker", layout="wide")
st.title("Intelligent Candidate Ranker — India.RUNS Track 1")
st.caption("Upload up to 100 candidate records (JSON array or JSONL). "
           "Runs the full ranking pipeline on CPU and returns the ranked CSV.")

up = st.file_uploader("Candidate sample (.json / .jsonl)", type=["json", "jsonl"])
k = st.slider("Top K", 5, 100, 25)

if up:
    raw = up.read().decode("utf-8").strip()
    try:
        cands = json.loads(raw)
        if isinstance(cands, dict):
            cands = [cands]
    except json.JSONDecodeError:
        cands = [json.loads(l) for l in raw.splitlines() if l.strip()]
    cands = cands[:100]
    st.write(f"Loaded **{len(cands)}** candidates. Ranking…")

    ev = [F.evidence_text(c) for c in cands]
    sem = S.tfidf_similarity(ev, J.JD_QUERY)
    scored = SC.score_all(cands, sem)
    scored.sort(key=lambda x: (-x[2], x[0]["candidate_id"]))
    top = scored[:k]
    raw_max = max((s for _, _, s in top), default=1.0) or 1.0
    rows = [{"candidate_id": c["candidate_id"], "rank": i + 1,
             "score": round(s * 0.99 / raw_max, 6),
             "reasoning": R.build_reasoning(c, comp, i + 1, k)}
            for i, (c, comp, s) in enumerate(top)]
    st.dataframe(rows, use_container_width=True)
    buf = io.StringIO()
    import csv
    w = csv.DictWriter(buf, fieldnames=["candidate_id", "rank", "score", "reasoning"])
    w.writeheader(); w.writerows(rows)
    st.download_button("Download ranked CSV", buf.getvalue(),
                       file_name="submission_preview.csv", mime="text/csv")
