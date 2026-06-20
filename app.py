"""
app.py — Redrob Candidate Ranker sandbox (HuggingFace Spaces / Streamlit Cloud).

Satisfies submission_spec Section 10.5: accepts a small candidate sample (<=100),
runs the ranking system end-to-end on CPU within budget, and shows the ranked CSV.
Uses the TF-IDF backend (pure scikit-learn) so it needs no model download or
network and finishes in seconds.

Deploy on HF Spaces (SDK: streamlit) with this file at the repo root next to src/.
"""
import os, sys, json, io, csv, time
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import features as F, semantic as S, scoring as SC, reasoning as R, jd_spec as J

st.set_page_config(page_title="Redrob Candidate Ranker", page_icon="🎯", layout="wide")


@st.cache_resource
def _load_st_model(name="BAAI/bge-small-en-v1.5"):
    """Load the sentence-transformer once (cached). Used for the dense/hybrid modes
    so they work on arbitrary uploads (not just the precomputed pool)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)


with st.sidebar:
    st.header("How it ranks")
    st.markdown(
        "An **interpretable hybrid ranker** for the *Senior AI Engineer — Founding "
        "Team* JD. Each candidate is scored on:\n"
        "- **Semantic match** (TF-IDF) of the JD against *work evidence* (career "
        "history + summary), not the raw skill list — catches plain-language fits.\n"
        "- **Demonstrated domain work** (IR / ranking / search / recsys in the "
        "career text).\n"
        "- **Experience band** (6–8 yr sweet spot), **product-vs-services**, "
        "**trust-gated skills**, **location**.\n"
        "- A **behavioural availability** multiplier from the 23 Redrob signals.\n\n"
        "Guards: a **non-fit-title hard cap** sinks keyword stuffers (HR/Sales/"
        "Marketing with AI skills), a **high-precision honeypot check** zeroes "
        "impossible profiles, and an **abroad + won't-relocate** penalty drops "
        "unhirable profiles. Reasoning is **fact-grounded** (no LLM, no hallucination)."
    )
    st.caption("CPU-only · interpretable · fact-grounded reasoning. "
               "Pick a **Ranking mode** below to compare backends.")

st.title("🎯 Intelligent Candidate Ranker — India.RUNS Track 1")
st.caption(
    "Upload up to 100 candidate records (JSON array or JSONL), or load the bundled "
    "demo sample. Runs the full ranking pipeline on CPU and returns the ranked CSV "
    "in the official submission format."
)

c1, c2 = st.columns([3, 1])
with c1:
    # No `type=` filter on purpose: .jsonl has no standard MIME type and some
    # browsers reject it client-side. We accept any file and validate content below.
    up = st.file_uploader("Candidate sample (JSON array or JSONL — one object per line)")
with c2:
    st.write("")
    st.write("")
    demo = st.button("Load demo sample", use_container_width=True)
k = st.slider("Top K to display", 5, 100, 25)

MODES = {
    "Hybrid — embeddings + keywords (recommended)":
        "What we submit: 0.3·dense (bge-small) + 0.7·TF-IDF. Keyword precision at "
        "the top + semantic recall at the margin — keeps the strongest profile #1 "
        "and still catches plain-language fits.",
    "TF-IDF — keywords only (fast, fully offline)":
        "Pure scikit-learn keyword/phrase matching over work evidence. No model "
        "download, runs in milliseconds. The bulletproof air-gapped fallback.",
    "Dense — embeddings only (semantic)":
        "Cosine over bge-small embeddings. Understands meaning/paraphrase and "
        "catches buzzword-free fits, but flattens the top (can demote the "
        "strongest keyword-rich profile).",
}
mode = st.radio("Ranking mode", list(MODES.keys()),
                help="Switch backends and watch the ranking change on your own upload.")
st.caption(MODES[mode])

cands = None
if up is not None:
    try:
        raw = up.read().decode("utf-8", errors="replace").strip()
        try:
            data = json.loads(raw)                      # JSON array (or single object)
            cands = data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            cands = [json.loads(l) for l in raw.splitlines() if l.strip()]  # JSONL
        if not cands:
            st.warning("No candidate records found in that file.")
            cands = None
    except Exception as e:
        st.error(f"Could not parse the uploaded file ({e}). "
                 "Upload a JSON array or JSONL (one JSON object per line).")
        cands = None
elif demo:
    for name in ("demo_sample.jsonl", "sample.jsonl"):
        p = os.path.join(os.path.dirname(__file__), name)
        if os.path.exists(p):
            cands = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
            break

if cands:
    cands = cands[:100]
    t0 = time.time()
    ev = [F.evidence_text(c) for c in cands]
    if mode.startswith("TF-IDF"):
        sem = S.tfidf_similarity(ev, J.JD_QUERY)
        backend_label = "TF-IDF (keywords)"
    else:
        try:
            with st.spinner("Loading bge-small embeddings (first run downloads ~130 MB)…"):
                model = _load_st_model()
            dense = S.encode_live(model, ev, J.JD_QUERY)
            if mode.startswith("Dense"):
                sem, backend_label = dense, "Dense (bge-small)"
            else:
                tfidf = S.tfidf_similarity(ev, J.JD_QUERY)
                sem = J.HYBRID_W_DENSE * dense + (1 - J.HYBRID_W_DENSE) * tfidf
                backend_label = f"Hybrid ({J.HYBRID_W_DENSE}·dense + {round(1-J.HYBRID_W_DENSE,2)}·TF-IDF)"
        except Exception:
            st.info(
                "ℹ️ **Dense / Hybrid** use the bge-small embedding model, which isn't "
                "bundled in this lightweight live demo (kept lean so the sandbox stays "
                "reliably up). The full **hybrid** is what we submit — reproduce it from "
                "the repo with `python src/rank.py --backend hybrid` (embeddings shipped "
                "via Git LFS), and see the deck for the measured 3-mode comparison. "
                "Showing the **TF-IDF** ranking here.")
            sem = S.tfidf_similarity(ev, J.JD_QUERY)
            backend_label = "TF-IDF (demo)"
    scored = SC.score_all(cands, sem)
    scored.sort(key=lambda x: (-x[2], x[0]["candidate_id"]))
    top = scored[:k]
    raw_max = max((s for _, _, s in top), default=1.0) or 1.0

    rows, csv_rows = [], []
    for i, (c, comp, s) in enumerate(top):
        p = c["profile"]
        score = round(s * 0.99 / raw_max, 6)
        reasoning = R.build_reasoning(c, comp, i + 1, k)
        flag = "honeypot" if comp.get("is_honeypot") else ("non-fit title" if comp.get("title_nonfit") else "")
        rows.append({
            "rank": i + 1, "candidate_id": c["candidate_id"], "score": score,
            "title": p.get("current_title", ""), "yrs": p.get("years_of_experience", ""),
            "company": p.get("current_company", ""), "location": p.get("location", ""),
            "flag": flag, "reasoning": reasoning,
        })
        csv_rows.append({"candidate_id": c["candidate_id"], "rank": i + 1,
                         "score": score, "reasoning": reasoning})
    dt = time.time() - t0

    n_hp = sum(1 for _, comp, _ in top if comp.get("is_honeypot"))
    n_nf = sum(1 for _, comp, _ in top if comp.get("title_nonfit"))
    st.success(f"Ranked **{len(cands)}** candidates → top **{k}** in **{dt:.2f}s** "
               f"(CPU · **{backend_label}**).")
    st.caption(f"In the shown top-{k}: honeypots = {n_hp} · non-fit titles = {n_nf} "
               "(both should be ~0 at the top — the traps sink to the bottom).")
    st.dataframe(rows, use_container_width=True, hide_index=True)

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["candidate_id", "rank", "score", "reasoning"])
    w.writeheader(); w.writerows(csv_rows)
    st.download_button("⬇ Download ranked CSV (submission format)", buf.getvalue(),
                       file_name="submission_preview.csv", mime="text/csv")
else:
    st.info("Upload a file or click **Load demo sample** to see the ranker in action.")
