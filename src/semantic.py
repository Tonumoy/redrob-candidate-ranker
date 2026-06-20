"""
semantic.py
===========
Semantic JD<->candidate match. Two interchangeable backends:

1. DENSE (preferred, higher quality): uses precomputed candidate embeddings
   + JD embedding from artifacts/ (produced offline by precompute_embeddings.py
   with a local sentence-transformer). Loaded as numpy — no network, no model
   load, no GPU at ranking time. This is the "plain-language Tier-5" recall engine.

2. TF-IDF (always-available fallback): pure scikit-learn over evidence text.
   Needs no downloads, runs in seconds on CPU, fully reproducible in any sandbox.
   rank.py uses this automatically if the dense artifacts are absent, so the
   submission ALWAYS reproduces within the 5-min/no-network/CPU budget.

Both return a per-candidate similarity in [0,1] aligned to the input order.
"""
import os
import numpy as np

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts")


def _minmax(x):
    x = np.asarray(x, dtype="float64")
    lo, hi = np.nanmin(x), np.nanmax(x)
    if hi - lo < 1e-9:
        return np.full_like(x, 0.5)
    return (x - lo) / (hi - lo)


def dense_available():
    return (os.path.exists(os.path.join(ART, "candidate_embeddings.npy"))
            and os.path.exists(os.path.join(ART, "candidate_ids.txt"))
            and os.path.exists(os.path.join(ART, "jd_embedding.npy")))


def dense_similarity(cand_ids):
    """Cosine(JD, candidate) from precomputed artifacts, reindexed to cand_ids."""
    emb = np.load(os.path.join(ART, "candidate_embeddings.npy"))
    ids = [l.strip() for l in open(os.path.join(ART, "candidate_ids.txt"))]
    jd = np.load(os.path.join(ART, "jd_embedding.npy")).astype("float64").ravel()
    idx = {cid: i for i, cid in enumerate(ids)}
    emb = emb.astype("float64")
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    jd /= (np.linalg.norm(jd) + 1e-9)
    sims = emb @ jd
    out = np.array([sims[idx[c]] if c in idx else 0.0 for c in cand_ids])
    return _minmax(out)


def tfidf_similarity(evidence_texts, jd_query):
    """Cosine in TF-IDF space between JD query and each candidate evidence text."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                          max_features=50000, sublinear_tf=True, min_df=2)
    mat = vec.fit_transform(evidence_texts + [jd_query])
    jd_vec = mat[-1]
    cand = mat[:-1]
    sims = linear_kernel(jd_vec, cand).ravel()
    return _minmax(sims)


def hybrid_similarity(cand_ids, evidence_texts, jd_query, w_dense=0.3):
    """Fuse dense (semantic meaning, catches plain-language/paraphrase fits) with
    TF-IDF (exact lexical precision, keeps keyword-strong elites at the top). Both
    components are min-max normalised to [0,1], so a convex blend is well-scaled.
    Lexical-led by default (w_dense=0.3): TF-IDF governs the very top, dense adds
    recall at the margin. Requires the dense artifacts; callers fall back to
    tfidf_similarity when they are absent."""
    dense = dense_similarity(cand_ids)
    tfidf = tfidf_similarity(evidence_texts, jd_query)
    return w_dense * dense + (1.0 - w_dense) * tfidf
