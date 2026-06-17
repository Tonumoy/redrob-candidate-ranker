#!/usr/bin/env python3
"""
precompute_embeddings.py  (OFFLINE — may use network/GPU, may exceed 5 min)
===========================================================================
Produces the dense artifacts consumed by rank.py at ranking time:

    artifacts/candidate_embeddings.npy   float32 [N, d], L2 order matches candidate_ids.txt
    artifacts/candidate_ids.txt          one CAND_ id per line, aligned to rows above
    artifacts/jd_embedding.npy           float32 [d] for the JD query
    artifacts/model_name.txt             the model used (for reproducibility)

Run once, locally, before the final rank.py run:

    python src/precompute_embeddings.py --candidates ./candidates.jsonl

Model: a small, strong, *local* sentence encoder. BAAI/bge-small-en-v1.5 (384-d)
is a good speed/quality balance on CPU; all-MiniLM-L6-v2 is a lighter fallback.
The model is cached locally so rank.py never needs network. If you skip this
step, rank.py automatically uses the TF-IDF backend (still fully valid).
"""
import argparse, gzip, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import jd_spec as J
import features as F

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts")


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)  # downloads once, then cached locally

    cands = list(load(args.candidates))
    ids = [c["candidate_id"] for c in cands]
    texts = [F.evidence_text(c) for c in cands]

    emb = model.encode(texts, batch_size=args.batch, show_progress_bar=True,
                       normalize_embeddings=True).astype("float32")
    jd = model.encode([J.JD_QUERY], normalize_embeddings=True).astype("float32")[0]

    os.makedirs(ART, exist_ok=True)
    np.save(os.path.join(ART, "candidate_embeddings.npy"), emb)
    np.save(os.path.join(ART, "jd_embedding.npy"), jd)
    with open(os.path.join(ART, "candidate_ids.txt"), "w") as f:
        f.write("\n".join(ids))
    with open(os.path.join(ART, "model_name.txt"), "w") as f:
        f.write(args.model)
    print(f"[precompute] {emb.shape[0]} embeddings dim={emb.shape[1]} -> {ART}")


if __name__ == "__main__":
    main()
