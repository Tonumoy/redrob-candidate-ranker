#!/usr/bin/env python3
"""
rank.py — THE ranking step (single reproduce command).

    python src/rank.py --candidates ./candidates.jsonl --out ./submission.csv

Constraints honoured: CPU-only, no network, ≤5 min wall-clock, ≤16 GB RAM.
Streams the JSONL; uses precomputed dense embeddings if artifacts/ are present,
otherwise an always-available TF-IDF semantic backend (so it reproduces anywhere).

Output CSV is validated by the bundled validator:
    candidate_id,rank,score,reasoning   |   exactly 100 rows
    ranks 1..100 unique, scores non-increasing, ties broken by candidate_id asc.
"""
import argparse, csv, gzip, io, json, os, sys, time

sys.path.insert(0, os.path.dirname(__file__))
import jd_spec as J
import features as F
import semantic as S
import scoring as SC
import reasoning as R


def load_candidates(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top_k", type=int, default=J.TOP_K)
    ap.add_argument("--backend", choices=["auto", "tfidf", "dense", "hybrid"],
                    default="auto",
                    help="semantic backend: auto = hybrid when dense artifacts are "
                         "present, else tfidf. tfidf is fully offline/no-download.")
    args = ap.parse_args()

    t0 = time.time()
    cands = list(load_candidates(args.candidates))
    print(f"[rank] loaded {len(cands)} candidates in {time.time()-t0:.1f}s", file=sys.stderr)

    ids = [c["candidate_id"] for c in cands]
    ev = [F.evidence_text(c) for c in cands]

    # --- semantic backend selection (robust: any dense failure -> tfidf) ---
    want = args.backend
    sem, backend = None, None
    if want != "tfidf" and S.dense_available():
        try:
            if want == "dense":
                sem = S.dense_similarity(ids)
                backend = "dense(precomputed)"
            else:  # hybrid, or auto with artifacts present
                sem = S.hybrid_similarity(ids, ev, J.JD_QUERY, J.HYBRID_W_DENSE)
                backend = f"hybrid(dense*{J.HYBRID_W_DENSE}+tfidf*{round(1-J.HYBRID_W_DENSE,2)})"
        except Exception as e:
            # e.g. artifacts present only as Git-LFS pointer stubs, or corrupt.
            print(f"[rank] dense artifacts unusable ({e}); using tfidf", file=sys.stderr)
            sem = None
    if sem is None:
        sem = S.tfidf_similarity(ev, J.JD_QUERY)
        backend = "tfidf"
        if want in ("dense", "hybrid"):
            backend += f" (requested {want}; no usable dense artifacts/)"
    print(f"[rank] semantic backend = {backend} ({time.time()-t0:.1f}s)", file=sys.stderr)

    # --- score everyone ---
    scored = SC.score_all(cands, sem)
    print(f"[rank] scored {len(scored)} ({time.time()-t0:.1f}s)", file=sys.stderr)

    # --- select & order: score desc, tie-break candidate_id asc ---
    scored.sort(key=lambda x: (-x[2], x[0]["candidate_id"]))
    top = scored[:args.top_k]

    # Normalize selected scores to (0,1] (max -> 0.99) for clean, comparable output.
    raw_max = max((s for _, _, s in top), default=1.0) or 1.0
    norm = 0.99 / raw_max

    # Guarantee non-increasing scores after rounding; keep tie-break by id asc.
    rows = []
    for rank, (c, comp, score) in enumerate(top, start=1):
        rows.append([c["candidate_id"], rank, round(float(score) * norm, 6),
                     R.build_reasoning(c, comp, rank, args.top_k)])
    # enforce monotonicity defensively (rounding can't create an increase here
    # because we sorted on raw score, but clamp to be safe)
    for i in range(1, len(rows)):
        if rows[i][2] > rows[i-1][2]:
            rows[i][2] = rows[i-1][2]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        w.writerows(rows)

    # diagnostics
    n_hp = sum(1 for _, comp, _ in top if comp.get("is_honeypot"))
    n_nonfit = sum(1 for _, comp, _ in top if comp.get("title_nonfit"))
    print(f"[rank] wrote {len(rows)} rows to {args.out} in {time.time()-t0:.1f}s", file=sys.stderr)
    print(f"[rank] honeypots in top-{args.top_k}: {n_hp} | non-fit titles: {n_nonfit}", file=sys.stderr)


if __name__ == "__main__":
    main()
