#!/usr/bin/env python3
"""
local_eval.py — offline sanity harness.

The TRUE ground truth (relevance tiers) is hidden until results are announced,
so we cannot compute the real NDCG. Instead this harness gives defensible,
self-consistent checks you CAN run locally, plus a *weak proxy* ranking built
from the JD's own stated rules to confirm your model agrees with the JD logic.

Run:
    python eval/local_eval.py --candidates ./candidates.jsonl --submission ./submission.csv

Checks:
  1. Spec compliance (delegates to validate_submission.py expectations).
  2. Honeypot rate in top-100  (DQ if > 10%).
  3. Non-fit-title rate in top-100 (should be ~0%).
  4. Top-10 inspection dump (manual eyeball: are these obviously the best fits?).
  5. Weak-proxy agreement: Spearman-style overlap between your top-100 and a
     transparent rule-only ranking (a coherence signal, NOT an accuracy claim).
"""
import argparse, csv, json, gzip, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import features as F, scoring as SC, semantic as S, jd_spec as J
from validation import check_impossible


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--submission", required=True)
    args = ap.parse_args()

    cands = {c["candidate_id"]: c for c in load(args.candidates)}
    sub = list(csv.DictReader(open(args.submission)))
    top_ids = [r["candidate_id"] for r in sub]

    # 2. honeypot rate
    hp = sum(1 for cid in top_ids if cid in cands and check_impossible(cands[cid])[0])
    print(f"Honeypot rate in top-{len(top_ids)}: {hp} ({100*hp/len(top_ids):.1f}%)  "
          f"{'OK' if hp/len(top_ids) <= 0.10 else 'DQ RISK'}")

    # 3. non-fit titles
    nf = sum(1 for cid in top_ids if cid in cands and F.title_nonfit(cands[cid]))
    print(f"Non-fit titles in top-{len(top_ids)}: {nf} ({100*nf/len(top_ids):.1f}%)")

    # 4. top-10 dump
    print("\n--- TOP 10 (eyeball these) ---")
    for r in sub[:10]:
        c = cands.get(r["candidate_id"], {})
        p = c.get("profile", {})
        print(f"  #{r['rank']:>3} {r['candidate_id']}  {p.get('current_title','?')[:28]:28} "
              f"{p.get('years_of_experience','?')}y  {p.get('current_company','?')[:18]:18} "
              f"{p.get('location','?')[:14]}")

    # 5. weak-proxy coherence
    allc = list(cands.values())
    ev = [F.evidence_text(c) for c in allc]
    sem = S.tfidf_similarity(ev, J.JD_QUERY)
    scored = SC.score_all(allc, sem)
    scored.sort(key=lambda x: (-x[2], x[0]["candidate_id"]))
    proxy_top = set(c["candidate_id"] for c, _, _ in scored[:len(top_ids)])
    overlap = len(proxy_top & set(top_ids)) / len(top_ids)
    print(f"\nWeak-proxy top-{len(top_ids)} overlap with submission: {100*overlap:.0f}% "
          "(coherence signal only)")


if __name__ == "__main__":
    main()
