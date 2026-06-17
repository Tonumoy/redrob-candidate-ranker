"""Format invariants required by validate_submission.py — run after every change."""
import csv, re, subprocess, sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAT = re.compile(r"^CAND_[0-9]{7}$")

def test_pipeline_produces_valid_structure():
    out = os.path.join(ROOT, "sample_out.csv")
    subprocess.run([sys.executable, os.path.join(ROOT, "src", "rank.py"),
                    "--candidates", os.path.join(ROOT, "sample.jsonl"),
                    "--out", out, "--top_k", "50"], check=True)
    rows = list(csv.reader(open(out)))
    hdr, data = rows[0], rows[1:]
    assert hdr == ["candidate_id", "rank", "score", "reasoning"]
    ids = [r[0] for r in data]; ranks = [int(r[1]) for r in data]; scores = [float(r[2]) for r in data]
    assert all(PAT.match(i) for i in ids)
    assert sorted(ranks) == list(range(1, len(data) + 1))
    assert len(set(ids)) == len(ids)
    assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
    for i in range(len(scores)-1):
        if scores[i] == scores[i+1]:
            assert ids[i] < ids[i+1]

if __name__ == "__main__":
    test_pipeline_produces_valid_structure()
    print("OK: format invariants hold")
