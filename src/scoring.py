"""
scoring.py
==========
Combine structured features + semantic similarity + behavioural availability
+ honeypot safety net into one final score per candidate.

final = base_fit * (1 - penalties) * availability_multiplier
        with hard caps for non-fit titles and hard-zero for honeypots.

`semantic_scores` is a precomputed array (dense or TF-IDF) aligned to `cands`.
"""
import jd_spec as J
import features as F
from validation import check_impossible


def score_all(cands, semantic_scores):
    rows = []
    for c, sem in zip(cands, semantic_scores):
        comp = {}
        comp["semantic"] = float(sem)
        comp["domain_evidence"] = F.f_domain_evidence(c)
        comp["experience"] = F.f_experience(c)
        comp["product"] = F.f_product(c)
        comp["skills_trust"] = F.f_skills_trust(c)
        comp["location"] = F.f_location(c)

        base = (J.WEIGHTS["semantic"] * comp["semantic"]
                + J.WEIGHTS["domain_evidence"] * comp["domain_evidence"]
                + J.WEIGHTS["experience"] * comp["experience"]
                + J.WEIGHTS["product"] * comp["product"]
                + J.WEIGHTS["skills_trust"] * comp["skills_trust"]
                + J.WEIGHTS["location"] * comp["location"])

        pen, pen_reasons = F.f_penalties(c)
        avail, avail_notes = F.availability_multiplier(c)
        comp["penalty"] = pen
        comp["penalty_reasons"] = pen_reasons
        comp["avail"] = avail
        comp["avail_notes"] = avail_notes

        score = base * (1.0 - pen) * avail

        # Non-fit title hard cap (keyword-stuffer guard).
        comp["title_nonfit"] = F.title_nonfit(c)
        if comp["title_nonfit"]:
            score = min(score, J.NONFIT_TITLE_CAP * avail)

        # Honeypot safety net (high-precision impossibility).
        is_hp, hp_reasons = check_impossible(c)
        comp["is_honeypot"] = is_hp
        comp["honeypot_reasons"] = hp_reasons
        if is_hp:
            score = base * 0.001  # forced to the floor; never reaches top-100

        comp["base"] = base
        comp["final"] = score
        rows.append((c, comp, score))
    return rows
