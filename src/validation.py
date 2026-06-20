"""
validation.py
=============
High-PRECISION impossibility detection (honeypots) + data-quality flags.

Philosophy: we only hard-zero a profile when it is *genuinely impossible*,
because over-flagging demotes real Tier-5 candidates. We do NOT need perfect
honeypot recall — honeypots are weak on every other axis and a good scorer
avoids them anyway. The hard-zero is a safety net to keep the top-100 honeypot
rate at ~0% (DQ threshold is >10%).

Empirically validated on this pool: this detector flags ~23/100,000 profiles, all
genuinely impossible (e.g. 75 months of sequential role tenure vs 32 months of
stated experience) -- which catches a fit-shaped honeypot (Search Engineer that
otherwise scored into the top-100 at #53) plus the obviously-broken non-fit ones.
Crucially we do NOT treat "skill duration_months > career length" as impossible:
that pattern
holds for ~30% of the top-100 (skill durations are sampled up to ~96 months
independent of years_of_experience), so it is dataset noise, not a honeypot
signature — flagging it would demote a third of the strongest real candidates.
No top-100 profile carries the "expert skill with 0 months used" signature.

Returns (is_honeypot: bool, reasons: list[str]).
"""
from datetime import date


def _parse(d):
    if not d:
        return None
    try:
        y, m, day = map(int, d.split("-"))
        return date(y, m, day)
    except Exception:
        return None


def check_impossible(c):
    reasons = []
    p = c.get("profile", {})
    yoe = float(p.get("years_of_experience", 0) or 0)
    yoe_m = yoe * 12.0
    ch = c.get("career_history", []) or []
    skills = c.get("skills", []) or []

    hard = 0   # genuinely impossible
    soft = 0   # strongly suspicious

    # --- HARD: chronological impossibilities in career history.
    sum_dur = 0
    for r in ch:
        sd, ed = _parse(r.get("start_date")), _parse(r.get("end_date"))
        dur = r.get("duration_months", 0) or 0
        sum_dur += dur
        if sd and ed and ed < sd:
            hard += 1
            reasons.append(f"role end_date < start_date at {r.get('company')}")
        if sd and ed:
            real = (ed.year - sd.year) * 12 + (ed.month - sd.month)
            if abs(real - dur) > 18:  # duration_months wildly disagrees with dates
                soft += 1
                reasons.append(f"duration_months {dur} != dates ~{real} at {r.get('company')}")

    # --- HARD: total role tenure far exceeds stated experience. You cannot work
    # substantially more months than your career is long. Measured on this pool,
    # legitimate candidates sit at sum_dur/YoE ~= 1.00 (99th percentile = 1.00);
    # honeypots that lowered years_of_experience while leaving the career history
    # intact spike well above 1.5x. Threshold 1.5x + 30-month excess is therefore
    # high-precision (zero legitimate top-candidate false positives observed).
    if yoe_m > 0 and sum_dur > yoe_m * 1.5 and (sum_dur - yoe_m) > 30:
        hard += 1
        reasons.append(f"sum role tenure {sum_dur}m >> stated YoE ({yoe_m:.0f}m) -- impossible")

    # --- HARD: "expert in many skills with zero usage" honeypot signature.
    expert_zero = sum(1 for s in skills
                      if s.get("proficiency") == "expert"
                      and (s.get("duration_months", 0) or 0) == 0
                      and (s.get("endorsements", 0) or 0) == 0)
    if expert_zero >= 3:
        hard += 1
        reasons.append(f"{expert_zero} 'expert' skills with 0 months & 0 endorsements")

    # --- SOFT: a skill used longer than the entire (career + 2yr grace) period.
    overlong = sum(1 for s in skills
                   if (s.get("duration_months", 0) or 0) > yoe_m + 24)
    if overlong >= 2:
        soft += 1
        reasons.append(f"{overlong} skills used longer than career+2yr")

    # --- SOFT: claims many advanced/expert skills but near-empty profile / no endorsements at all.
    adv = [s for s in skills if s.get("proficiency") in ("advanced", "expert")]
    if len(adv) >= 6 and sum((s.get("endorsements", 0) or 0) for s in adv) == 0:
        soft += 1
        reasons.append("many advanced/expert skills, zero endorsements across all")

    is_honeypot = (hard >= 1) or (soft >= 2)
    return is_honeypot, reasons
