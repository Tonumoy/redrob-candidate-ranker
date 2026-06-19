"""
features.py
===========
Deterministic, vectorisable structured features per candidate, each in [0,1]
unless noted. Pure Python/stdlib — no network, no model. Fast.

Every feature maps to an explicit JD statement (see jd_spec.py / the brief) so
each ranking decision is defensible at Stage 4/5.
"""
from datetime import date
import jd_spec as J


def _txt(c):
    """Concatenated DEMONSTRATED-work text: headline+summary+career descriptions+titles."""
    p = c.get("profile", {})
    parts = [p.get("headline", ""), p.get("summary", "")]
    for r in c.get("career_history", []) or []:
        parts.append(r.get("title", ""))
        parts.append(r.get("description", ""))
    for e in c.get("education", []) or []:
        parts.append(e.get("field_of_study", ""))
    return " ".join(parts).lower()


def _skill_text(c):
    return " ".join((s.get("name", "") or "").lower() for s in (c.get("skills") or []))


def evidence_text(c):
    """Text used for the semantic match (work evidence, not raw skill tokens)."""
    p = c.get("profile", {})
    parts = [p.get("headline", ""), p.get("summary", "")]
    for r in c.get("career_history", []) or []:
        parts.append(f"{r.get('title','')}. {r.get('description','')}")
    return " ".join(parts)


def f_domain_evidence(c):
    """Did they demonstrably do IR/ranking/search/recsys + ML, in their CAREER text?"""
    t = _txt(c)
    skill_t = _skill_text(c)
    core = sum(1 for kw in J.DOMAIN_CORE if kw in t)
    eval_ = sum(1 for kw in J.DOMAIN_EVAL if kw in t)
    mlg = sum(1 for kw in J.DOMAIN_ML_GENERAL if kw in t)
    # credit demonstrated work; tiny credit for skill-only mentions (keyword-stuffer guard)
    core_skill_only = sum(1 for kw in J.DOMAIN_CORE if kw in skill_t and kw not in t)
    score = (min(core, 5) / 5.0) * 0.6 + (min(eval_, 2) / 2.0) * 0.2 + (min(mlg, 4) / 4.0) * 0.2
    score += min(core_skill_only, 3) / 3.0 * 0.05   # at most +0.05 for skill-list only
    return min(score, 1.0)


def f_experience(c):
    yoe = float(c["profile"].get("years_of_experience", 0) or 0)
    if J.EXP_IDEAL_LO <= yoe <= J.EXP_IDEAL_HI:
        return 1.0
    if J.EXP_OK_LO <= yoe < J.EXP_IDEAL_LO:
        return 0.55 + 0.45 * (yoe - J.EXP_OK_LO) / (J.EXP_IDEAL_LO - J.EXP_OK_LO)
    if J.EXP_IDEAL_HI < yoe <= J.EXP_OK_HI:
        return 0.55 + 0.45 * (J.EXP_OK_HI - yoe) / (J.EXP_OK_HI - J.EXP_IDEAL_HI)
    if yoe < J.EXP_OK_LO:
        return max(0.0, 0.55 * (yoe / J.EXP_OK_LO))
    return max(0.15, 0.55 - 0.05 * (yoe - J.EXP_OK_HI))  # >9 yrs: gentle decay


def _industry_class(ind):
    ind = (ind or "").lower()
    if any(k in ind for k in J.PRODUCT_INDUSTRIES):
        return "product"
    if any(k in ind for k in J.SERVICES_INDUSTRIES):
        return "services"
    if any(k in ind for k in J.NONTECH_INDUSTRIES):
        return "nontech"
    return "unknown"


def f_product(c):
    """Fraction of career tenure spent at product companies (by months)."""
    ch = c.get("career_history", []) or []
    tot = sum(r.get("duration_months", 0) or 0 for r in ch) or 1
    prod = 0
    services_only = True
    for r in ch:
        cls = _industry_class(r.get("industry"))
        comp = (r.get("company", "") or "").lower()
        is_services_firm = any(f in comp for f in J.SERVICES_FIRMS)
        if cls == "product" and not is_services_firm:
            prod += r.get("duration_months", 0) or 0
            services_only = False
        elif cls in ("unknown",) and not is_services_firm:
            prod += (r.get("duration_months", 0) or 0) * 0.4  # neutral benefit of doubt
            services_only = False
        elif cls == "nontech":
            services_only = False  # non-tech isn't the "consulting-only" trap
    frac = prod / tot
    # JD: "only worked at consulting firms entire career" => strong negative.
    if services_only and len(ch) >= 1:
        return 0.12
    return max(0.12, min(frac, 1.0))


def f_skills_trust(c):
    """Trust-gated overlap with JD core skills. Stuffed skills (0 endorse/0 dur) earn ~nothing."""
    hits = 0.0
    for s in c.get("skills", []) or []:
        name = (s.get("name", "") or "").lower()
        if not any(cs in name or name in cs for cs in J.JD_CORE_SKILLS):
            continue
        endorse = s.get("endorsements", 0) or 0
        dur = s.get("duration_months", 0) or 0
        prof = s.get("proficiency", "beginner")
        trust = 0.0
        trust += min(endorse, 20) / 20.0 * 0.5
        trust += min(dur, 36) / 36.0 * 0.3
        trust += {"beginner": 0.0, "intermediate": 0.4, "advanced": 0.8, "expert": 1.0}.get(prof, 0) * 0.2
        hits += trust
    return min(hits / 4.0, 1.0)  # ~4 trusted core skills saturates


def f_location(c):
    p = c["profile"]
    loc = (p.get("location", "") or "").lower()
    country = (p.get("country", "") or "").lower()
    relocate = bool(c.get("redrob_signals", {}).get("willing_to_relocate", False))
    if any(city in loc for city in J.LOCATION_PRIMARY):
        return 1.0
    if "india" in country:
        return 0.72
    if relocate:
        return 0.5
    return 0.15


# ---------- Negative / disqualifier signals (return penalty in [0,1]) ----------

def title_nonfit(c):
    t = (c["profile"].get("current_title", "") or "").lower()
    if any(nf in t for nf in J.NONFIT_TITLES):
        # allow "engineer"-bearing fit titles to escape generic matches
        if any(ft in t for ft in J.FIT_TITLES) and "manager" not in t:
            return False
        return True
    return False


def f_penalties(c):
    """Aggregate soft penalties (subtractive). Returns penalty in [0, 0.6]."""
    t = _txt(c)
    pen = 0.0
    reasons = []
    # Based abroad AND unwilling to relocate => JD offers no visa sponsorship,
    # so this candidate cannot be hired into the Pune/Noida role. Strong penalty
    # so a keyword-strong-but-unhirable profile cannot hold a top slot.
    country = (c["profile"].get("country", "") or "").lower()
    relocate = bool(c.get("redrob_signals", {}).get("willing_to_relocate", False))
    if country and "india" not in country and not relocate:
        pen += J.ABROAD_NORELOCATE_PENALTY
        reasons.append("based abroad and not willing to relocate (JD: no visa sponsorship)")
    # CV/speech/robotics primary, no IR/NLP evidence
    wrong = sum(1 for kw in J.WRONG_SPECIALISATION if kw in t)
    nlp_ir = any(kw in t for kw in ("nlp", "natural language", "retrieval", "ranking",
                                    "search", "recommendation", "information retrieval"))
    if wrong >= 2 and not nlp_ir:
        pen += 0.3; reasons.append("CV/speech/robotics-primary, no NLP/IR evidence")
    # framework-enthusiast: langchain/openai present but no pre-LLM ML depth
    if ("langchain" in t or "openai api" in t) and not any(
            k in t for k in ("xgboost", "scikit", "pre-llm", "ranking", "retrieval",
                             "recommendation", "production ml", "feature engineering")):
        pen += 0.12; reasons.append("framework-enthusiast pattern (LangChain/OpenAI only)")
    # title-chaser: many short stints
    ch = c.get("career_history", []) or []
    short = sum(1 for r in ch if (r.get("duration_months", 0) or 0) < 18 and not r.get("is_current"))
    if short >= 3:
        pen += 0.1; reasons.append(f"{short} sub-18-month stints (job-hopping pattern)")
    # pure academia/research only
    titles = " ".join((r.get("title", "") or "").lower() for r in ch)
    if any(k in titles for k in ("postdoc", "phd researcher", "research fellow", "lecturer", "professor")) \
            and not any(k in t for k in ("production", "shipped", "deployed", "users", "platform")):
        pen += 0.2; reasons.append("research/academia-only, no production deployment")
    return min(pen, 0.6), reasons


def availability_multiplier(c):
    s = c.get("redrob_signals", {}) or {}
    m = 1.0
    notes = []
    # recency
    la = s.get("last_active_date")
    if la:
        try:
            y, mo, d = map(int, la.split("-"))
            days = (date(2026, 6, 1) - date(y, mo, d)).days
            if days <= 14: m += 0.05
            elif days <= 45: m += 0.0
            elif days <= 120: m -= 0.08
            else: m -= 0.18; notes.append(f"inactive ~{days}d")
        except Exception:
            pass
    if s.get("open_to_work_flag"): m += 0.05
    else: m -= 0.02
    rr = s.get("recruiter_response_rate", 0.5)
    if rr is not None:
        if rr >= 0.6: m += 0.04
        elif rr < 0.15: m -= 0.12; notes.append(f"low response rate {rr}")
    icr = s.get("interview_completion_rate", 1.0)
    if icr is not None and icr < 0.4: m -= 0.06; notes.append(f"low interview completion {icr}")
    npd = s.get("notice_period_days", 60)
    if npd is not None:
        if npd <= 30: m += 0.04
        elif npd >= 90: m -= 0.06; notes.append(f"notice {npd}d")
    pcs = s.get("profile_completeness_score", 70)
    if pcs is not None and pcs < 50: m -= 0.05
    saved = s.get("saved_by_recruiters_30d", 0) or 0
    if saved >= 5: m += 0.03
    return max(J.AVAIL_MIN, min(J.AVAIL_MAX, m)), notes
