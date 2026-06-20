"""
jd_spec.py
==========
Structured, auditable representation of the released Job Description
("Senior AI Engineer — Founding Team", Redrob AI).

Everything the scorer rewards or penalises is declared here as plain data so
it can be defended line-by-line at Stage 4/5. No magic constants buried in
logic. Edit weights here, not in scoring.py.

Decoded directly from job_description.docx — see the brief for the mapping.
"""

# ---- Plain-language ideal-candidate query (used for the dense/lexical semantic match).
# Written in *natural language*, NOT keyword soup, so it matches "plain-language
# Tier-5" candidates who describe real systems work without buzzwords.
JD_QUERY = (
    "Senior AI engineer who has shipped production embeddings-based retrieval, "
    "hybrid and vector search, and ranking, search, recommendation or personalization "
    "systems to real users at a product company. Strong Python engineer who has "
    "designed evaluation frameworks for ranking systems using NDCG, MRR, MAP and "
    "A/B testing, and has handled embedding drift, index refresh and retrieval-quality "
    "regression in production. Around six to eight years of applied machine learning "
    "experience at product companies rather than pure research or services. Based in "
    "or willing to relocate to Pune, Noida, Hyderabad, Mumbai, Delhi NCR or Bangalore."
)

# ---- Evidence terms that indicate the candidate ACTUALLY DID the core work.
# Matched against career-history descriptions + summary (demonstrated work),
# weighted higher than the same words appearing only in the skills list.
DOMAIN_CORE = [
    "ranking", "rank ", "learning to rank", "ranker", "relevance",
    "retrieval", "information retrieval", "semantic search", "search relevance",
    "recommendation", "recommender", "personalization", "personalisation",
    "embedding", "embeddings", "vector search", "vector database", "nearest neighbor",
    "bm25", "elasticsearch", "opensearch", "faiss", "pinecone", "weaviate",
    "qdrant", "milvus", "sentence-transformers", "bge", "e5",
    "matching", "candidate matching", "two-tower", "ann index",
]
DOMAIN_EVAL = ["ndcg", "mrr", "map@", "mean average precision", "a/b test",
               "ab test", "offline metric", "online metric", "ctr", "engagement metric"]
DOMAIN_ML_GENERAL = ["machine learning", "deep learning", "nlp", "natural language",
                     "transformer", "fine-tun", "lora", "qlora", "peft", "xgboost",
                     "model training", "inference", "production ml", "mlops"]

# JD-named skills (used in the *trust-gated* skill component).
JD_CORE_SKILLS = {
    "embeddings", "sentence-transformers", "retrieval", "vector search", "faiss",
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch",
    "python", "nlp", "information retrieval", "learning to rank", "ranking",
    "recommendation systems", "ndcg", "machine learning", "deep learning",
    "fine-tuning llms", "llm", "rag", "pytorch", "tensorflow",
}

# ---- Experience band. Ideal 6-8, acceptable 5-9, soft outside.
EXP_IDEAL_LO, EXP_IDEAL_HI = 6.0, 8.0
EXP_OK_LO, EXP_OK_HI = 5.0, 9.0

# ---- Location. Tier-1 Indian cities explicitly welcomed in the JD.
LOCATION_PRIMARY = {"pune", "noida", "hyderabad", "mumbai", "delhi", "gurgaon",
                    "gurugram", "new delhi", "bengaluru", "bangalore", "delhi ncr",
                    "ghaziabad", "faridabad"}

# ---- Company classification.
# Product-company industries (positive). Services / non-tech (negative-ish).
# NOTE: keep these specific. Bare substrings like "ai" or "product" cause false
# positives via `k in industry` matching -- e.g. "ai" matched "Retail" (r-et-AI-l)
# and "product" matched "Paper Products", misclassifying non-tech firms as product.
PRODUCT_INDUSTRIES = {"software", "fintech", "e-commerce", "ecommerce", "saas",
                      "gaming", "edtech", "food delivery", "internet",
                      "ai/", " ai", "artificial intelligence",
                      "social media", "healthtech", "adtech"}
SERVICES_INDUSTRIES = {"it services", "consulting", "outsourcing", "staffing"}
NONTECH_INDUSTRIES = {"manufacturing", "paper products", "conglomerate",
                      "construction", "logistics", "retail", "bpo"}
# Named consulting/services firms the JD calls out ("only worked here whole career" = not a fit).
SERVICES_FIRMS = {"tcs", "tata consultancy", "infosys", "wipro", "accenture",
                  "cognizant", "capgemini", "mindtree", "ltimindtree", "hcl",
                  "tech mahindra", "mphasis", "deloitte", "pwc", "kpmg", "ey",
                  "ernst", "genpact", "dxc"}

# ---- Disqualifier / non-fit title buckets (the keyword-stuffer decoys).
# A profile whose CURRENT title is one of these is hard-capped low even if the
# skills section is stuffed with AI terms. This is the single most decisive
# anti-trap signal (confirmed by the organiser's own metadata example).
NONFIT_TITLES = {
    "hr manager", "human resources", "recruiter", "talent acquisition",
    "sales executive", "sales manager", "business development",
    "marketing manager", "digital marketing", "content writer", "copywriter",
    "graphic designer", "ui designer", "ux designer",  # design-only, no eng
    "accountant", "finance manager", "auditor",
    "customer support", "customer success", "operations manager",
    "civil engineer", "mechanical engineer", "electrical engineer",
    "project manager", "program manager", "business analyst",
    "product manager",  # PM is not the IC AI-eng role this JD wants
    "teacher", "professor", "lecturer",
}
# Titles indicating a *fit-shaped* IC engineer (positive prior).
FIT_TITLES = {
    "ai engineer", "machine learning engineer", "ml engineer",
    "applied scientist", "applied ml", "research engineer",
    "data scientist", "senior data scientist", "nlp engineer",
    "search engineer", "relevance engineer", "recommendation engineer",
    "software engineer", "senior software engineer", "staff engineer",
    "principal engineer", "backend engineer", "platform engineer",
    "mle", "ml scientist",
}
# Adjacent-but-wrong specialisations (JD: CV/speech/robotics primary => not a fit).
WRONG_SPECIALISATION = ["computer vision", "image classification", "object detection",
                        "speech recognition", "asr", "robotics", "ros ", "slam",
                        "autonomous", "embedded vision"]

# ---- Component weights for the base fit score (sum ≈ 1.0). Tune on the local
# eval harness; these are sensible, defensible starting points.
WEIGHTS = {
    "semantic":        0.30,  # dense/lexical JD↔evidence match (recall for plain-language fits)
    "domain_evidence": 0.26,  # demonstrated IR/ranking/search/recsys work in career text
    "experience":      0.12,  # 6-8 yr band fit
    "product":         0.14,  # product-company vs services/non-tech
    "skills_trust":    0.08,  # trust-gated skill overlap (catches keyword stuffers)
    "location":        0.10,  # Tier-1 India / relocate
}

# Availability multiplier clamp (behavioural signals as a modifier, per signals doc).
AVAIL_MIN, AVAIL_MAX = 0.55, 1.08

# Hybrid semantic blend: weight on the dense (bge-small) cosine vs the TF-IDF
# cosine, both min-max normalised to [0,1]. Lexical-led (0.3) keeps keyword
# precision at the very top (the strongest profiles stay #1) while folding in the
# dense signal for recall on plain-language fits at the margin. Measured: 0.3
# keeps the elite at #1 and pulls genuine missed Tier-5s into the top-100.
HYBRID_W_DENSE = 0.3

# Hard cap applied to non-fit-title profiles (so stuffers cannot reach the top).
NONFIT_TITLE_CAP = 0.18

# Penalty for candidates based OUTSIDE India who are NOT willing to relocate.
# The JD is explicit: "Outside India: case-by-case, but we don't sponsor work
# visas." Such a candidate cannot actually be hired into the Pune/Noida hybrid
# role, so they must not occupy a top slot regardless of how strong the profile
# reads. Abroad-but-willing-to-relocate candidates are NOT penalised here (they
# already get the relocate location credit and sit mid-pack).
ABROAD_NORELOCATE_PENALTY = 0.35

TOP_K = 100
