"""
ats.py
ATS scoring engine: combines embedding cosine similarity with
LLM-based reasoning to produce explainable sub-scores.
Also contains skill-gap analysis and lightweight fraud detection.
"""

import json
import re
import numpy as np
from utils.llm_utils import get_embedding


def cosine_similarity(a, b) -> float:
    a, b = np.array(a), np.array(b)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def normalize_skill(s: str) -> str:
    return re.sub(r"[^a-z0-9+#. ]", "", s.lower().strip())


def skill_gap_analysis(required_skills: list, candidate_skills: list) -> dict:
    req_norm = {normalize_skill(s): s for s in required_skills}
    cand_norm = {normalize_skill(s): s for s in candidate_skills}

    matched = [req_norm[k] for k in req_norm if k in cand_norm]
    missing = [req_norm[k] for k in req_norm if k not in cand_norm]
    additional = [cand_norm[k] for k in cand_norm if k not in req_norm]

    match_pct = round((len(matched) / len(req_norm) * 100), 1) if req_norm else 0.0

    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "additional_skills": additional,
        "match_percentage": match_pct,
    }


ATS_EXPLAIN_PROMPT = """You are an ATS (Applicant Tracking System) scoring engine used by HR teams.

Given the Job Description requirements and a Candidate's resume data below, score the candidate
on each dimension from 0-100 and give a short one-line reason for each. Return STRICT JSON only.

JSON schema:
{{
  "skills_match": {{"score": 0-100, "reason": "string"}},
  "experience_match": {{"score": 0-100, "reason": "string"}},
  "education_match": {{"score": 0-100, "reason": "string"}},
  "projects_match": {{"score": 0-100, "reason": "string"}},
  "certification_match": {{"score": 0-100, "reason": "string"}},
  "resume_formatting": {{"score": 0-100, "reason": "string"}},
  "overall_ats_score": 0-100,
  "strengths": ["list of 3-5 strengths"],
  "weaknesses": ["list of 3-5 weaknesses"],
  "hire_recommendation": "one of: Highly Recommended, Recommended, Consider After Interview, Hold, Reject",
  "recommendation_reason": "2-3 sentence explanation",
  "hr_summary": "max 200 word HR-style summary of this candidate"
}}

JOB DESCRIPTION (structured):
{jd_json}

CANDIDATE RESUME (structured):
{resume_json}

Weight the overall_ats_score roughly as:
skills 35%, experience 25%, education 15%, projects 15%, certifications 5%, formatting 5%.
Be honest and critical, not overly generous. Base every score strictly on the evidence given.
"""


def compute_ats_score(client, model, jd_data: dict, resume_data: dict) -> dict:
    prompt = ATS_EXPLAIN_PROMPT.format(
        jd_json=json.dumps(jd_data, indent=2)[:6000],
        resume_json=json.dumps(resume_data, indent=2)[:6000],
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a strict, fair, explainable ATS scoring engine. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)}


def embedding_similarity_score(client, jd_text: str, resume_text: str) -> float:
    """Fast supplementary semantic-similarity score (0-100)."""
    try:
        jd_emb = get_embedding(client, jd_text)
        res_emb = get_embedding(client, resume_text)
        sim = cosine_similarity(jd_emb, res_emb)
        return round(max(0.0, min(1.0, sim)) * 100, 1)
    except Exception:
        return 0.0


# ------------------------------------------------------------------
# FRAUD / RED-FLAG DETECTION (heuristic, no LLM call needed)
# ------------------------------------------------------------------

def detect_red_flags(resume_text: str, resume_data: dict) -> list:
    flags = []
    text_lower = resume_text.lower()

    # Missing contact info
    if not resume_data.get("email") or resume_data.get("email") == "Not found":
        flags.append("⚠️ Missing email address")
    if not resume_data.get("phone") or resume_data.get("phone") == "Not found":
        flags.append("⚠️ Missing phone number")

    # Keyword stuffing: same skill word repeated excessively
    words = re.findall(r"[a-zA-Z+#]{3,}", text_lower)
    if words:
        from collections import Counter
        counts = Counter(words)
        common_stopwords = {"the", "and", "for", "with", "this", "that", "from", "have", "will"}
        stuffed = [w for w, c in counts.items() if c > 12 and w not in common_stopwords]
        if stuffed:
            flags.append(f"⚠️ Possible keyword stuffing detected: {', '.join(stuffed[:5])}")

    # Repeated content (duplicate lines)
    lines = [l.strip() for l in resume_text.split("\n") if len(l.strip()) > 25]
    dupes = len(lines) - len(set(lines))
    if dupes > 3:
        flags.append(f"⚠️ {dupes} duplicated content lines detected — possible copy-paste padding")

    # Resume too short
    if len(resume_text.split()) < 80:
        flags.append("⚠️ Resume content unusually short — may be incomplete")

    # Suspicious absolute claims
    suspicious_terms = ["expert in everything", "guaranteed results", "world's best",
                         "100% perfect", "no experience needed"]
    for term in suspicious_terms:
        if term in text_lower:
            flags.append(f"⚠️ Suspicious claim found: '{term}'")

    if not flags:
        flags.append("✅ No major red flags detected")

    return flags
