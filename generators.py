"""
generators.py
LLM-powered generators for: interview questions, HR emails,
job descriptions, and candidate career suggestions.
"""

import json
from utils.llm_utils import chat


# ------------------------------------------------------------------
# INTERVIEW QUESTIONS
# ------------------------------------------------------------------

def generate_interview_questions(client, model, resume_data: dict, jd_data: dict, difficulty: str = "Medium") -> dict:
    prompt = f"""Generate interview questions for this candidate based on their resume and the job description.
Difficulty level: {difficulty}.

Return STRICT JSON only with this schema:
{{
  "technical_questions": ["..."],
  "behavioral_questions": ["..."],
  "hr_questions": ["..."],
  "project_questions": ["..."],
  "coding_questions": ["..."]
}}
Generate 4-6 questions per category, tailored to the candidate's actual skills/projects and the JD requirements.

CANDIDATE RESUME DATA:
{json.dumps(resume_data, indent=2)[:5000]}

JOB DESCRIPTION DATA:
{json.dumps(jd_data, indent=2)[:3000]}
"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert technical interviewer. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)}


# ------------------------------------------------------------------
# EMAIL GENERATOR
# ------------------------------------------------------------------

EMAIL_TYPES = {
    "Interview Invitation": "a warm, professional interview invitation email, including placeholders for [Date], [Time], [Mode/Location], and [Interviewer Name]",
    "Rejection Email": "a polite, respectful rejection email that leaves a positive impression of the company",
    "Offer Letter": "a concise offer letter summary email (not the legal contract) with placeholders for [Role], [Salary], [Joining Date], [Reporting Manager]",
    "Shortlisted Email": "an email informing the candidate they have been shortlisted for the next round, with placeholders for [Next Steps] and [Timeline]",
}


def generate_email(client, model, email_type: str, candidate_name: str, role: str, company: str, extra_notes: str = "") -> str:
    instruction = EMAIL_TYPES.get(email_type, "a professional HR email")
    system = "You are an expert HR communications specialist. Write clear, warm, professional emails."
    user = f"""Write {instruction}.

Candidate Name: {candidate_name}
Role: {role}
Company: {company}
Additional notes: {extra_notes or 'None'}

Include a subject line at the top formatted as: Subject: ...
Keep it concise, professional, and ready to send."""
    return chat(client, model, system, user, temperature=0.5)


# ------------------------------------------------------------------
# JOB DESCRIPTION GENERATOR
# ------------------------------------------------------------------

def generate_job_description(client, model, role: str, experience: str, skills: str, location: str, salary: str) -> str:
    system = "You are an expert HR content writer specializing in job descriptions."
    user = f"""Write a professional, well-structured job description with these inputs:

Role: {role}
Experience Required: {experience}
Key Skills: {skills}
Location: {location}
Salary Range: {salary}

Structure it with sections: Job Title, About the Role, Key Responsibilities,
Required Skills & Qualifications, Preferred Qualifications, Experience, Location, Salary, How to Apply.
Use clean markdown formatting with headers and bullet points."""
    return chat(client, model, system, user, temperature=0.5)


# ------------------------------------------------------------------
# CAREER SUGGESTIONS
# ------------------------------------------------------------------

def generate_career_suggestions(client, model, resume_data: dict, jd_data: dict, missing_skills: list) -> str:
    system = "You are a career coach and technical mentor giving constructive, encouraging advice."
    user = f"""Based on this candidate's profile and the missing skills relative to a target role,
suggest a personalized growth plan.

Candidate skills: {resume_data.get('skills', [])}
Candidate experience: {resume_data.get('total_experience_years', 'unknown')} years
Target role: {jd_data.get('job_title', 'the role')}
Missing skills: {missing_skills}

Provide, using markdown with headers:
1. Recommended Courses (2-4, with platform names like Coursera/Udemy where relevant)
2. Certifications to pursue (2-3)
3. Project ideas to build (2-3, specific and practical)
4. Skills to prioritize learning next (ranked)
5. Suggested career path / next role progression (2-3 sentence narrative)"""
    return chat(client, model, system, user, temperature=0.6)


# ------------------------------------------------------------------
# JD SUMMARY (used right after JD upload)
# ------------------------------------------------------------------

def summarize_jd(client, model, jd_text: str) -> str:
    system = "You are an HR analyst who writes crisp job description summaries."
    user = f"Summarize this job description in 4-6 sentences, highlighting role, key skills, experience level, and standout requirements:\n\n{jd_text[:6000]}"
    return chat(client, model, system, user, temperature=0.3)
