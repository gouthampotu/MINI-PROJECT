"""
parser.py
Handles reading PDF / DOCX / TXT files and extracting structured
candidate information (name, email, phone, skills, education, etc.)
"""

import io
import re
import json
import docx2txt
from pypdf import PdfReader


# ------------------------------------------------------------------
# RAW TEXT EXTRACTION
# ------------------------------------------------------------------

def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
        return text.strip()
    except Exception as e:
        return f"[ERROR extracting PDF: {e}]"


def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        text = docx2txt.process(io.BytesIO(file_bytes))
        return (text or "").strip()
    except Exception as e:
        return f"[ERROR extracting DOCX: {e}]"


def extract_text_from_txt(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8", errors="ignore").strip()
    except Exception as e:
        return f"[ERROR extracting TXT: {e}]"


def extract_text(uploaded_file) -> str:
    """
    uploaded_file: a Streamlit UploadedFile object
    Returns plain text content regardless of format.
    """
    name = uploaded_file.name.lower()
    file_bytes = uploaded_file.getvalue()

    if name.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    elif name.endswith(".docx"):
        return extract_text_from_docx(file_bytes)
    elif name.endswith(".txt"):
        return extract_text_from_txt(file_bytes)
    else:
        return "[Unsupported file type]"


# ------------------------------------------------------------------
# REGEX-BASED QUICK EXTRACTION (fast, no LLM cost, used as fallback)
# ------------------------------------------------------------------

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(r"(\+?\d{1,3}[-.\s]?)?\(?\d{3,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}")
LINKEDIN_RE = re.compile(r"(https?://)?(www\.)?linkedin\.com/[a-zA-Z0-9_/\-]+", re.I)
GITHUB_RE = re.compile(r"(https?://)?(www\.)?github\.com/[a-zA-Z0-9_/\-]+", re.I)

COMMON_SKILLS = [
    "python", "java", "c++", "c#", "javascript", "typescript", "sql", "nosql",
    "aws", "azure", "gcp", "docker", "kubernetes", "machine learning",
    "deep learning", "nlp", "computer vision", "data science", "pandas",
    "numpy", "scikit-learn", "tensorflow", "pytorch", "keras", "react",
    "angular", "vue", "node.js", "django", "flask", "fastapi", "spring boot",
    "html", "css", "git", "linux", "excel", "power bi", "tableau",
    "spark", "hadoop", "airflow", "kafka", "mongodb", "postgresql",
    "mysql", "redis", "graphql", "rest api", "microservices", "ci/cd",
    "terraform", "jenkins", "agile", "scrum", "streamlit", "langchain",
    "openai", "generative ai", "llm", "faiss", "vector database"
]


def regex_quick_extract(text: str) -> dict:
    emails = EMAIL_RE.findall(text)
    phones = PHONE_RE.findall(text)
    linkedin = LINKEDIN_RE.search(text)
    github = GITHUB_RE.search(text)

    text_lower = text.lower()
    found_skills = sorted({s for s in COMMON_SKILLS if s in text_lower})

    return {
        "email": emails[0] if emails else "Not found",
        "phone": _clean_phone(text),
        "linkedin": linkedin.group(0) if linkedin else "Not found",
        "github": github.group(0) if github else "Not found",
        "skills_regex": found_skills,
    }


def _clean_phone(text: str):
    matches = re.findall(r"(\+?\d[\d\-.\s]{8,14}\d)", text)
    return matches[0].strip() if matches else "Not found"


# ------------------------------------------------------------------
# LLM-BASED STRUCTURED EXTRACTION (accurate, used for final parsing)
# ------------------------------------------------------------------

RESUME_EXTRACTION_PROMPT = """You are an expert resume parser used inside an ATS system.
Extract structured information from the resume text below and return STRICT JSON only,
no markdown, no commentary, no code fences.

JSON schema:
{{
  "name": "string",
  "email": "string",
  "phone": "string",
  "location": "string",
  "linkedin": "string",
  "github": "string",
  "summary": "string (2-3 lines)",
  "skills": ["list of skills"],
  "education": [{{"degree": "", "institution": "", "year": ""}}],
  "experience": [{{"title": "", "company": "", "duration": "", "description": ""}}],
  "total_experience_years": "number (estimate, numeric only)",
  "projects": [{{"name": "", "description": "", "tech_stack": []}}],
  "certifications": ["list"],
  "languages": ["spoken/written languages"]
}}

If a field is missing, use an empty string, empty list, or 0 where appropriate.
Never invent facts not present in the resume text.

RESUME TEXT:
---
{resume_text}
---
"""


def llm_extract_resume(client, model, resume_text: str) -> dict:
    """
    Uses the LLM to produce structured resume JSON.
    `client` is an OpenAI() client instance.
    """
    prompt = RESUME_EXTRACTION_PROMPT.format(resume_text=resume_text[:12000])
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You extract structured resume data and return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        return data
    except Exception as e:
        return {"error": str(e)}


JD_EXTRACTION_PROMPT = """You are an expert HR analyst. Extract structured requirements from
the job description below. Return STRICT JSON only, no markdown, no commentary.

JSON schema:
{{
  "job_title": "string",
  "required_skills": ["list"],
  "preferred_skills": ["list"],
  "min_experience_years": "number",
  "education_requirements": ["list"],
  "certifications_required": ["list"],
  "keywords": ["list of important ATS keywords"],
  "summary": "2-4 sentence AI summary of the role"
}}

JOB DESCRIPTION:
---
{jd_text}
---
"""


def llm_extract_jd(client, model, jd_text: str) -> dict:
    prompt = JD_EXTRACTION_PROMPT.format(jd_text=jd_text[:12000])
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You extract structured job description data and return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        return {"error": str(e)}
