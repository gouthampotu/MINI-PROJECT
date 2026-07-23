"""
report_generator.py
Generates a per-candidate PDF report (ReportLab) and a multi-candidate
Excel workbook (openpyxl via pandas) for HR download.
"""

import io
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)


def generate_candidate_pdf(candidate: dict, interview_questions: dict = None) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle("TitleStyle", parent=styles["Title"], textColor=colors.HexColor("#4F46E5"))
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], textColor=colors.HexColor("#1E293B"), spaceBefore=12)
    body = styles["BodyText"]

    resume = candidate["resume_data"]
    ats = candidate["ats"]

    elements = []
    elements.append(Paragraph("AI Resume Screening Report", title_style))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(f"<b>Candidate:</b> {resume.get('name', 'Unknown')}", body))
    elements.append(Paragraph(f"<b>Email:</b> {resume.get('email', 'N/A')} &nbsp;&nbsp; "
                               f"<b>Phone:</b> {resume.get('phone', 'N/A')}", body))
    elements.append(Paragraph(f"<b>Overall ATS Score:</b> {ats.get('overall_ats_score', 'N/A')} / 100", body))
    elements.append(Paragraph(f"<b>Recommendation:</b> {ats.get('hire_recommendation', 'N/A')}", body))
    elements.append(Spacer(1, 10))

    # ATS breakdown table
    elements.append(Paragraph("ATS Score Breakdown", h2))
    score_rows = [["Category", "Score", "Reason"]]
    for key, label in [
        ("skills_match", "Skills Match"), ("experience_match", "Experience Match"),
        ("education_match", "Education Match"), ("projects_match", "Projects Match"),
        ("certification_match", "Certification Match"), ("resume_formatting", "Formatting"),
    ]:
        val = ats.get(key, {})
        score_rows.append([label, f"{val.get('score', '-')}%", val.get("reason", "-")])

    table = Table(score_rows, colWidths=[4 * cm, 2 * cm, 9.5 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4F46E5")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 10))

    # Strengths / Weaknesses
    elements.append(Paragraph("Strengths", h2))
    for s in ats.get("strengths", []):
        elements.append(Paragraph(f"• {s}", body))

    elements.append(Paragraph("Weaknesses", h2))
    for w in ats.get("weaknesses", []):
        elements.append(Paragraph(f"• {w}", body))

    elements.append(Paragraph("HR Summary", h2))
    elements.append(Paragraph(ats.get("hr_summary", "N/A"), body))

    elements.append(Paragraph("Recommendation Reason", h2))
    elements.append(Paragraph(ats.get("recommendation_reason", "N/A"), body))

    # Skills
    elements.append(Paragraph("Skills", h2))
    elements.append(Paragraph(", ".join(resume.get("skills", [])) or "N/A", body))

    # Interview questions (optional)
    if interview_questions:
        elements.append(PageBreak())
        elements.append(Paragraph("Suggested Interview Questions", title_style))
        for category, qs in interview_questions.items():
            if not qs:
                continue
            elements.append(Paragraph(category.replace("_", " ").title(), h2))
            for q in qs:
                elements.append(Paragraph(f"• {q}", body))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


def generate_excel_report(candidates: list) -> bytes:
    rows = []
    for c in candidates:
        resume = c["resume_data"]
        ats = c["ats"]
        rows.append({
            "Candidate Name": resume.get("name", "Unknown"),
            "Email": resume.get("email", ""),
            "Phone": resume.get("phone", ""),
            "Overall ATS Score": ats.get("overall_ats_score", 0),
            "Skills Match": ats.get("skills_match", {}).get("score", 0),
            "Experience Match": ats.get("experience_match", {}).get("score", 0),
            "Education Match": ats.get("education_match", {}).get("score", 0),
            "Projects Match": ats.get("projects_match", {}).get("score", 0),
            "Certification Match": ats.get("certification_match", {}).get("score", 0),
            "Formatting Score": ats.get("resume_formatting", {}).get("score", 0),
            "Experience (yrs)": resume.get("total_experience_years", 0),
            "Recommendation": ats.get("hire_recommendation", ""),
            "Skills": ", ".join(resume.get("skills", [])),
            "Missing Skills": ", ".join(c.get("skill_gap", {}).get("missing_skills", [])),
        })

    df = pd.DataFrame(rows).sort_values("Overall ATS Score", ascending=False)

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Candidate Ranking")
        worksheet = writer.sheets["Candidate Ranking"]
        for i, col in enumerate(df.columns, 1):
            max_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
            worksheet.column_dimensions[worksheet.cell(row=1, column=i).column_letter].width = min(max_len, 45)

    buffer.seek(0)
    return buffer.getvalue()
