"""
analytics.py
Builds Plotly charts for the Analytics Dashboard from the list of
processed candidates stored in session state.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from collections import Counter


def candidates_to_dataframe(candidates: list) -> pd.DataFrame:
    rows = []
    for c in candidates:
        rows.append({
            "Candidate": c["resume_data"].get("name", "Unknown"),
            "Overall ATS": c["ats"].get("overall_ats_score", 0),
            "Skills Match": c["ats"].get("skills_match", {}).get("score", 0),
            "Experience Match": c["ats"].get("experience_match", {}).get("score", 0),
            "Education Match": c["ats"].get("education_match", {}).get("score", 0),
            "Recommendation": c["ats"].get("hire_recommendation", "N/A"),
            "Experience (yrs)": c["resume_data"].get("total_experience_years", 0),
        })
    return pd.DataFrame(rows)


def avg_ats_gauge(df: pd.DataFrame):
    avg = round(df["Overall ATS"].mean(), 1) if not df.empty else 0
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg,
        title={"text": "Average ATS Score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#6366F1"},
            "steps": [
                {"range": [0, 50], "color": "#FCE7E7"},
                {"range": [50, 75], "color": "#FEF3C7"},
                {"range": [75, 100], "color": "#D1FAE5"},
            ],
        },
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def ranking_bar_chart(df: pd.DataFrame):
    d = df.sort_values("Overall ATS", ascending=True)
    fig = px.bar(
        d, x="Overall ATS", y="Candidate", orientation="h",
        color="Overall ATS", color_continuous_scale="Viridis",
        text="Overall ATS", title="Candidate Ranking by ATS Score",
    )
    fig.update_traces(texttemplate="%{text}", textposition="outside")
    fig.update_layout(height=max(300, 40 * len(d)), margin=dict(l=10, r=10, t=50, b=10))
    return fig


def top_skills_chart(candidates: list, top_n: int = 15):
    all_skills = []
    for c in candidates:
        all_skills.extend([s.lower().strip() for s in c["resume_data"].get("skills", [])])
    counts = Counter(all_skills).most_common(top_n)
    if not counts:
        return None
    df = pd.DataFrame(counts, columns=["Skill", "Count"])
    fig = px.bar(df, x="Count", y="Skill", orientation="h", color="Count",
                 color_continuous_scale="Blues", title="Top Skills Across Candidates")
    fig.update_layout(height=450, margin=dict(l=10, r=10, t=50, b=10), yaxis={"categoryorder": "total ascending"})
    return fig


def experience_distribution_chart(df: pd.DataFrame):
    fig = px.histogram(df, x="Experience (yrs)", nbins=10, title="Experience Distribution",
                        color_discrete_sequence=["#8B5CF6"])
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def education_distribution_chart(candidates: list):
    degrees = []
    for c in candidates:
        for edu in c["resume_data"].get("education", []):
            deg = edu.get("degree", "").strip()
            if deg:
                degrees.append(deg)
    if not degrees:
        return None
    counts = Counter(degrees).most_common(10)
    df = pd.DataFrame(counts, columns=["Degree", "Count"])
    fig = px.pie(df, names="Degree", values="Count", title="Education Distribution", hole=0.45)
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def certification_distribution_chart(candidates: list):
    certs = []
    for c in candidates:
        certs.extend([x.strip() for x in c["resume_data"].get("certifications", []) if x.strip()])
    if not certs:
        return None
    counts = Counter(certs).most_common(10)
    df = pd.DataFrame(counts, columns=["Certification", "Count"])
    fig = px.bar(df, x="Certification", y="Count", color="Count", color_continuous_scale="Teal",
                 title="Certification Distribution")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def recommendation_pie(df: pd.DataFrame):
    if df.empty:
        return None
    counts = df["Recommendation"].value_counts().reset_index()
    counts.columns = ["Recommendation", "Count"]
    color_map = {
        "Highly Recommended": "#10B981",
        "Recommended": "#3B82F6",
        "Consider After Interview": "#F59E0B",
        "Hold": "#F97316",
        "Reject": "#EF4444",
    }
    fig = px.pie(counts, names="Recommendation", values="Count", hole=0.45,
                 color="Recommendation", color_discrete_map=color_map,
                 title="Hiring Recommendation Breakdown")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
    return fig
