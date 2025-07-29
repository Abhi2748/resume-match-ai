import streamlit as st
from utils.data_utils import parse_resume, parse_job_description
from match_engine import compute_match_score, compare_skills
from prompts import get_suggestions
import tempfile

st.set_page_config(page_title="Resume Match AI", page_icon="🧠", layout="wide")

st.title("🧠 Resume Match AI")
st.markdown("Upload your resume and paste a job description. We'll match and analyze it using AI.")

# --- Upload Resume ---
resume_file = st.file_uploader("📄 Upload your Resume (PDF)", type=["pdf"])

# --- Job Description ---
jd_text = st.text_area("💼 Paste the Job Description", height=300, placeholder="Paste the JD here...")

if resume_file and jd_text:
    with st.spinner("Analyzing..."):

        # Save resume to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(resume_file.read())
            resume_path = tmp.name

        # Parse
        resume_data = parse_resume(resume_path)
        jd_data = parse_job_description(jd_text)

        # Match Score
        score = compute_match_score(resume_data["raw_text"], jd_data["raw_text"])
        skill_match = compare_skills(resume_data["skills"], jd_data["skills"])
        suggestions = get_suggestions(resume_data["raw_text"], jd_data["raw_text"])

    # --- Output Section ---
    st.markdown("## 📊 Results")

    st.metric("🔍 Match Score", f"{score} %")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ✅ Common Skills")
        st.write(", ".join(skill_match["common_skills"]) or "None found.")

    with col2:
        st.markdown("### ❌ Missing Skills")
        st.write(", ".join(skill_match["missing_skills"]) or "No major gaps!")

    st.markdown("### 🤖 Suggestions from AI")
    st.markdown(suggestions)

elif resume_file or jd_text:
    st.warning("Please upload both resume and job description.")
