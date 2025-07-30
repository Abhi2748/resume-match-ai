import streamlit as st
import tempfile
from graphs.resume_match_graph import app as resume_graph
from streamlit_extras.stylable_container import stylable_container
import time
import fitz  # PyMuPDF
import base64

# Load SVG once and inject as background
def get_svg_background_base64(svg_path="blob-scene-haikei.svg"):
    with open(svg_path, "rb") as svg_file:
        encoded = base64.b64encode(svg_file.read()).decode("utf-8")
    return f"data:image/svg+xml;base64,{encoded}"

svg_data_url = get_svg_background_base64()

# ✅ Page Configuration
st.set_page_config(page_title="Resume Match AI", layout="wide")

# ✅ Global Style with SVG Background
st.markdown(f"""
    <style>
        .stApp {{
            background-image: url("{svg_data_url}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            background-repeat: no-repeat;
        }}
        .header {{
            text-align: center;
            padding: 4rem 2rem 2rem;
            color: white;
        }}
        .header h1 {{
            font-size: 3.2rem;
            margin-bottom: 0.4rem;
        }}
        .header p {{
            font-size: 1.2rem;
            color: #e0e0e0;
        }}
        .section {{
            background: transparent;
            border-radius: 0px;
            padding: 25px 0px;
            margin-bottom: 30px;
            box-shadow: none;
            animation: fadeIn 0.6s ease-in-out;
        }}
        .match-title {{
            font-size: 1.5rem;
            font-weight: bold;
            margin-bottom: 15px;
            color: #ff6b35;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
        }}
        .match-score-circle {{
            width: 200px;
            height: 200px;
            border-radius: 50%;
            border: 12px solid #f59e0b;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2.5rem;
            font-weight: bold;
            color: #1e293b;
            margin: 20px auto;
            background: white;
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        }}
        .score-row {{
            display: flex;
            flex-direction: row;
            justify-content: space-around;
            align-items: flex-start;
        }}
        .score-column {{
            flex: 1;
            text-align: center;
            padding: 0 20px;
        }}
        .roles-column {{
            flex: 1;
            text-align: left;
            padding: 0 20px;
        }}
        /* White text for upload labels on dark background */
        .stFileUploader > label {{
            color: white !important;
        }}
        .stFileUploader > div > div > div > div > span {{
            color: white !important;
        }}
        /* White text for content inside sections */
        .section p {{
            color: white !important;
            font-size: 1.1rem;
            line-height: 1.6;
            margin-bottom: 8px;
        }}
        .section b {{
            color: #fbbf24 !important;
            font-weight: 600;
        }}
        .section em {{
            color: #e5e7eb !important;
            font-style: italic;
        }}
        @keyframes fadeIn {{
            0% {{ opacity: 0; transform: translateY(15px); }}
            100% {{ opacity: 1; transform: translateY(0); }}
        }}
    </style>
""", unsafe_allow_html=True)

# ✅ Hero Section
st.markdown("""
<div class="header">
    <h1>📄 Resume Match AI</h1>
    <p>Upload your resume (PDF) and job description (TXT) to evaluate your fit, identify gaps, and receive actionable recommendations.</p>
</div>
""", unsafe_allow_html=True)

# 📤 Upload Files
col1, col2 = st.columns(2)
with col1:
    resume_file = st.file_uploader("📎 Upload Resume (PDF)", type=["pdf"])
with col2:
    jd_file = st.file_uploader("📋 Upload Job Description (TXT)", type=["txt"])

run_button = st.button("🚀 Analyze Resume")

# ⏳ Processing
if run_button and resume_file and jd_file:
    with st.spinner("Running resume analysis... ⏳"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(resume_file.read())
            resume_path = temp_pdf.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_txt:
            temp_txt.write(jd_file.read())
            jd_path = temp_txt.name

        doc = fitz.open(resume_path)
        resume_text = "\n".join([page.get_text() for page in doc])
        jd_text = open(jd_path, "r", encoding="utf-8").read()

        inputs = {"resume_text": resume_text, "jd_text": jd_text}
        result = resume_graph.invoke(inputs)

    st.markdown("---")

    # ✅ Match Score and Job Roles - Left and Right Layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class='section' style='display: flex; flex-direction: column; justify-content: center; height: 100%;'>
            <div class='match-title'>🔍 Match Score</div>
            <div class='match-score-circle'>""" + str(int(result.get('semantic_match_score', 0.0) * 100)) + """%</div>
        </div>""", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='section'>
            <div class='match-title'>🎯 Suggested Job Roles</div>""", unsafe_allow_html=True)
        
        roles = result.get("realistic_roles_with_reasons", [])
        if isinstance(roles, list) and roles:
            for role in roles:
                title = role.get("title", "Unknown Role")
                reason = role.get("reason", "No reason provided.")
                st.markdown(f"<p><b>{title}:</b> <em>{reason}</em></p>", unsafe_allow_html=True)
        else:
            st.markdown("<p>No job role suggestions available.</p>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

    time.sleep(0.3)

    # ✅ Matching Points
    st.markdown("""
    <div class='section'>
        <div class='match-title'>✅ Matching Points</div>""", unsafe_allow_html=True)
    
    for point in result.get("matching_points", []):
        st.markdown(f"<p>• {point}</p>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    # ❌ Missing Points
    st.markdown("""
    <div class='section'>
        <div class='match-title'>❌ Missing Points</div>""", unsafe_allow_html=True)
    
    for point in result.get("missing_points", []):
        st.markdown(f"<p>• {point}</p>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    # 📈 Career Tips
    st.markdown("""
    <div class='section'>
        <div class='match-title'>📈 Career Improvement Tips</div>""", unsafe_allow_html=True)
    
    for tip in result.get("career_improvement_tips", []):
        st.markdown(f"<p>💡 {tip}</p>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.warning("📥 Please upload both Resume and Job Description to begin.")