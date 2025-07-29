import fitz  # PyMuPDF
import re
from prompts import (
    extract_resume_info_with_llm,
    extract_jd_info_with_llm,
    llm_match_skills_and_responsibilities
)

def extract_text_from_pdf(pdf_path):
    """Extract raw text from PDF using PyMuPDF."""
    with fitz.open(pdf_path) as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

def clean_text(text):
    """Basic cleaning: remove extra spaces and normalize."""
    return re.sub(r'\s+', ' ', text.strip())

def parse_resume_text(resume_text):
    # LLM extraction with regex fallback
    skills, projects = extract_resume_info_with_llm(resume_text)
    if not skills:
        pattern = r'(?i)SKILLS\s*[:\-]?\s*(.*?)(?=\n[A-Z][A-Z ]{2,}|$)'
        match = re.search(pattern, resume_text, re.DOTALL)
        if match:
            skills_line = match.group(1)
            skills = [s.strip() for s in re.split(r',|\n|•|·', skills_line) if s.strip()]
    if not projects:
        pattern = r'(?i)PROJECTS\s+(.*?)(?=\n[A-Z][A-Z ]{2,}|$)'
        match = re.search(pattern, resume_text, re.DOTALL)
        if match:
            projects_block = match.group(1)
            projects = [p.strip() for p in re.split(r'\n|•|·', projects_block) if p.strip()]
    return {
        "raw_text": resume_text,
        "SKILLS": ", ".join(skills),
        "Key projects": projects
    }

def parse_jd_text(jd_text):
    # LLM extraction with regex fallback
    skills, responsibilities = extract_jd_info_with_llm(jd_text)
    if not skills:
        pattern = r'(?i)SKILLS\s*[:\-]?\s*(.*?)(?=\n[A-Z][A-Z ]{2,}|$)'
        match = re.search(pattern, jd_text, re.DOTALL)
        if match:
            skills_line = match.group(1)
            skills = [s.strip() for s in re.split(r',|\n|•|·', skills_line) if s.strip()]
    if not responsibilities:
        pattern = r'(?i)(Responsibilities|Requirements)[:-]?\s+(.*?)(?=(Qualifications|$))'
        match = re.search(pattern, jd_text, re.DOTALL)
        if match:
            resp_block = match.group(2)
            responsibilities = [r.strip() for r in re.split(r'\n|•|·', resp_block) if r.strip()]
    return {
        "critical_skills": skills,
        "key_responsibilities": responsibilities
    }

def get_llm_matching(resume_data, jd_data):
    resume_skills = resume_data.get("SKILLS", "")
    jd_skills = jd_data.get("critical_skills", [])
    resume_projects = resume_data.get("Key projects", [])
    jd_responsibilities = jd_data.get("key_responsibilities", [])
    return llm_match_skills_and_responsibilities(
        resume_skills, jd_skills, resume_projects, jd_responsibilities
    )