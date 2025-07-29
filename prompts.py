import os
from dotenv import load_dotenv
from utils.llm_utils import call_llm_json

# Load env vars
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing. Check your .env file.")

def extract_resume_info_with_llm(text):
    prompt = """
    From the following resume text, extract:
    - All relevant skills
    - All key project titles or descriptions
    Return as JSON:
    {
      "skills": [...],
      "projects": [...]
    }
    Resume:
    {text}
    """
    result = call_llm_json(prompt, {"text": text})
    return result.get("skills", []), result.get("projects", [])

def extract_jd_info_with_llm(text):
    prompt = """
    From the following job description, extract:
    - All relevant skills
    - All key responsibilities or requirements
    Return as JSON:
    {
      "skills": [...],
      "responsibilities": [...]
    }
    JD:
    {text}
    """
    result = call_llm_json(prompt, {"text": text})
    return result.get("skills", []), result.get("responsibilities", [])

def llm_match_skills_and_responsibilities(resume_skills, jd_skills, resume_projects, jd_responsibilities):
    prompt = f"""
    Compare the following resume skills and projects with the job description skills and responsibilities.
    Return as JSON:
    {{
      "matched_skills": [...],
      "unmatched_skills": [...],
      "matched_responsibilities": [...],
      "unmatched_responsibilities": [...]
    }}
    Resume Skills: {resume_skills}
    Resume Projects: {resume_projects}
    JD Skills: {jd_skills}
    JD Responsibilities: {jd_responsibilities}
    """
    result = call_llm_json(prompt, {})
    return result