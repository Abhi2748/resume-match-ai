import os
from dotenv import load_dotenv
from utils.llm_utils import call_llm_json
from langchain_core.prompts import PromptTemplate

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing. Check your .env file.")

def extract_resume_info_with_llm(text):
    prompt = PromptTemplate.from_template("""
    From the following resume text, extract:
    - All relevant skills
    - All key project titles or descriptions

    Return as JSON:
    {{
      "skills": [...],
      "projects": [...]
    }}

    Resume:
    {text}
    """)
    result = call_llm_json(prompt, {"text": text})
    return result.get("skills", []), result.get("projects", [])

def extract_jd_info_with_llm(text):
    prompt = PromptTemplate.from_template("""
    From the following job description, extract:
    - All relevant skills
    - All key responsibilities or requirements

    Return as JSON:
    {{
      "skills": [...],
      "responsibilities": [...]
    }}

    JD:
    {text}
    """)
    result = call_llm_json(prompt, {"text": text})
    return result.get("skills", []), result.get("responsibilities", [])

def llm_match_skills_and_responsibilities(resume_skills, jd_skills, resume_projects, jd_responsibilities):
    prompt = PromptTemplate.from_template("""
You are an AI assistant trained to intelligently compare resumes and job descriptions.

Analyze the following:
- Resume Skills: {resume_skills}
- Resume Projects: {resume_projects}
- JD Skills: {jd_skills}
- JD Responsibilities: {jd_responsibilities}

Now identify:

1. ✅ Matching Points:
- Where resume aligns well with the job (skills, tools, projects, frameworks, domains)

2. ❌ Missing Points:
- Areas where the resume lacks something critical mentioned in the JD

3. 🎯 Suggested Job Roles (with reasons):
- Suggest realistic job titles this candidate can apply to
- Explain *why* they are suitable for each

Return as structured JSON:
{{
  "matched_skills": [...],
  "unmatched_skills": [...],
  "matched_responsibilities": [...],
  "unmatched_responsibilities": [...],
  "matching_points": [...],
  "missing_points": [...],
  "suggested_roles": [
    {{"role": "AI Engineer", "reason": "Candidate has experience with LLMs and deployment on AWS"}},
    ...
  ]
}}
""")

    variables = {
        "resume_skills": resume_skills,
        "resume_projects": resume_projects,
        "jd_skills": jd_skills,
        "jd_responsibilities": jd_responsibilities,
    }

    return call_llm_json(prompt, variables)


