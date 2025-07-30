from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from sentence_transformers import SentenceTransformer
from utils.data_utils import (
    parse_resume_text,
    parse_jd_text,
    get_llm_matching
)
from utils.embedding_utils import get_mean_embedding, cosine_similarity
from utils.llm_utils import call_llm_json
from prompts import extract_resume_info_with_llm, extract_jd_info_with_llm, llm_match_skills_and_responsibilities
embedder = SentenceTransformer("all-mpnet-base-v2")
llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

INITIAL_KEYS = [
    "resume_text", "jd_text"
]

# --- Node 1: Semantic Skill Matcher ---
def semantic_skill_matcher_node(state):
    resume_skills = state["resume_data"].get("skills", [])
    resume_projects = state["resume_data"].get("Key projects", [])
    jd_skills = state["jd_data"].get("skills", [])
    jd_responsibilities = state["jd_data"].get("key_responsibilities", [])

    match_result = llm_match_skills_and_responsibilities(
        resume_skills, jd_skills, resume_projects, jd_responsibilities
    )
    # Extract relevant outputs
    state["matched_skills"] = match_result.get("matched_skills", [])
    state["unmatched_skills"] = match_result.get("unmatched_skills", [])
    state["matched_responsibilities"] = match_result.get("matched_responsibilities", [])
    state["unmatched_responsibilities"] = match_result.get("unmatched_responsibilities", [])
    state["matching_points"] = match_result.get("matching_points", [])
    state["missing_points"] = match_result.get("missing_points", [])

    # Optional: Simple semantic score (can improve later)
    total = len(state["matching_points"]) + len(state["missing_points"])
    state["semantic_match_score"] = round(len(state["matching_points"]) / total, 2) if total else 0.0

    return state

# --- Node 2: LLM Experience Verifier ---
llm_verifier_prompt = PromptTemplate.from_template("""
You are an expert resume reviewer. Answer the following questions about the candidate's experience.

Resume:
{resume}

Questions:
{questions}

Respond as a JSON list: [{{"question": "...", "verdict": "✅ or ❌", "reason": "..."}}]
""")

def llm_experience_verifier_node(state):
    resume = state["resume_text"]
    projects = state["resume_data"].get("Key projects", [])
    questions = [
        "Does this resume show hands-on experience with LLMs?",
        f"Does the project '{projects[0] if projects else ''}' qualify as LLM work?"
    ]
    result = llm.invoke(llm_verifier_prompt.format(resume=resume, questions="\n".join(questions)))
    import json, re
    json_text = re.search(r'\[.*\]', result.content, re.DOTALL)
    if json_text:
        state["llm_verdicts"] = json.loads(json_text.group())
    else:
        state["llm_verdicts"] = []
    return state

# --- Node 3: Intelligent Advisor ---
advisor_prompt = PromptTemplate.from_template("""
You are an intelligent career advisor. Based on the resume and job description, suggest realistic job roles, reasons why they are a good fit, improvement tips, and skill verification.

Resume:
{resume}

Job Description:
{jd}

Respond in this JSON format:
{{
  "realistic_roles_with_reasons": [
    {{
      "title": "Full Stack Developer",
      "reason": "Candidate has strong backend and moderate frontend skills."
    }},
    {{
      "title": "DevOps Engineer",
      "reason": "Experience with Docker, Kubernetes, CI/CD pipelines, and cloud deployment."
    }}
  ],
  "advisor_suggestions": [
    "Contribute to open-source projects involving Flask or FastAPI.",
    "Enhance frontend skills using React or Vue."
  ],
  "career_improvement_tips": [
    "Get certified in AWS.",
    "Work on projects involving full-stack systems."
  ],
  "verified_skill_verdicts": [
    "Python: Verified",
    "React: Not Verified"
  ]
}}
""")


def intelligent_advisor_node(state):
    resume = state["resume_text"]
    jd = state["jd_text"]
    result = llm.invoke(advisor_prompt.format(resume=resume, jd=jd))
    import json, re
    json_text = re.search(r'\{.*\}', result.content, re.DOTALL)
    if json_text:
        advisor_data = json.loads(json_text.group())
        state["realistic_roles_with_reasons"] = advisor_data.get("realistic_roles_with_reasons", [])
        state["advisor_suggestions"] = advisor_data.get("advisor_suggestions", [])
        state["career_improvement_tips"] = advisor_data.get("career_improvement_tips", [])
        state["verified_skill_verdicts"] = advisor_data.get("verified_skill_verdicts", [])
    else:
        state["realistic_roles_with_reasons"] = []
        state["advisor_suggestions"] = []
        state["career_improvement_tips"] = []
        state["verified_skill_verdicts"] = []
    return state

# --- Final Output Node ---
def final_output_node(state):
    return {
        "semantic_match_score": state.get("semantic_match_score", 0.0),
        "matching_points": state.get("matching_points", []),
        "missing_points": state.get("missing_points", []),
        "realistic_roles_with_reasons": state.get("realistic_roles_with_reasons", []),
        "advisor_suggestions": state.get("advisor_suggestions", []),
        "career_improvement_tips": state.get("career_improvement_tips", []),
        "verified_skill_verdicts": state.get("verified_skill_verdicts", [])
    }

# --- Build Graph ---
workflow = StateGraph(dict)
workflow.add_node("parse_resume", RunnableLambda(lambda state: {**state, "resume_data": parse_resume_text(state["resume_text"]) }))
workflow.add_node("parse_jd", RunnableLambda(lambda state: {**state, "jd_data": parse_jd_text(state["jd_text"]) }))
workflow.add_node("semantic_skill_matcher", RunnableLambda(semantic_skill_matcher_node))
workflow.add_node("llm_experience_verifier", RunnableLambda(llm_experience_verifier_node))
workflow.add_node("intelligent_advisor", RunnableLambda(intelligent_advisor_node))
workflow.add_node("final_output", RunnableLambda(final_output_node))

workflow.set_entry_point("parse_resume")
workflow.add_edge("parse_resume", "parse_jd")
workflow.add_edge("parse_jd", "semantic_skill_matcher")
workflow.add_edge("semantic_skill_matcher", "llm_experience_verifier")
workflow.add_edge("llm_experience_verifier", "intelligent_advisor")
workflow.add_edge("intelligent_advisor", "final_output")
workflow.add_edge("final_output", END)

app = workflow.compile()
__all__ = ["app", "INITIAL_KEYS"]
