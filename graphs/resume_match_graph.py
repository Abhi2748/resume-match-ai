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

embedder = SentenceTransformer("all-MiniLM-L6-v2")
llm = ChatOpenAI(model="gpt-4", temperature=0)

INITIAL_KEYS = [
    "resume_text", "jd_text"
]

# --- Node 1: Semantic Skill Matcher ---
def semantic_skill_matcher_node(state):
    # Debug: print extracted resume and JD data
    print("\n--- DEBUG: Resume Data ---")
    print(state["resume_data"])
    print("--- DEBUG: JD Data ---")
    print(state["jd_data"])
    # Use LLM for matching
    match_result = get_llm_matching(state["resume_data"], state["jd_data"])
    print("--- DEBUG: LLM Match Result ---")
    print(match_result)
    state["matched_skills"] = match_result.get("matched_skills", [])
    state["unmatched_skills"] = match_result.get("unmatched_skills", [])
    state["matched_responsibilities"] = match_result.get("matched_responsibilities", [])
    state["unmatched_responsibilities"] = match_result.get("unmatched_responsibilities", [])
    # For backward compatibility, also fill semantic_common_skills/gaps
    state["semantic_common_skills"] = state["matched_skills"]
    state["semantic_gaps"] = state["unmatched_skills"]
    # Semantic match score (simple ratio)
    total_skills = len(state["matched_skills"]) + len(state["unmatched_skills"])
    state["semantic_match_score"] = round(
        len(state["matched_skills"]) / total_skills if total_skills else 0.0, 2
    )
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
You are an intelligent career advisor. Based on the resume and job description, suggest realistic job roles, smart reasoning, career improvement tips, and verified skill verdicts for this candidate.

Resume:
{resume}

Job Description:
{jd}

Respond as JSON:
{{
  "realistic_roles": [...],
  "advisor_suggestions": [...],
  "career_improvement_tips": [...],
  "verified_skill_verdicts": [...]
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
        state["realistic_roles"] = advisor_data.get("realistic_roles", [])
        state["advisor_suggestions"] = advisor_data.get("advisor_suggestions", [])
        state["career_improvement_tips"] = advisor_data.get("career_improvement_tips", [])
        state["verified_skill_verdicts"] = advisor_data.get("verified_skill_verdicts", [])
    else:
        state["realistic_roles"] = []
        state["advisor_suggestions"] = []
        state["career_improvement_tips"] = []
        state["verified_skill_verdicts"] = []
    return state

# --- Final Output Node ---
def final_output_node(state):
    return {
        "semantic_common_skills": state.get("semantic_common_skills", []),
        "semantic_gaps": state.get("semantic_gaps", []),
        "semantic_match_score": state.get("semantic_match_score", 0.0),
        "matched_skills": state.get("matched_skills", []),
        "unmatched_skills": state.get("unmatched_skills", []),
        "matched_responsibilities": state.get("matched_responsibilities", []),
        "unmatched_responsibilities": state.get("unmatched_responsibilities", []),
        "llm_verdicts": state.get("llm_verdicts", []),
        "realistic_roles": state.get("realistic_roles", []),
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