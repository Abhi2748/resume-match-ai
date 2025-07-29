from utils.data_utils import parse_resume, parse_job_description

resume_path = "sample_data/sample_resume.pdf"
jd_path = "sample_data/sample_jd.txt"

# Read JD text
with open(jd_path, "r", encoding="utf-8") as f:
    jd_text = f.read()

# Parse
resume_data = parse_resume(resume_path)
jd_data = parse_job_description(jd_text)

print("🧾 Resume Skills:", resume_data["skills"])
print("📄 JD Skills:", jd_data["skills"])
print("✅ Resume Snippet:", resume_data["raw_text"][:300], "...")
print("✅ JD Snippet:", jd_data["raw_text"][:300], "...")

from match_engine import compute_match_score, compare_skills

score = compute_match_score(resume_data["raw_text"], jd_data["raw_text"])
print(f"\n📊 Resume–JD Match Score: {score}%")

skill_match = compare_skills(resume_data["skills"], jd_data["skills"])
print("✅ Common Skills:", skill_match["common_skills"])
print("❌ Missing Skills:", skill_match["missing_skills"])

from prompts import get_suggestions

print("\n🤖 LLM Suggestions:")
print(get_suggestions(resume_data["raw_text"], jd_data["raw_text"]))
