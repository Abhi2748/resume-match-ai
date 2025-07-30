from graphs.resume_match_graph import app
import fitz  # PyMuPDF

# Load resume
with open("sample_data/sample_resume.pdf", "rb") as f:
    doc = fitz.open(stream=f.read(), filetype="pdf")
    resume_text = "\n".join([page.get_text() for page in doc])

# Load JD
with open("sample_data/sample_jd.txt", "r", encoding="utf-8") as f:
    jd_text = f.read()

# Prepare inputs
inputs = {
    "resume_text": resume_text,
    "jd_text": jd_text
}

# Run graph
result = app.invoke(inputs)

# --- Cleaned Final Output ---
print("\n✅ Final Output\n")

print("🔍 Match Score:", result.get("semantic_match_score", "N/A"))

print("\n✅ Matching Points:")
for point in result.get("matching_points", []):
    print("•", point)

print("\n❌ Missing Points:")
for point in result.get("missing_points", []):
    print("•", point)

print("\n📈 Career Improvement Tips:")
for tip in result.get("career_improvement_tips", []):
    print("👉", tip)

print("\n🎯 Suggested Job Roles:")
roles = result.get("realistic_roles_with_reasons", [])
if isinstance(roles, list):
    for role in roles:
        print(f"🎯 {role.get('title')}: {role.get('reason')}")
else:
    print("⚠️ realistic_roles_with_reasons is not a list:", roles)

print("\n🧠 Verified Skill Verdicts:")
for item in result.get("verified_skill_verdicts", []):
    print("•", item)
