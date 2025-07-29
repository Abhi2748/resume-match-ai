from graphs.resume_match_graph import app
import fitz  # PyMuPDF

# Load resume
with open("sample_data/sample_resume.pdf", "rb") as f:
    doc = fitz.open(stream=f.read(), filetype="pdf")
    resume_text = "\n".join([page.get_text() for page in doc])

# Load JD
with open("sample_data/sample_jd.txt", "r", encoding="utf-8") as f:
    jd_text = f.read()

inputs = {
    "resume_text": resume_text,
    "jd_text": jd_text
}

# Run graph
result = app.invoke(inputs)

print("\n✅ Final Output:")
for k, v in result.items():
    if isinstance(v, (list, dict)):
        print(f"\n🔹 {k.upper()}:")
        if isinstance(v, list):
            for i in v:
                print("•", i)
        else:
            for sub_k, sub_v in v.items():
                print(f"{sub_k}: {sub_v}")
    else:
        print(f"{k}: {v}")

# Friendly summaries
print("\n🔍 Match Score:", result.get("semantic_match_score", "N/A"))
print("✅ Matched Skills:", result.get("semantic_common_skills", []))
print("❌ Gaps:", result.get("semantic_gaps", []))

print("\n🔄 Responsibility Match:")
resp = result.get("responsibility_match", {})
print("Matched:", resp.get("matched", []))
print("Unmatched:", resp.get("unmatched", []))

print("\n🧠 Verified Skill Verdicts:")
for skill in result.get("verified_skills", []):
    print(f"• {skill['skill']}: {skill['verdict']} – {skill['reason']}")

print("\n📈 Career Improvement Tips:")
for tip in result.get("improvement_suggestions", []):
    print(f"👉 {tip}")

print("\n🎯 Suggested Job Roles:")
roles = result.get("realistic_job_roles", [])
if isinstance(roles, list):
    for role in roles:
        print("🎯", role)
else:
    print("⚠️ realistic_job_roles is not a list:", roles)