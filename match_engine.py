from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def get_embeddings(text1, text2):
    model = SentenceTransformer('all-mpnet-base-v2')  
    embeddings = model.encode([text1, text2])
    return embeddings

def compute_match_score(resume_text, jd_text):
    resume_emb, jd_emb = get_embeddings(resume_text, jd_text)
    score = cosine_similarity([resume_emb], [jd_emb])[0][0]
    return round(score * 100, 2)  # As percentage

def compare_skills(resume_skills, jd_skills):
    resume_set = set([s.lower() for s in resume_skills])
    jd_set = set([s.lower() for s in jd_skills])
    
    common = sorted(list(resume_set & jd_set))
    missing = sorted(list(jd_set - resume_set))

    return {
        "common_skills": common,
        "missing_skills": missing
    }
