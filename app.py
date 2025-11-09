# app.py
import streamlit as st
import pandas as pd
import numpy as np
import io
import pdfplumber
import docx
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from transformers import pipeline
import spacy

# ---------- Config ----------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # small & fast
nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer(EMBEDDING_MODEL)
# Summarizer (optional, may be slow on CPU)
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Example skill list (extend per job)
COMMON_SKILLS = ["python", "sql", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
                 "nlp", "computer vision", "docker", "aws", "spark", "hadoop", "git", "matplotlib"]

# ---------- Helpers ----------
def extract_text(file):
    name = file.name.lower()
    content = file.getvalue()  # safe: reads file content from memory buffer

    if name.endswith(".pdf"):
        text = ""
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text

    elif name.endswith(".docx"):
        doc = docx.Document(io.BytesIO(content))
        return "\n".join([p.text for p in doc.paragraphs])

    elif name.endswith(".txt"):
        try:
            return content.decode("utf-8", errors="ignore")
        except:
            return str(content)

    else:
        return ""

def extract_skills(text, skill_list=COMMON_SKILLS):
    text_lower = text.lower()
    found = []
    for skill in skill_list:
        # simple substring match; for robust matching use fuzzy/wb boundaries
        if re.search(r"\b" + re.escape(skill.lower()) + r"\b", text_lower):
            found.append(skill)
    return found

def estimate_years_experience(text):
    # naive: look for patterns like "X years" or "X yrs"
    years = []
    for match in re.finditer(r"(\d+(\.\d+)?)\s*(?:years|yrs|year)", text.lower()):
        try:
            years.append(float(match.group(1)))
        except:
            pass
    if years:
        return max(years)  # crude: assume max value is total experience
    # fallback: check for "experience" + number elsewhere -> 0
    return 0.0

def embed_texts(texts):
    # texts: list[str]
    return embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)

def cosine_sim(a, b):
    return float(cosine_similarity(a.reshape(1,-1), b.reshape(1,-1))[0,0])

def summarize_text(text, max_length=80):
    # short summarization; be careful with long text input
    try:
        out = summarizer(text, max_length=max_length, min_length=30, do_sample=False)
        return out[0]["summary_text"]
    except Exception:
        # fallback simple first 2 sentences
        return " ".join(re.split(r'(?<=[.!?]) +', text)[:2])

# ---------- Streamlit UI ----------
st.set_page_config(layout="wide", page_title="AI Resume Screener")
st.title("🧠 AI-Powered Resume Screening System")

with st.sidebar:
    st.header("Upload")
    uploaded_files = st.file_uploader("Upload resumes (PDF/DOCX/TXT)", accept_multiple_files=True, type=["pdf","docx","txt"])
    job_desc = st.text_area("Job description (paste here)", height=200)
    required_skills_input = st.text_input("Required skills (comma separated) — optional", value="")
    required_years = st.number_input("Required minimum years of experience", min_value=0.0, value=0.0, step=0.5)
    top_n = st.number_input("Top N candidates to show", min_value=1, max_value=50, value=5)

if st.button("Process"):
    if not uploaded_files:
        st.warning("Please upload at least one resume.")
        st.stop()
    if not job_desc.strip():
        st.warning("Please paste job description.")
        st.stop()
    # build required skill list
    required_skills = [s.strip().lower() for s in required_skills_input.split(",") if s.strip()] or []

    # Parse resumes
    candidates = []
    texts = []
    for f in uploaded_files:
        try:
            txt = extract_text(f)
        except Exception as e:
            st.error(f"Failed to read {f.name}: {e}")
            continue
        texts.append(txt)
        candidates.append({"file_name": f.name, "text": txt})

    # Embeddings
    st.info("Generating embeddings...")
    job_emb = embed_texts([job_desc])[0]
    resume_embs = embed_texts([c["text"] for c in candidates])

    rows = []
    for i, c in enumerate(candidates):
        txt = c["text"]
        emb = resume_embs[i]
        ES = cosine_sim(emb, job_emb)  # -1..1 but as similarity; should be in 0..1 if model normalized
        # normalize ES to 0..1
        ES = (ES + 1) / 2.0

        found_skills = extract_skills(txt, skill_list=COMMON_SKILLS + required_skills)
        # Skill match relative to required_skills if provided, otherwise relative to COMMON_SKILLS
        if required_skills:
            matched_required = sum(1 for s in required_skills if re.search(r"\b"+re.escape(s)+r"\b", txt.lower()))
            SM = matched_required / max(1, len(required_skills))
        else:
            SM = len(found_skills) / max(1, len(COMMON_SKILLS))

        years = estimate_years_experience(txt)
        XM = 1.0 if required_years == 0 else min(1.0, years / required_years)

        # simple keyword coverage
        keywords = re.findall(r'\b[A-Za-z0-9+\-_.]{2,}\b', job_desc.lower())
        keywords = list(set([k for k in keywords if len(k)>2]))[:50]  # reduce
        present = sum(1 for k in keywords if k in txt.lower())
        KC = present / max(1, len(keywords))

        score = 0.45*ES + 0.30*SM + 0.15*XM + 0.10*KC

        # summarization (short)
        short_summary = summarize_text(txt[:600])  # summarize first chunk

        rows.append({
            "file": c["file_name"],
            "score": score,
            "embedding_sim": ES,
            "skill_match": SM,
            "years_est": years,
            "exp_match": XM,
            "keyword_cov": KC,
            "skills_found": ", ".join(found_skills),
            "summary": short_summary
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("score", ascending=False)
    st.subheader("Top candidates")
    st.write(f"Showing top {min(len(df), top_n)} of {len(df)}")
    st.dataframe(df.head(top_n)[["file","score","embedding_sim","skill_match","years_est","skills_found"]])

    # Show expanded cards
    for idx, r in df.head(top_n).iterrows():
        with st.expander(f"{r['file']} — Score: {r['score']:.3f}"):
            st.markdown(f"**Summary:** {r['summary']}")
            st.markdown(f"**Score breakdown**: EmbSim={r['embedding_sim']:.3f}, SkillMatch={r['skill_match']:.3f}, ExpMatch={r['exp_match']:.3f}, KwCov={r['keyword_cov']:.3f}")
            st.markdown("**Skills found:** " + (r['skills_found'] or "—"))
            if st.button(f"Show raw text: {r['file']}", key=f"raw_{r['file']}"):
                original = next((c['text'] for c in candidates if c['file_name']==r['file']), "")
                st.text_area("Resume text", value=original, height=400)

    # export CSV
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download full results CSV", data=csv, file_name="resume_scores.csv", mime="text/csv")


