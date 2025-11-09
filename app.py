import streamlit as st
import pdfplumber, docx, io
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Resume Screener", page_icon="🧠", layout="wide")

# ---------- Helper: Extract Text ----------
def extract_text(file):
    name = file.name.lower()
    content = file.getvalue()
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
        return content.decode("utf-8", errors="ignore")
    else:
        return ""

# ---------- Streamlit UI ----------
st.title("🧠 AI-Powered Resume Screening System")
st.markdown("### Match resumes to a job description using TF-IDF similarity")

jd_text = st.text_area("📋 Paste Job Description", height=200)
uploaded_files = st.file_uploader("📂 Upload multiple resumes (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if st.button("Process") and uploaded_files and jd_text:
    resumes = []
    names = []
    for file in uploaded_files:
        resumes.append(extract_text(file))
        names.append(file.name)

    texts = [jd_text] + resumes
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform(texts)
    jd_vec = tfidf[0]
    resume_vecs = tfidf[1:]
    similarities = cosine_similarity(jd_vec, resume_vecs).flatten()

    df = pd.DataFrame({"Resume": names, "Similarity": similarities})
    df = df.sort_values(by="Similarity", ascending=False)

    st.success("✅ Screening Complete! Top Candidates:")
    st.dataframe(df)

    for i, row in df.iterrows():
        with st.expander(f"📄 {row['Resume']} — Similarity: {row['Similarity']:.2f}"):
            st.write(resumes[i])

    st.download_button("📥 Download Results CSV", df.to_csv(index=False), "results.csv")
