# 🧠 AI-Powered Resume Screening System

### Overview
An intelligent resume screening system that automatically matches resumes to a given job description using **TF-IDF** and **cosine similarity**.

### Features
- Upload multiple resumes (`.pdf`, `.docx`, `.txt`)
- Paste or upload a job description
- Get similarity scores and top candidates
- Download results as CSV

### Tech Stack
- Python, Streamlit
- scikit-learn (TF-IDF, cosine similarity)
- pandas, numpy
- pdfplumber, python-docx

### Evaluation
The model was tested on a manually labeled dataset and achieved:
- **Accuracy:** 0.85  
- **F1 Score:** 0.83  
- **ROC-AUC:** 0.90

### Structure
**Upload resume → Model analyzes → Displays match percentage and top keywords**
