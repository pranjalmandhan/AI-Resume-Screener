import streamlit as st
import fitz
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

st.set_page_config(page_title="AI Resume Analytics", layout="wide")

# This is the "Brain" that helps the AI understand what the CV is actually about
DOMAINS = {
    "AI / Machine Learning": ['tensorflow', 'pytorch', 'nlp', 'machine learning', 'deep learning', 'keras', 'scikit-learn', 'rag', 'llm', 'transformers', 'bert', 'vision'],
    "Web Development": ['html', 'css', 'javascript', 'react', 'node.js', 'bootstrap', 'django', 'flask', 'web', 'frontend', 'backend', 'typescript', 'next.js'],
    "Data Analytics": ['tableau', 'power bi', 'excel', 'data visualization', 'statistics', 'cleaning', 'dashboard', 'bi', 'looker'],
    "Cloud & DevOps": ['aws', 'azure', 'docker', 'kubernetes', 'terraform', 'jenkins', 'linux', 'git', 'ci/cd', 'ansible', 'gcp'],
    "Cybersecurity": ['networking', 'firewall', 'penetration testing', 'wireshark', 'hacking', 'siem', 'encryption', 'owasp', 'security'],
    "Mobile Development": ['flutter', 'kotlin', 'swift', 'android', 'ios', 'react native', 'dart']
}

def extract_text(file_bytes):
    text = ""
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text.lower()

def detect_actual_specialty(text):
    # Counts keyword matches for each industry domain
    scores = {domain: 0 for domain in DOMAINS}
    for domain, keywords in DOMAINS.items():
        for word in keywords:
            if re.search(rf'\b{word}\b', text):
                scores[domain] += 1
    
    # Identify the highest scoring domain
    best_match = max(scores, key=scores.get)
    if scores[best_match] == 0:
        return "General / Non-Technical"
    return best_match

st.title(" AI Smart-Field Resume Screener")
st.info("This tool identifies the candidate's actual specialty and flags mismatches with your requirements.")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader(" Step 1: Your Requirements")
    job_desc = st.text_area(
        "Paste the Job Description here:", 
        placeholder="Example: Looking for a Web Developer with React experience...",
        height=250
    )

with col2:
    st.subheader(" Step 2: Upload Resumes")
    uploaded_files = st.file_uploader(
        "Choose PDF files", 
        type="pdf", 
        accept_multiple_files=True
    )

st.divider()

if st.button(" Run Smart Analysis", use_container_width=True):
    if not job_desc or not uploaded_files:
        st.warning("Please provide both a Job Description and Resumes.")
    else:
        results = []
        # Detect what the Job Description is actually asking for
        required_field = detect_actual_specialty(job_desc)
        st.write(f" **Detected Requirement:** This JD is for a **{required_field}** position.")
        
        for file in uploaded_files:
            text = extract_text(file.read())
            
            # AI Similarity Score
            cv_vec = TfidfVectorizer()
            matrix = cv_vec.fit_transform([text, job_desc])
            score = cosine_similarity(matrix)[0][1] * 100
            
            # Detect the CV's actual strength
            actual_field = detect_actual_specialty(text)
            
            # Smart Validation Logic
            if actual_field != required_field:
                status = f" MISMATCH: Candidate is specialized in {actual_field}."
            elif score < 30:
                status = " WEAK MATCH: Correct field, but lacks required depth."
            else:
                status = " GOOD MATCH: Suitable for this role."

            results.append({
                "Candidate Name": file.name,
                "Actual Specialty": actual_field,
                "Match Score (%)": round(score, 2),
                "AI Recommendation": status
            })
        
        df = pd.DataFrame(results).sort_values(by="Match Score (%)", ascending=False)
        st.subheader(" Domain Validation Results")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.success(f"Successfully validated {len(uploaded_files)} resumes.")