import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# --- 1. APP CONFIG ---
st.set_page_config(page_title="AI Resume Screener", layout="wide")

# --- 2. LOGIC FUNCTIONS ---
def extract_text(file_bytes):
    """Extracts text from a PDF file provided as bytes."""
    text = ""
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text.lower()

def get_match_score(resume_text, job_description):
    """Calculates TF-IDF similarity score."""
    content = [resume_text, job_description]
    cv = TfidfVectorizer()
    matrix = cv.fit_transform(content)
    similarity_matrix = cosine_similarity(matrix)
    return similarity_matrix[0][1] * 100

def get_missing_keywords(resume_text, job_description):
    """Checks for specific technical keywords."""
    jd_words = set(re.findall(r'\w+', job_description.lower()))
    resume_words = set(re.findall(r'\w+', resume_text.lower()))
    skills_to_check = {'python', 'machine', 'learning', 'data', 'analytics', 'pandas', 'matplotlib', 'sql', 'scikit', 'ai'}
    missing = (jd_words & skills_to_check) - resume_words
    return missing

# --- 3. STREAMLIT UI ---
st.title("📄 AI Resume Matcher")
st.markdown("Compare your resume against any Job Description instantly.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Step 1: Job Description")
    job_desc = st.text_area("Paste the job requirements here:", height=250, 
                            placeholder="Looking for a Python Developer with SQL...")

with col2:
    st.subheader("Step 2: Upload Resume")
    uploaded_file = st.file_uploader("Upload your CV (PDF)", type="pdf")

st.divider()

if st.button("Calculate Match Score", use_container_width=True):
    if not job_desc or not uploaded_file:
        st.warning("Please provide both a Job Description and a Resume.")
    else:
        # Run the analysis
        with st.spinner("Analyzing your profile..."):
            resume_content = extract_text(uploaded_file.read())
            score = get_match_score(resume_content, job_desc)
            missing = get_missing_keywords(resume_content, job_desc)
            
            # Display Results
            st.metric(label="Match Confidence", value=f"{score:.2f}%")
            
            if missing:
                st.error(f"❌ Missing Keywords: {', '.join(missing)}")
                st.info("💡 Tip: Add these keywords to your resume to increase your score!")
            else:
                st.success("✅ Great! Your resume matches all key technical skills found.")