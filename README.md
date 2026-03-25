# AI-Resume-Screener
AI-Powered Resume Screening & Ranking System

Welcome to the **AI Resume Intelligence** project. This is a high-level recruitment automation tool designed to solve the problem of manual resume filtering. By treating text as mathematical vectors, the system provides an objective, data-driven ranking of candidates against any specific Job Description (JD).

 # Features

* **Automated Text Extraction:** Uses `PyMuPDF` to parse unstructured PDF data into normalized plaintext, handling complex layouts with high precision.
* **Semantic Similarity Engine:** Moves beyond simple keyword counting by using **TF-IDF (Term Frequency-Inverse Document Frequency)** and **Cosine Similarity** to measure the actual relevance between a CV and a JD.
* **Domain Intelligence:** Built-in technical taxonomy that automatically classifies candidates into 6+ industries including *AI/ML, Web Development, Data Analytics, Cloud/DevOps, and Cybersecurity*.
* **Skill-Gap Visualization:** Dynamically compares the required technical stack of a JD against the candidate's profile to flag missing critical skills.
* **Batch Analysis:** Optimized to process multiple resumes simultaneously, returning a sorted ranking table for high-volume hiring.

# Core Technologies & Logic

# 1. Natural Language Processing (NLP)
The system treats documents as vectors in a multidimensional space. 
* **TF-IDF:** This algorithm weighs the importance of a word. Rare technical terms (like "TensorFlow") are given more weight than common words (like "Experience"), ensuring the score reflects actual expertise.
* **Cosine Similarity:** The system calculates the "distance" between the Resume Vector and the JD Vector. A higher percentage represents a smaller mathematical angle between the two documents.



# 2. Technical Stack
* **Language:** Python 3.10+
* **Interface:** Streamlit (Web-based Dashboard)
* **AI Engine:** Scikit-Learn (Vectorization & Linear Algebra)
* **Parsing:** PyMuPDF (Fitz)
* **Data Handling:** Pandas & Regex (re)

 #📂 Project Structure

A professional "src/data" architecture is implemented to keep the core logic separate from the datasets and documentation.

```text
AI-Resume-Screener/
├── src/
│   ├── main.py          # Primary Application (Streamlit Dashboard)
│   └── app.py           # Core NLP Processing Logic & Functions
├── data/
│   └── samples/         # Anonymized PDF resumes for system testing
├── requirements.txt      # List of all Python dependencies
├── .gitignore            # Version control safety (ignores pycache/venv)
└── README.md             # Technical documentation & project guide
