import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def match_score(resume, jd):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([resume, jd])
    return round(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0], 2)

st.set_page_config(page_title="Resume Matcher", page_icon="🧠", layout="centered")
st.title("📄 Resume vs Job Description Matcher 🔍")

resume_input = st.text_area("✍️ Paste Resume Text")
jd_input = st.text_area("📋 Paste Job Description Text")

if st.button("🚀 Match Now"):
    if resume_input and jd_input:
        score = match_score(resume_input, jd_input)
        st.success(f"✅ Match Score: {score*100:.2f}%")
        st.balloons()
    else:
        st.warning("Please fill in both fields!")