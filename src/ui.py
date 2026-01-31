import streamlit as st
import requests
import os

# Use the Docker service name 'api' or fallback to localhost
API_URL = os.getenv("API_URL", "http://api:8000")

st.set_page_config(page_title="Sentiment Analyzer", page_icon="🐦")
st.title("🐦 Twitter Sentiment Analysis (BERT)")

user_input = st.text_area("Enter a tweet to analyze:", placeholder="I am having a great day!")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        try:
            response = requests.post(f"{API_URL}/predict", json={"text": user_input})
            if response.status_code == 200:
                result = response.json()
                sentiment = result['sentiment']
                conf = result['confidence']
                
                if sentiment == "positive":
                    st.success(f"**Positive** (Confidence: {conf*100:.1f}%)")
                else:
                    st.error(f"**Negative** (Confidence: {conf*100:.1f}%)")
            else:
                st.error("API returned an error. Check logs.")
        except Exception as e:
            st.error(f"Could not connect to API: {e}")
    else:
        st.warning("Please enter some text first.")