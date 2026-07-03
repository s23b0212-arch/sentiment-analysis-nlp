import streamlit as st
import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (only if missing)
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

try:
    lemmatizer = WordNetLemmatizer()
    lemmatizer.lemmatize("movies")
except LookupError:
    nltk.download("wordnet")
    lemmatizer = WordNetLemmatizer()

# Load model and TF-IDF vectorizer
model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf.pkl")

# Text preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

# Streamlit page
st.set_page_config(page_title="Movie Review Sentiment Analysis", page_icon="🎬")

st.title("🎬 Movie Review Sentiment Analysis")
st.write("This application predicts whether a movie review is **Positive** or **Negative** using Natural Language Processing (TF-IDF + Logistic Regression).")

review = st.text_area("Enter a movie review:")

if st.button("Predict"):

    if review.strip() == "":
        st.warning("Please enter a movie review.")
    else:

        cleaned = clean_text(review)
        vector = tfidf.transform([cleaned])

        prediction = model.predict(vector)[0]
        probability = model.predict_proba(vector)

        confidence = probability.max() * 100

        st.subheader("Prediction")

        if prediction == 1:
            st.success("😊 Positive Review")
        else:
            st.error("😞 Negative Review")

        st.write(f"**Confidence:** {confidence:.2f}%")

st.markdown("---")
st.caption("Developed using Python, NLTK, TF-IDF and Logistic Regression")
