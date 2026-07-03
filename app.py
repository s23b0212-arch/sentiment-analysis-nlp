import streamlit as st
import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="IMDb Sentiment Analyzer",
    layout="centered"
)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("🎬 About This App")
st.sidebar.write("NLP Sentiment Analysis System")
st.sidebar.write("Model: Logistic Regression")
st.sidebar.write("Vectorizer: TF-IDF")
st.sidebar.write("Dataset: IMDb Reviews")

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf.pkl")

# -----------------------------
# DOWNLOAD NLTK DATA
# -----------------------------
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# -----------------------------
# TEXT CLEANING FUNCTION
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

# -----------------------------
# TITLE
# -----------------------------
st.title("🎬 IMDb Movie Review Sentiment Analyzer")
st.write("Type a movie review and the system will predict sentiment (Positive / Negative).")

# -----------------------------
# SAMPLE BUTTONS
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    if st.button("😊 Try Positive Example"):
        st.session_state.input_text = "This movie was absolutely amazing and I loved it"

with col2:
    if st.button("😞 Try Negative Example"):
        st.session_state.input_text = "This movie was boring and waste of time"

# -----------------------------
# INPUT BOX
# -----------------------------
user_input = st.text_area(
    "Enter Movie Review:",
    value=st.session_state.get("input_text", "")
)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review first.")
    else:
        cleaned = clean_text(user_input)
        vector = tfidf.transform([cleaned])

        prediction = model.predict(vector)
        proba = model.predict_proba(vector)[0]

        st.subheader("Result")

        if prediction[0] == 1:
            st.success("😊 Positive Review")
        else:
            st.error("😞 Negative Review")

        st.subheader("Confidence Score")

        st.write(f"Positive: {proba[1]*100:.2f}%")
        st.write(f"Negative: {proba[0]*100:.2f}%")

        st.progress(float(proba[1]))

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("Built with NLP ❤️ | TF-IDF + Logistic Regression")
