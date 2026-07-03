import streamlit as st
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (run once in Streamlit)
nltk.download('stopwords')
nltk.download('wordnet')

# Load model and vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

# Initialize NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Text cleaning function (same as your Colab)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

# UI Design
st.set_page_config(page_title="Sentiment Analysis App", layout="centered")

st.title("🎬 Movie Review Sentiment Analysis")
st.write("Enter a movie review and the model will predict whether it is Positive or Negative.")

# Input box
user_input = st.text_area("Enter your movie review here:")

# Prediction button
if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review first.")
    else:
        cleaned = clean_text(user_input)
        vector = tfidf.transform([cleaned])
        prediction = model.predict(vector)

        if prediction[0] == 1:
            st.success("😊 Positive Review")
        else:
            st.error("😞 Negative Review")

# Footer
st.markdown("---")
st.markdown("Built using NLP (TF-IDF + Logistic Regression)")
