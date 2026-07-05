import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import plotly.express as px

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="NLP Sentiment Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================
# HEADER
# ======================
st.markdown("""
    <h1 style='text-align: center; color: #4F8BF9;'>
    🎬 NLP Sentiment Intelligence Dashboard
    </h1>
    <p style='text-align: center; color: gray;'>
    AI-powered Movie Review Analytics (TF-IDF + Logistic Regression + Emotion Detection)
    </p>
""", unsafe_allow_html=True)

# ======================
# DATASET (FIXED & STRONG)
# ======================
data = {
    "review": [
        "This movie is amazing",
        "I absolutely loved this film",
        "Fantastic story and great acting",
        "One of the best movies ever",
        "I really enjoyed this movie",
        "Beautiful cinematography and music",
        "Great film I will watch again",
        "Excellent and very enjoyable",
        "Amazing performance by actors",
        "I love this movie so much",

        "Worst movie ever",
        "I hate this film",
        "Very boring and waste of time",
        "Terrible acting and bad plot",
        "I did not enjoy this movie",
        "Awful experience watching this",
        "Bad movie not worth watching",
        "Disappointing and boring film",
        "I will never watch this again",
        "Not good at all"
    ],
    "sentiment": [
        1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0
    ]
}

df = pd.DataFrame(data)

# ======================
# EMOTION FUNCTION
# ======================
def get_emotion(text):
    text = text.lower()

    joy_words = ["love", "amazing", "great", "fantastic", "excellent", "best", "enjoy", "good"]
    anger_words = ["hate", "worst", "boring", "terrible", "bad", "waste", "awful", "disappointing"]

    if any(w in text for w in joy_words):
        return "Joy 😊"
    elif any(w in text for w in anger_words):
        return "Anger 😡"
    else:
        return "Neutral 😐"

# ======================
# MODEL
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    df["review"], df["sentiment"], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

# ======================
# HISTORY
# ======================
if "history" not in st.session_state:
    st.session_state.history = []

# ======================
# SIDEBAR
# ======================
menu = st.sidebar.radio(
    "Navigation",
    ["Overview", "Predict", "Dashboard", "History"]
)

# ======================
# OVERVIEW
# ======================
if menu == "Overview":

    st.subheader("📌 Problem Statement")

    st.info("""
Movie reviews are massive and hard to analyze manually.

This system automatically classifies reviews into Positive or Negative sentiment using Machine Learning.
""")

    st.subheader("🎯 Objectives")

    st.success("""
✔ Sentiment classification (Positive / Negative)  
✔ Emotion detection (Joy / Anger / Neutral)  
✔ TF-IDF + Logistic Regression model  
✔ Interactive dashboard visualization  
""")

    st.subheader("🧠 NLP Pipeline")

    st.write("""
Text → TF-IDF → Logistic Regression → Sentiment → Emotion → Dashboard
""")

# ======================
# PREDICT
# ======================
elif menu == "Predict":

    st.subheader("💬 Try Sentiment Prediction")

    sample_reviews = [
        "This movie is amazing!",
        "Worst movie ever",
        "It was okay",
        "I love the acting",
        "Very boring film"
    ]

    user_input = st.selectbox("Select review", sample_reviews)

    if st.button("Analyze"):

        vec = vectorizer.transform([user_input])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]

        sentiment = "Positive 😊" if pred == 1 else "Negative 😡"
        emotion = get_emotion(user_input)

        col1, col2 = st.columns(2)

        with col1:
            if pred == 1:
                st.success(f"{sentiment} | {emotion}")
            else:
                st.error(f"{sentiment} | {emotion}")

        with col2:
            st.metric("Confidence", f"{max(prob)*100:.2f}%")

        st.progress(int(max(prob)*100))

        st.session_state.history.append({
            "review": user_input,
            "sentiment": sentiment,
            "emotion": emotion
        })

# ======================
# DASHBOARD
# ======================
elif menu == "Dashboard":

    st.subheader("📊 Analytics Dashboard")

    if len(st.session_state.history) == 0:
        st.warning("No data yet.")
        st.stop()

    df_hist = pd.DataFrame(st.session_state.history)

    col1, col2 = st.columns(2)

    with col1:
        sentiment_counts = df_hist["sentiment"].value_counts()
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            hole=0.5,
            title="Sentiment Breakdown"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        emotion_counts = df_hist["emotion"].value_counts()
        fig2 = px.bar(
            x=emotion_counts.index,
            y=emotion_counts.values,
            title="Emotion Frequency"
        )
        st.plotly_chart(fig2, use_container_width=True)

# ======================
# HISTORY
# ======================
elif menu == "History":

    st.subheader("📜 Prediction History")

    if len(st.session_state.history) == 0:
        st.info("No predictions yet.")
    else:
        st.dataframe(pd.DataFrame(st.session_state.history))
