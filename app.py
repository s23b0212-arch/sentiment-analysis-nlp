import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import plotly.express as px

# ======================
# PAGE CONFIG (A+ STYLE)
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
# DATASET
# ======================
data = {
    "review": [
        "This movie is amazing and I love it",
        "Worst movie ever, very boring",
        "It was okay not too bad",
        "I really enjoyed the story",
        "Terrible acting and bad plot",
        "Fantastic film with great actors",
        "Not good, not bad, just average",
        "I hate this movie so much",
        "Beautiful cinematography and great music",
        "Waste of time"
    ],
    "sentiment": [1,0,1,1,0,1,1,0,1,0]
}

df = pd.DataFrame(data)

# ======================
# EMOTION FUNCTION
# ======================
def get_emotion(text):
    text = text.lower()

    if any(word in text for word in ["amazing", "love", "great", "fantastic", "good"]):
        return "Joy 😊"
    elif any(word in text for word in ["hate", "worst", "boring", "terrible"]):
        return "Anger 😡"
    elif any(word in text for word in ["sad", "bad", "waste"]):
        return "Sadness 😢"
    else:
        return "Neutral 😐"

# ======================
# MODEL TRAINING
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    df["review"], df["sentiment"], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

# ======================
# SIDEBAR
# ======================
menu = st.sidebar.radio(
    "Navigation",
    ["Overview", "Predict", "Dashboard", "History"]
)

# ======================
# SESSION HISTORY
# ======================
if "history" not in st.session_state:
    st.session_state.history = []

# ======================
# OVERVIEW (FULL A+ CONTENT)
# ======================
if menu == "Overview":

    st.subheader("Problem Statement")

    st.info("""
In today's digital world, movie reviews are generated in large volumes on platforms such as IMDb and social media.

Manually analyzing these reviews is time-consuming, subjective, and inefficient.

This project builds an NLP-based system to automatically classify movie reviews into Positive or Negative sentiment using Machine Learning techniques.
""")

    st.subheader("Objectives")

    st.success("""
✔ To develop a sentiment analysis model using TF-IDF and Logistic Regression  
✔ To classify movie reviews into Positive and Negative categories  
✔ To detect basic emotional tone (Joy, Anger, Sadness, Neutral)  
✔ To visualize sentiment insights using interactive dashboards  
✔ To provide a real-time prediction interface for user input  
""")

    st.subheader("NLP Pipeline")

    st.write("""
Text Input → Preprocessing → TF-IDF Vectorization → Logistic Regression → Sentiment Prediction → Emotion Detection → Visualization Dashboard
""")

    st.subheader("🌍 Real-World Application")

    st.warning("""
This system can be used in platforms like Netflix, IMDb, and social media analytics tools to automatically understand user opinions and improve recommendation systems.
""")

# ======================
# PREDICT PAGE
# ======================
elif menu == "Predict":

    st.subheader("Try Sentiment Prediction")

    sample_reviews = [
        "This movie is amazing!",
        "Worst movie ever",
        "It was okay",
        "I love the acting",
        "Very boring film"
    ]

    user_input = st.selectbox("Select sample review", sample_reviews)

    if st.button("Analyze Sentiment"):

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
            st.metric("Confidence Score", f"{max(prob)*100:.2f}%")

        st.progress(int(max(prob)*100))

        # FIXED HISTORY (NOW INCLUDES EMOTION)
        st.session_state.history.append({
            "review": user_input,
            "sentiment": sentiment,
            "emotion": emotion
        })

# ======================
# DASHBOARD (FIXED + SAFE)
# ======================
elif menu == "Dashboard":

    st.header("Graphical Analytics Dashboard")

    if len(st.session_state.history) == 0:
        st.warning("No data yet. Please make predictions first.")
        st.stop()

    df_hist = pd.DataFrame(st.session_state.history)

    col1, col2 = st.columns(2)

    # SENTIMENT PIE
    with col1:
        sentiment_counts = df_hist["sentiment"].value_counts()
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            hole=0.5,
            title="Sentiment Breakdown"
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    # EMOTION BAR
    with col2:
        emotion_counts = df_hist["emotion"].value_counts()
        fig2 = px.bar(
            x=emotion_counts.index,
            y=emotion_counts.values,
            title="Emotion Frequency",
            color=emotion_counts.values
        )
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)

    # HEATMAP
    st.subheader("Sentiment & Emotion Correlation")
    matrix = pd.crosstab(df_hist["emotion"], df_hist["sentiment"])
    fig3 = px.imshow(matrix, text_auto=True, height=350)
    st.plotly_chart(fig3, use_container_width=True)

    # TREND
    st.subheader("Sentiment Trend Over Time")
    df_hist["index"] = range(len(df_hist))

    fig4 = px.line(
        df_hist,
        x="index",
        y=df_hist["sentiment"].map({"Positive 😊": 1, "Negative 😡": -1}),
        markers=True
    )
    fig4.update_layout(height=300)
    st.plotly_chart(fig4, use_container_width=True)

# ======================
# HISTORY
# ======================
elif menu == "History":

    st.subheader("Prediction History")

    if len(st.session_state.history) == 0:
        st.info("No predictions yet.")
    else:
        hist = pd.DataFrame(st.session_state.history)
        st.dataframe(hist)

        st.download_button(
            "⬇ Download History",
            hist.to_csv(index=False).encode("utf-8"),
            "history.csv",
            "text/csv"
        )
