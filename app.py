import streamlit as st
import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import plotly.express as px

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="Smart Sentiment Dashboard", layout="wide")

st.title("🎬 Smart Movie Sentiment Analytics Dashboard")
st.markdown("NLP System using TF-IDF + Logistic Regression + Emotion Analytics")

# ======================
# SAMPLE DATA (NO FILE ERROR)
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
    if any(w in text for w in ["happy", "good", "great", "love", "amazing"]):
        return "Joy 😊"
    elif any(w in text for w in ["bad", "worst", "boring", "hate", "waste"]):
        return "Anger 😡"
    elif any(w in text for w in ["sad", "cry", "disappointed"]):
        return "Sadness 😢"
    elif any(w in text for w in ["wow", "surprise", "unexpected"]):
        return "Surprise 😲"
    else:
        return "Neutral 😐"

df["emotion"] = df["review"].apply(get_emotion)

# ======================
# TRAIN MODEL
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
# HISTORY
# ======================
if "history" not in st.session_state:
    st.session_state.history = []

# ======================
# SIDEBAR MENU
# ======================
menu = st.sidebar.radio(
    "Navigation",
    ["🏠 Overview", "🎯 Predict", "📊 Dashboard", "📁 Upload CSV", "📜 History"]
)

# ======================
# OVERVIEW
# ======================
if menu == "🏠 Overview":
    st.header("📌 Project Overview")
    st.write("""
This system analyzes movie reviews using NLP.

✔ Sentiment Analysis  
✔ Emotion Detection  
✔ Interactive Dashboard  
✔ CSV Bulk Analysis  
✔ History Tracking  
""")

# ======================
# PREDICT
# ======================
elif menu == "🎯 Predict":
    st.header("💬 Sentiment Prediction")

    sample_reviews = [
        "This movie is amazing!",
        "Worst movie ever",
        "It was okay",
        "I love the acting",
        "Very boring film"
    ]

    user_input = st.selectbox("Try sample review:", sample_reviews)

    if st.button("Analyze"):
        vec = vectorizer.transform([user_input])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]

        sentiment = "Positive 😊" if pred == 1 else "Negative 😡"
        emotion = get_emotion(user_input)

        # EMOJI UI
        if pred == 1:
            st.success(f"😊 {sentiment}")
        else:
            st.error(f"😡 {sentiment}")

        st.info(f"Emotion: {emotion}")

        # CONFIDENCE BAR
        st.progress(int(max(prob) * 100))

        st.session_state.history.append({
            "review": user_input,
            "sentiment": sentiment,
            "emotion": emotion
        })

# ======================
# DASHBOARD (MODERN SMALL GRAPHS)
# ======================
elif menu == "📊 Dashboard":
    st.header("📊 Graphical Analytics Dashboard")

    df_hist = pd.DataFrame(st.session_state.history) if len(st.session_state.history) > 0 else df

    col1, col2 = st.columns(2)

    # SENTIMENT PIE (SMALL)
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

    # EMOTION BAR (SMALL)
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
# CSV UPLOAD
# ======================
elif menu == "📁 Upload CSV":
    st.header("📁 Bulk Analysis")

    file = st.file_uploader("Upload CSV (must have 'review')", type=["csv"])

    if file:
        data = pd.read_csv(file)

        vec = vectorizer.transform(data["review"])
        preds = model.predict(vec)

        data["sentiment"] = ["Positive 😊" if p == 1 else "Negative 😡" for p in preds]
        data["emotion"] = data["review"].apply(get_emotion)

        st.dataframe(data)

        st.download_button(
            "Download Results",
            data.to_csv(index=False).encode("utf-8"),
            "results.csv",
            "text/csv"
        )

# ======================
# HISTORY
# ======================
elif menu == "📜 History":
    st.header("Prediction History")

    if len(st.session_state.history) == 0:
        st.info("No history yet")
    else:
        hist = pd.DataFrame(st.session_state.history)
        st.dataframe(hist)

        st.download_button(
            "Download History",
            hist.to_csv(index=False).encode("utf-8"),
            "history.csv",
            "text/csv"
        )
