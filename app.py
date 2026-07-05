import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="Smart Sentiment Dashboard", layout="wide")

st.title("🎬 Smart Movie Sentiment Analytics Dashboard")
st.markdown("NLP System using TF-IDF + Logistic Regression + Emotion Detection")

# ======================
# MINI DATASET (NO CSV NEEDED)
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
# MODEL TRAINING
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['sentiment'], test_size=0.3, random_state=42
)

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

# ======================
# EMOTION FUNCTION
# ======================
def get_emotion(text):
    text = text.lower()
    if any(w in text for w in ["happy", "good", "great", "love", "amazing", "enjoyed"]):
        return "Joy 😊"
    elif any(w in text for w in ["bad", "worst", "boring", "hate", "terrible"]):
        return "Anger 😡"
    elif any(w in text for w in ["sad", "cry", "disappointed"]):
        return "Sadness 😢"
    elif any(w in text for w in ["wow", "surprise", "unexpected"]):
        return "Surprise 😲"
    else:
        return "Neutral 😐"

# ======================
# SESSION HISTORY
# ======================
if "history" not in st.session_state:
    st.session_state.history = []

# ======================
# SIDEBAR MENU
# ======================
menu = st.sidebar.radio("Navigation", ["🏠 Overview", "🎯 Predict", "📊 Dashboard", "📜 History"])

# ======================
# OVERVIEW
# ======================
if menu == "🏠 Overview":
    st.header("📌 Project Overview")

    st.write("""
This is a **Movie Sentiment Analysis System using NLP**.

### Features:
- Sentiment Prediction (Positive / Negative)
- Emotion Detection (Joy, Anger, Sadness, etc.)
- Machine Learning Model (TF-IDF + Logistic Regression)
- Performance Dashboard
- Prediction History

### NLP Pipeline:
Text → Cleaning → TF-IDF → ML Model → Prediction
""")

# ======================
# PREDICT
# ======================
elif menu == "🎯 Predict":
    st.header("💬 Enter Movie Review")

    user_input = st.text_area("Write your review:")

    if st.button("Analyze"):
        if user_input.strip():

            vec = vectorizer.transform([user_input])
            pred = model.predict(vec)[0]
            prob = model.predict_proba(vec)[0]

            sentiment = "Positive 😊" if pred == 1 else "Negative 😡"
            emotion = get_emotion(user_input)

            st.success(f"Sentiment: {sentiment}")
            st.info(f"Emotion: {emotion}")
            st.progress(int(max(prob) * 100))

            st.session_state.history.append({
                "review": user_input,
                "sentiment": sentiment,
                "emotion": emotion
            })

        else:
            st.warning("Please enter a review")

# ======================
# DASHBOARD
# ======================
elif menu == "📊 Dashboard":
    st.header("📊 Model Performance Metrics")

    y_pred = model.predict(X_test_vec)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
    col2.metric("Precision", f"{precision_score(y_test, y_pred):.2f}")
    col3.metric("Recall", f"{recall_score(y_test, y_pred):.2f}")
    col4.metric("F1 Score", f"{f1_score(y_test, y_pred):.2f}")

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.subheader("Sentiment Distribution")

    fig2, ax2 = plt.subplots()
    sns.countplot(x=df["sentiment"], ax=ax2)
    ax2.set_xticklabels(["Negative", "Positive"])
    st.pyplot(fig2)

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
