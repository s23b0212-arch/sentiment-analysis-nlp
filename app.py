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
# LOAD DATA (FIXED SAFE VERSION)
# ======================
@st.cache_data
def load_data():
    df = pd.read_csv("IMDB Dataset.csv")   # make sure file exists in repo
    df.columns = df.columns.str.strip().str.lower()
    df = df.rename(columns={"review": "review", "sentiment": "sentiment"})
    df['sentiment'] = df['sentiment'].map({'positive':1, 'negative':0})
    return df

df = load_data()

# ======================
# MODEL TRAINING
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['sentiment'], test_size=0.2, random_state=42
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
    if any(w in text for w in ["happy", "good", "great", "amazing", "love"]):
        return "Joy 😊"
    elif any(w in text for w in ["bad", "worst", "boring", "hate"]):
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
# SIDEBAR
# ======================
menu = st.sidebar.radio("Navigation", ["🏠 Overview", "🎯 Predict", "📁 Upload CSV", "📊 Dashboard", "📜 History"])

# ======================
# OVERVIEW
# ======================
if menu == "🏠 Overview":
    st.header("📌 Project Overview")

    st.write("""
This system performs **Movie Sentiment Analysis + Emotion Detection** using NLP.

### 🔥 Features:
- Manual review prediction
- CSV bulk analysis
- Sentiment classification (Positive / Negative)
- Emotion detection (Joy, Anger, Sadness, Surprise)
- History tracking
- Download results
- Analytics dashboard

### 🧠 Model:
TF-IDF + Logistic Regression
""")

# ======================
# PREDICT SINGLE REVIEW
# ======================
elif menu == "🎯 Predict":
    st.header("💬 Single Review Analysis")

    user_input = st.text_area("Enter movie review:")

    if st.button("Analyze"):
        if user_input.strip() != "":

            vec = vectorizer.transform([user_input])
            pred = model.predict(vec)[0]
            prob = model.predict_proba(vec)[0]

            sentiment = "Positive 😊" if pred == 1 else "Negative 😡"
            emotion = get_emotion(user_input)

            st.success(f"Sentiment: {sentiment}")
            st.info(f"Emotion: {emotion}")
            st.progress(int(max(prob)*100))

            # save history
            st.session_state.history.append({
                "review": user_input,
                "sentiment": sentiment,
                "emotion": emotion
            })

        else:
            st.warning("Please enter review!")

# ======================
# CSV UPLOAD
# ======================
elif menu == "📁 Upload CSV":
    st.header("📁 Bulk Sentiment Analysis")

    file = st.file_uploader("Upload CSV file (column: review)", type=["csv"])

    if file:
        data = pd.read_csv(file)

        if "review" not in data.columns:
            st.error("CSV must contain 'review' column")
        else:
            vec = vectorizer.transform(data["review"])
            preds = model.predict(vec)

            data["sentiment"] = ["Positive 😊" if p==1 else "Negative 😡" for p in preds]
            data["emotion"] = data["review"].apply(get_emotion)

            st.dataframe(data)

            csv = data.to_csv(index=False).encode("utf-8")

            st.download_button(
                "📥 Download Results",
                csv,
                "sentiment_results.csv",
                "text/csv"
            )

# ======================
# DASHBOARD
# ======================
elif menu == "📊 Dashboard":
    st.header("📊 Analytics Dashboard")

    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc:.2f}")
    col2.metric("Precision", f"{prec:.2f}")
    col3.metric("Recall", f"{rec:.2f}")
    col4.metric("F1 Score", f"{f1:.2f}")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # Sentiment distribution
    st.subheader("Sentiment Distribution")
    fig2, ax2 = plt.subplots()
    sns.countplot(x=df["sentiment"], ax=ax2)
    ax2.set_xticklabels(["Negative", "Positive"])
    st.pyplot(fig2)

# ======================
# HISTORY
# ======================
elif menu == "📜 History":
    st.header("📜 Prediction History")

    if len(st.session_state.history) == 0:
        st.info("No history yet")
    else:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df)

        csv = hist_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Download History",
            csv,
            "history.csv",
            "text/csv"
        )
