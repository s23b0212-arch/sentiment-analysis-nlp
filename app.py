import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Smart Sentiment Dashboard", layout="wide")

st.title("🎬 Smart Movie Sentiment Analytics Dashboard")
st.markdown("NLP System using TF-IDF + Logistic Regression + Emotion Analytics")

# =========================
# LOAD MODEL (IMPORTANT FIX)
# =========================
model = pickle.load(open("sentiment_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

# =========================
# SAMPLE DATA (NO CSV ERROR ANYMORE)
# =========================
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

# =========================
# SIDEBAR MENU
# =========================
menu = st.sidebar.radio(
    "Navigation",
    ["🏠 Overview", "💬 Predict", "📁 CSV Upload", "📊 Analytics", "📜 History"]
)

# =========================
# HISTORY STORAGE
# =========================
if "history" not in st.session_state:
    st.session_state.history = []

# =========================
# OVERVIEW
# =========================
if menu == "🏠 Overview":
    st.header("📌 Project Overview")

    st.success("AI system that detects sentiment from movie reviews")

    st.write("""
### Features:
- Sentiment Classification (Positive / Negative)
- TF-IDF + Logistic Regression Model
- Confusion Matrix & Metrics
- WordCloud Analysis
- CSV Bulk Prediction
- Download Results
- History Tracking
""")

# =========================
# PREDICT
# =========================
elif menu == "💬 Predict":
    st.header("💬 Live Sentiment Prediction")

    review = st.text_area("Enter movie review:")

    if st.button("Analyze"):
        if review.strip() == "":
            st.warning("Please enter text")
        else:
            vec = tfidf.transform([review])
            pred = model.predict(vec)[0]

            sentiment = "Positive 😊" if pred == 1 else "Negative 😡"

            st.subheader("Result")
            st.success(sentiment)

            st.session_state.history.append({
                "review": review,
                "sentiment": sentiment
            })

# =========================
# CSV UPLOAD
# =========================
elif menu == "📁 CSV Upload":
    st.header("📁 Bulk Review Analysis")

    file = st.file_uploader("Upload CSV file (must have column: review)", type=["csv"])

    if file:
        data = pd.read_csv(file)

        if "review" not in data.columns:
            st.error("CSV must contain 'review' column")
        else:
            vec = tfidf.transform(data["review"])
            preds = model.predict(vec)

            data["sentiment"] = ["Positive 😊" if p==1 else "Negative 😡" for p in preds]

            st.dataframe(data)

            st.download_button(
                "Download Results",
                data.to_csv(index=False).encode("utf-8"),
                "results.csv",
                "text/csv"
            )

# =========================
# ANALYTICS DASHBOARD
# =========================
elif menu == "📊 Analytics":
    st.header("📊 Model Performance")

    X = tfidf.transform(df["review"])
    y = df["sentiment"]

    y_pred = model.predict(X)

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{acc:.2f}")
    col2.metric("Precision", f"{prec:.2f}")
    col3.metric("Recall", f"{rec:.2f}")
    col4.metric("F1 Score", f"{f1:.2f}")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    st.subheader("Word Cloud")

    all_words = " ".join(df["review"])
    wc = WordCloud(width=800, height=400, background_color="white").generate(all_words)

    fig2, ax2 = plt.subplots()
    ax2.imshow(wc)
    ax2.axis("off")
    st.pyplot(fig2)

# =========================
# HISTORY
# =========================
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
