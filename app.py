import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from wordcloud import WordCloud

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="Movie Sentiment Dashboard", layout="wide")

st.title("🎬 Movie Sentiment & Emotion Analytics Dashboard")
st.markdown("NLP using TF-IDF + Logistic Regression")

# ======================
# LOAD DATA (SAFE)
# ======================
@st.cache_data
def load_data():
    df = pd.read_csv("IMDB Dataset.csv")
    df.columns = df.columns.str.strip().str.lower()
    df = df.rename(columns={"review": "text"})
    df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})
    return df

df = load_data()

# ======================
# MODEL TRAINING
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["sentiment"], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train)

# ======================
# SIDEBAR
# ======================
menu = st.sidebar.radio(
    "Navigation",
    ["🏠 Overview", "🧠 Predict Sentiment", "📊 Analytics Dashboard"]
)

# ======================
# OVERVIEW
# ======================
if menu == "🏠 Overview":
    st.header("📌 Project Overview")

    st.write("""
This system analyzes movie reviews using NLP.

### Features:
- Sentiment prediction (Positive / Negative)
- CSV upload analysis
- Live dashboard visualization
- Word cloud insights

### NLP Pipeline:
Text → TF-IDF → Logistic Regression → Prediction
""")

# ======================
# PREDICT PAGE
# ======================
elif menu == "🧠 Predict Sentiment":

    st.header("💬 Try Sentiment Analysis")

    option = st.radio("Choose input type:", ["Write Review", "Upload CSV", "Try Samples"])

    texts = []

    # 1. Manual input
    if option == "Write Review":
        user_text = st.text_area("Enter movie review:")
        if user_text:
            texts = [user_text]

    # 2. Upload CSV
    elif option == "Upload CSV":
        file = st.file_uploader("Upload CSV file (must have a 'review' column)", type=["csv"])
        if file:
            uploaded_df = pd.read_csv(file)
            uploaded_df.columns = uploaded_df.columns.str.lower()
            texts = uploaded_df["review"].dropna().tolist()

    # 3. Sample data
    else:
        texts = [
            "This movie was amazing and fantastic!",
            "Worst movie ever, boring and slow",
            "It was okay not too bad but not great"
        ]

    # Prediction
    if st.button("Analyze") and len(texts) > 0:

        vec = vectorizer.transform(texts)
        preds = model.predict(vec)

        results = []

        for t, p in zip(texts, preds):
            sentiment = "Positive 😊" if p == 1 else "Negative 😞"
            results.append([t, sentiment])

        result_df = pd.DataFrame(results, columns=["Review", "Sentiment"])

        st.dataframe(result_df)

# ======================
# DASHBOARD
# ======================
elif menu == "📊 Analytics Dashboard":

    st.header("📊 Model Performance Dashboard")

    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # METRICS
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{acc:.2f}")
    col2.metric("Precision", f"{prec:.2f}")
    col3.metric("Recall", f"{rec:.2f}")
    col4.metric("F1-score", f"{f1:.2f}")

    # CONFUSION MATRIX
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # WORD CLOUD
    st.subheader("Word Cloud")

    all_text = " ".join(df["text"].astype(str))
    wc = WordCloud(width=800, height=400, background_color="white").generate(all_text)

    fig2, ax2 = plt.subplots()
    ax2.imshow(wc, interpolation="bilinear")
    ax2.axis("off")
    st.pyplot(fig2)

    # SENTIMENT DISTRIBUTION
    st.subheader("Sentiment Distribution")

    fig3, ax3 = plt.subplots()
    sns.countplot(x=df["sentiment"], ax=ax3)
    ax3.set_xticklabels(["Negative", "Positive"])
    st.pyplot(fig3)
