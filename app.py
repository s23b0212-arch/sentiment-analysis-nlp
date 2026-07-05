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
st.set_page_config(page_title="Smart Movie Sentiment Dashboard", layout="wide")

st.title("🎬 Smart Movie Sentiment Analytics Dashboard")
st.markdown("NLP System using TF-IDF + Logistic Regression")

# ======================
# LOAD DATA
# ======================

df.columns = df.columns.str.strip().str.lower()
df['sentiment'] = df['sentiment'].map({'positive':1, 'negative':0})

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
# SIDEBAR
# ======================
menu = st.sidebar.radio("Navigation", ["🏠 Overview", "🎯 Predict", "📊 Analytics Dashboard"])

# ======================
# HOME
# ======================
if menu == "🏠 Overview":
    st.header("📌 Project Overview")

    st.write("""
This system performs **Movie Sentiment Analysis using NLP**.

### 🔥 Features:
- Accepts movie review text OR link input (simulated)
- Predicts sentiment (Positive / Negative)
- Shows confidence score
- Provides full analytics dashboard

### 🧠 NLP Process:
Text → TF-IDF → Logistic Regression → Prediction
""")

    st.info("👉 This system helps analyze audience opinion on movies automatically.")

# ======================
# PREDICTION PAGE
# ======================
elif menu == "🎯 Predict":
    st.header("💬 Sentiment Prediction Engine")

    input_type = st.radio("Choose input type:", ["Movie Review Text", "Movie Link (simulated)"])

    if input_type == "Movie Review Text":
        user_input = st.text_area("Enter movie review:")

    else:
        user_input = st.text_input("Paste movie link (we will extract text manually):")

    if st.button("Analyze Sentiment"):

        if user_input.strip() == "":
            st.warning("Please enter input!")
        else:
            vec = vectorizer.transform([user_input])
            pred = model.predict(vec)[0]
            prob = model.predict_proba(vec)[0]

            sentiment = "Positive 😊" if pred == 1 else "Negative 😞"

            st.success(f"Prediction: {sentiment}")
            st.info(f"Confidence Score: {max(prob)*100:.2f}%")

            st.progress(int(max(prob)*100))

# ======================
# DASHBOARD
# ======================
elif menu == "📊 Analytics Dashboard":
    st.header("📊 Model Performance Analytics")

    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # ===== METRICS =====
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{acc:.2f}")
    col2.metric("Precision", f"{prec:.2f}")
    col3.metric("Recall", f"{rec:.2f}")
    col4.metric("F1 Score", f"{f1:.2f}")

    # ===== CONFUSION MATRIX =====
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

    # ===== SENTIMENT DISTRIBUTION =====
    st.subheader("Sentiment Distribution")

    fig2, ax2 = plt.subplots()
    sns.countplot(x=df['sentiment'], ax=ax2)
    ax2.set_xticklabels(["Negative", "Positive"])

    st.pyplot(fig2)

    # ===== SAMPLE INSIGHT =====
    st.subheader("Insight")
    st.info("Model performs best on clear and simple movie reviews. Complex sarcasm may reduce accuracy.")
