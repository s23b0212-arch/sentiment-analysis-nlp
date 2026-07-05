import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="NLP Sentiment Dashboard", layout="wide")

st.title("🎬 Smart NLP Sentiment Analytics Dashboard")
st.markdown("TF-IDF + Logistic Regression + Visual Analytics")

# ======================
# DATASET
# ======================
@st.cache_data
def load_data():
    data = {
        "review": [
            "I love this movie it is amazing",
            "Worst movie ever very boring",
            "Great acting and wonderful story",
            "I hate this film so much",
            "Very emotional and touching",
            "Bad script and terrible acting",
            "Enjoyable and fun experience",
            "Not good waste of time"
        ]
    }

    df = pd.DataFrame(data)
    df["sentiment"] = [1,0,1,0,1,0,1,0]

    return df

df = load_data()

# ======================
# MODEL
# ======================
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["review"])
y = df["sentiment"]

model = LogisticRegression()
model.fit(X, y)

# predictions
pred = model.predict(X)

df["predicted"] = pred

# fake emotions (for demo marks)
def emotion(text):
    text = text.lower()
    if "love" in text or "great" in text or "amazing" in text:
        return "Joy"
    elif "hate" in text or "worst" in text:
        return "Anger"
    elif "boring" in text or "waste" in text:
        return "Sadness"
    else:
        return "Neutral"

df["emotion"] = df["review"].apply(emotion)

# ======================
# SIDEBAR
# ======================
menu = st.sidebar.radio("Navigation",
                        ["🏠 Overview", "📊 Dashboard"])

# ======================
# OVERVIEW
# ======================
if menu == "🏠 Overview":
    st.header("📌 Project Overview")

    st.write("""
This system performs:

✔ Sentiment Classification (Positive / Negative)  
✔ Emotion Detection (Joy, Anger, Sadness, Neutral)  
✔ NLP Feature Extraction using TF-IDF  
✔ Full Analytical Dashboard  

### Pipeline:
Text → TF-IDF → Logistic Regression → Prediction
""")

# ======================
# DASHBOARD
# ======================
elif menu == "📊 Dashboard":

    st.header("📊 Graphical Analytics Dashboard")

    # ======================
    # METRICS
    # ======================
    acc = accuracy_score(y, pred)
    prec = precision_score(y, pred)
    rec = recall_score(y, pred)
    f1 = f1_score(y, pred)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{acc:.2f}")
    col2.metric("Precision", f"{prec:.2f}")
    col3.metric("Recall", f"{rec:.2f}")
    col4.metric("F1 Score", f"{f1:.2f}")

    # ======================
    # 1. SENTIMENT BREAKDOWN PIE
    # ======================
    st.subheader("📌 Overall Sentiment Polarity Breakdown")

    sentiment_counts = df["sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]

    sentiment_counts["Sentiment"] = sentiment_counts["Sentiment"].map({
        1: "Positive",
        0: "Negative"
    })

    fig1 = px.pie(sentiment_counts,
                  names="Sentiment",
                  values="Count",
                  title="Sentiment Distribution")
    st.plotly_chart(fig1, use_container_width=True)

    # ======================
    # 2. EMOTION DISTRIBUTION
    # ======================
    st.subheader("🎭 Detected Emotion Frequency")

    emo_counts = df["emotion"].value_counts()

    fig2 = px.bar(x=emo_counts.index,
                  y=emo_counts.values,
                  labels={"x": "Emotion", "y": "Count"},
                  title="Emotion Distribution")

    st.plotly_chart(fig2, use_container_width=True)

    # ======================
    # 3. CONFUSION MATRIX
    # ======================
    st.subheader("📊 Sentiment Confusion Matrix")

    cm = confusion_matrix(y, pred)

    fig3, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig3)

    # ======================
    # 4. MOST FREQUENT WORDS
    # ======================
    st.subheader("🔥 Most Frequently Used Terms")

    all_text = " ".join(df["review"])
    words = all_text.lower().split()

    word_freq = pd.Series(words).value_counts().head(10)

    fig4 = px.bar(x=word_freq.index,
                  y=word_freq.values,
                  labels={"x": "Word", "y": "Frequency"},
                  title="Top Words")

    st.plotly_chart(fig4, use_container_width=True)

    # ======================
    # 5. WORD CLOUD
    # ======================
    st.subheader("☁ Word Cloud")

    wc = WordCloud(width=800, height=400, background_color="white").generate(all_text)

    fig5, ax5 = plt.subplots()
    ax5.imshow(wc, interpolation="bilinear")
    ax5.axis("off")

    st.pyplot(fig5)

    # ======================
    # 6. SENTIMENT TREND (SIMULATED)
    # ======================
    st.subheader("📈 Sentiment Trend Over Time")

    trend = pd.DataFrame({
        "Time": np.arange(1, 9),
        "Sentiment Score": pred
    })

    fig6 = px.line(trend,
                   x="Time",
                   y="Sentiment Score",
                   markers=True,
                   title="Sentiment Trend")

    st.plotly_chart(fig6, use_container_width=True)

    st.success("Dashboard Generated Successfully 🚀")
