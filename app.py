import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="Movie Sentiment AI", layout="wide")

st.title("🎬 Smart Movie Sentiment Analytics Dashboard")
st.markdown("🔥 NLP + Emotion Detection + Interactive Analytics")

# ======================
# SAMPLE DATASET
# ======================
@st.cache_data
def load_data():
    data = {
        "review": [
            "I love this movie, it is amazing",
            "This movie is terrible and boring",
            "What a fantastic and emotional film",
            "Worst movie ever, I hate it",
            "Absolutely wonderful storyline",
            "Bad acting and poor script",
            "Very enjoyable and fun experience",
            "I am angry wasting time on this"
        ]
    }

    df = pd.DataFrame(data)

    # fake labels for training
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

# ======================
# EMOTION SIMULATION
# ======================
def get_emotion(text):
    text = text.lower()
    if "love" in text or "amazing" in text:
        return "Joy 😊"
    elif "hate" in text or "worst" in text:
        return "Anger 😡"
    elif "boring" in text:
        return "Sad 😢"
    else:
        return "Neutral 😐"

# ======================
# SIDEBAR
# ======================
menu = st.sidebar.radio("Navigation",
                        ["🏠 Overview", "🎯 Predict", "📊 Dashboard", "📂 Upload CSV"])

# ======================
# OVERVIEW
# ======================
if menu == "🏠 Overview":
    st.header("📌 Project Overview")

    st.write("""
This system performs:

✔ Sentiment Analysis (Positive / Negative / Neutral)  
✔ Emotion Detection (Joy, Anger, Sadness)  
✔ Movie Review Analytics Dashboard  
✔ Upload your own dataset  
✔ Simulated movie link analysis  

### NLP Pipeline:
Text → TF-IDF → Logistic Regression → Prediction
""")

# ======================
# PREDICT
# ======================
elif menu == "🎯 Predict":
    st.header("💬 Movie Review Analyzer")

    option = st.radio("Input type:", ["Type Review", "Paste Movie Link (simulated)"])

    if option == "Type Review":
        text = st.text_area("Enter review")
    else:
        text = st.text_input("Paste movie URL")

    if st.button("Analyze"):

        if text.strip() == "":
            st.warning("Enter something!")
        else:
            vec = vectorizer.transform([text])
            pred = model.predict(vec)[0]
            prob = model.predict_proba(vec)[0]

            sentiment = "Positive 😊" if pred == 1 else "Negative 😞"
            emotion = get_emotion(text)

            st.success(f"Sentiment: {sentiment}")
            st.info(f"Emotion: {emotion}")

            st.progress(int(max(prob)*100))

            # emoji result box
            st.markdown(f"### Result: {sentiment} | {emotion}")

# ======================
# DASHBOARD
# ======================
elif menu == "📊 Dashboard":

    st.header("📊 Analytics Dashboard")

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

    # animated bar chart (plotly WOW)
    fig = px.bar(
        x=["Accuracy", "Precision", "Recall", "F1"],
        y=[acc, prec, rec, f1],
        color=["Accuracy", "Precision", "Recall", "F1"],
        title="Model Performance Metrics"
    )
    st.plotly_chart(fig, use_container_width=True)

    # confusion matrix
    cm = confusion_matrix(y, y_pred)

    fig2, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    st.pyplot(fig2)

    # word cloud
    st.subheader("☁ Word Cloud")

    text = " ".join(df["review"])
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)

    fig3, ax3 = plt.subplots()
    ax3.imshow(wc, interpolation="bilinear")
    ax3.axis("off")

    st.pyplot(fig3)

# ======================
# CSV UPLOAD FEATURE
# ======================
elif menu == "📂 Upload CSV":

    st.header("📂 Upload Your Movie Reviews")

    file = st.file_uploader("Upload CSV file", type=["csv"])

    if file is not None:
        user_df = pd.read_csv(file)

        st.write("Preview:")
        st.dataframe(user_df.head())

        if "review" in user_df.columns:

            vec = vectorizer.transform(user_df["review"])
            preds = model.predict(vec)

            user_df["sentiment"] = preds

            st.success("Prediction Done!")

            st.dataframe(user_df)

            st.bar_chart(user_df["sentiment"].value_counts())

        else:
            st.error("CSV must contain 'review' column")
