import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import plotly.express as px

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="NLP Sentiment Intelligence",
    layout="wide"
)

# ======================
# HEADER
# ======================
st.markdown("""
<h1 style='text-align:center; color:#4F8BF9;'>🎬 NLP Sentiment Intelligence Dashboard</h1>
<p style='text-align:center; color:gray;'>TF-IDF + Logistic Regression + Emotion Analytics</p>
""", unsafe_allow_html=True)

# ======================
# SMALL TRAINING DATA
# ======================
data = {
    "review": [
        "This movie is amazing and I love it",
        "Worst movie ever very boring",
        "I really enjoyed the story",
        "Terrible acting and bad plot",
        "Fantastic film with great actors",
        "I hate this movie so much",
        "Beautiful cinematography and great music",
        "Waste of time",
        "It was okay not too bad",
        "Not good not bad just average"
    ],
    "sentiment": [1,0,1,0,1,0,1,0,1,1]
}

df = pd.DataFrame(data)

# ======================
# MODEL
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
# SESSION HISTORY
# ======================
if "history" not in st.session_state:
    st.session_state.history = []

# ======================
# EMOTION FUNCTION
# ======================
def get_emotion(text):
    text = text.lower()
    if any(w in text for w in ["love", "amazing", "great", "good", "fantastic"]):
        return "Joy 😊"
    elif any(w in text for w in ["hate", "worst", "boring", "terrible"]):
        return "Anger 😡"
    elif any(w in text for w in ["sad", "bad", "waste"]):
        return "Sadness 😢"
    else:
        return "Neutral 😐"

# ======================
# MENU
# ======================
menu = st.sidebar.radio(
    "Navigation",
    ["🏠 Overview", "🎯 Predict", "📊 Dashboard", "📜 History"]
)

# ======================
# OVERVIEW
# ======================
if menu == "🏠 Overview":
    st.subheader("📌 Project Overview")
    st.write("""
This system performs:
✔ Sentiment Analysis  
✔ Emotion Detection  
✔ Live Dashboard Updates  
✔ History Tracking  
""")

# ======================
# PREDICT (PASTE BOX - NO CSV)
# ======================
elif menu == "🎯 Predict":

    st.subheader("💬 Paste Multiple Reviews (Demo Mode)")

    user_input = st.text_area(
        "Enter reviews (one per line)",
        height=200,
        placeholder="This movie is amazing\nWorst movie ever\nI love it"
    )

    if st.button("Analyze All Reviews"):

        reviews = [r.strip() for r in user_input.split("\n") if r.strip()]

        results = []

        for r in reviews:
            vec = vectorizer.transform([r])
            pred = model.predict(vec)[0]
            prob = model.predict_proba(vec)[0]

            sentiment = "Positive 😊" if pred == 1 else "Negative 😡"
            emotion = get_emotion(r)

            results.append([r, sentiment, emotion, max(prob)])

            st.session_state.history.append({
                "review": r,
                "sentiment": sentiment,
                "emotion": emotion
            })

        df_result = pd.DataFrame(results, columns=["Review", "Sentiment", "Emotion", "Confidence"])

        st.success("Analysis Completed!")
        st.dataframe(df_result)

# ======================
# DASHBOARD
# ======================
elif menu == "📊 Dashboard":

    st.subheader("📊 Live Sentiment Dashboard")

    if len(st.session_state.history) == 0:
        st.warning("No data yet. Go to Predict page.")
    else:
        dfh = pd.DataFrame(st.session_state.history)

        col1, col2 = st.columns(2)

        # SENTIMENT PIE
        with col1:
            fig = px.pie(dfh, names="sentiment", title="Sentiment Breakdown", hole=0.5)
            st.plotly_chart(fig, use_container_width=True)

        # EMOTION BAR
        with col2:
            fig2 = px.bar(dfh["emotion"].value_counts(),
                          title="Emotion Frequency")
            st.plotly_chart(fig2, use_container_width=True)

        # TREND
        st.subheader("Sentiment Trend")
        dfh["index"] = range(len(dfh))

        fig3 = px.line(
            dfh,
            x="index",
            y=dfh["sentiment"].map({"Positive 😊": 1, "Negative 😡": -1}),
            markers=True
        )
        st.plotly_chart(fig3, use_container_width=True)

# ======================
# HISTORY
# ======================
elif menu == "📜 History":

    st.subheader("📜 Review History")

    if len(st.session_state.history) == 0:
        st.info("No history yet")
    else:
        hist = pd.DataFrame(st.session_state.history)
        st.dataframe(hist)

        st.download_button(
            "⬇ Download History",
            hist.to_csv(index=False).encode("utf-8"),
            "history.csv",
            "text/csv"
        )
