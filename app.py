import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="Smart Movie Sentiment Dashboard", layout="wide")

st.title("🎬 Smart Movie Sentiment Analytics Dashboard")
st.markdown("NLP System using TF-IDF + Logistic Regression + Emotion Analytics")

# ======================
# LOAD MODEL (FIXED)
# ======================
model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf.pkl")

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
    elif any(w in text for w in ["bad", "worst", "boring", "hate"]):
        return "Anger 😡"
    elif any(w in text for w in ["sad", "disappointed", "cry"]):
        return "Sadness 😢"
    elif any(w in text for w in ["wow", "surprise", "unexpected"]):
        return "Surprise 😲"
    else:
        return "Neutral 😐"

# ======================
# SIDEBAR MENU
# ======================
menu = st.sidebar.radio(
    "Navigation",
    ["🏠 Overview", "🎯 Predict", "📊 Dashboard", "📜 History"]
)

# ======================
# OVERVIEW
# ======================
if menu == "🏠 Overview":
    st.header("📌 Project Overview")
    st.write("""
This system performs **Movie Sentiment Analysis using NLP**.

### Features:
✔ Sentiment Prediction (Positive / Negative)  
✔ Emotion Detection  
✔ Confidence Score  
✔ Sample Review Dropdown  
✔ Analytics Dashboard  
✔ History Tracking  
✔ Download Results  

### NLP Pipeline:
Text → TF-IDF → Logistic Regression → Prediction
""")

# ======================
# PREDICTION PAGE
# ======================
elif menu == "🎯 Predict":
    st.header("💬 Sentiment Prediction Engine")

    sample_reviews = [
        "This movie is amazing!",
        "Worst movie ever",
        "It was okay not bad",
        "I love the acting",
        "Very boring film",
        "Fantastic storyline and great acting",
        "I hate this movie so much"
    ]

    input_type = st.radio("Choose input method:", ["Type Review", "Use Sample Review"])

    if input_type == "Type Review":
        user_input = st.text_area("Enter your movie review:")
    else:
        user_input = st.selectbox("Select a sample review:", sample_reviews)

    if st.button("Analyze Sentiment 🚀"):

        if user_input.strip() == "":
            st.warning("Please enter a review")
        else:
            vec = tfidf.transform([user_input])
            pred = model.predict(vec)[0]
            prob = model.predict_proba(vec)[0]

            confidence = int(max(prob) * 100)

            # SENTIMENT RESULT
            if pred == 1:
                st.success("😊 Positive Review")
            else:
                st.error("😡 Negative Review")

            st.progress(confidence)
            st.write(f"Confidence Score: {confidence}%")

            # EMOTION
            emotion = get_emotion(user_input)
            st.info(f"Detected Emotion: {emotion}")

            # SAVE HISTORY
            st.session_state.history.append({
                "review": user_input,
                "sentiment": "Positive" if pred == 1 else "Negative",
                "emotion": emotion,
                "confidence": confidence
            })

# ======================
# DASHBOARD
# ======================
elif menu == "📊 Dashboard":
    st.header("📊 Graphical Analytics Dashboard")

    if len(st.session_state.history) == 0:
        st.info("No data yet. Go to Predict page first.")

    else:
        df = pd.DataFrame(st.session_state.history)

        # ======================
        # 1. SENTIMENT BREAKDOWN
        # ======================
        st.subheader("📊 Overall Sentiment Polarity Breakdown")

        fig, ax = plt.subplots()
        df["sentiment"].value_counts().plot(
            kind="pie",
            autopct="%1.1f%%",
            ax=ax
        )
        ax.set_ylabel("")
        st.pyplot(fig)

        # ======================
        # 2. EMOTION DISTRIBUTION
        # ======================
        st.subheader("😄 Emotion Frequency Distribution")

        fig2, ax2 = plt.subplots()
        sns.countplot(x=df["emotion"], ax=ax2)
        plt.xticks(rotation=45)
        st.pyplot(fig2)

        # ======================
        # 3. SENTIMENT VS EMOTION MATRIX
        # ======================
        st.subheader("📊 Sentiment & Emotion Correlation Matrix")

        pivot = pd.crosstab(df["emotion"], df["sentiment"])

        fig3, ax3 = plt.subplots()
        sns.heatmap(pivot, annot=True, cmap="Blues", ax=ax3)
        st.pyplot(fig3)

        # ======================
        # 4. MOST FREQUENT WORDS
        # ======================
        st.subheader("☁ Most Frequently Used Terms")

        from collections import Counter
        import re

        all_words = " ".join(df["review"]).lower()
        words = re.findall(r'\b[a-z]+\b', all_words)

        stopwords = set(["the","is","and","a","this","it","to","was","i","very","movie"])
        words = [w for w in words if w not in stopwords]

        word_counts = Counter(words).most_common(10)

        word_df = pd.DataFrame(word_counts, columns=["Word", "Count"])

        fig4, ax4 = plt.subplots()
        sns.barplot(x="Count", y="Word", data=word_df, ax=ax4)
        st.pyplot(fig4)

        # ======================
        # 5. SENTIMENT TREND
        # ======================
        st.subheader("📈 Sentiment Trend Over Time")

        df["index"] = range(len(df))

        fig5, ax5 = plt.subplots()
        sns.lineplot(x="index", y=df["sentiment"].map({
            "Positive": 1,
            "Negative": -1,
            "Neutral": 0
        }), data=df, ax=ax5)

        ax5.set_ylabel("Sentiment Score")
        st.pyplot(fig5)

        # ======================
        # DOWNLOAD
        # ======================
        st.download_button(
            "📥 Download Full Analytics Data",
            df.to_csv(index=False).encode("utf-8"),
            "analytics.csv",
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
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df)

        st.download_button(
            "Download History",
            df.to_csv(index=False).encode("utf-8"),
            "history.csv",
            "text/csv"
        )
