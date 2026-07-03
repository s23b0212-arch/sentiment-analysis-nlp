import streamlit as st
import joblib
import re
import nltk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix

# ----------------------------
# SETUP
# ----------------------------
nltk.download('stopwords')
nltk.download('wordnet')

st.set_page_config(
    page_title="IMDb Sentiment AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# LOAD MODEL
# ----------------------------
model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf.pkl")

# ----------------------------
# CLEAN UI STYLE (NETFLIX THEME)
# ----------------------------
st.markdown("""
<style>
.stApp {
    background-color: #0f0f0f;
    color: white;
}

h1, h2, h3 {
    color: white;
}

.stButton button {
    background-color: #e50914;
    color: white;
    border-radius: 8px;
    font-weight: bold;
}

.stTextArea textarea {
    background-color: #1c1c1c;
    color: white;
}

div[data-testid="metric-container"] {
    background-color: #1c1c1c;
    padding: 10px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# NLP CLEANING
# ----------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

# ----------------------------
# SIDEBAR NAVIGATION
# ----------------------------
st.sidebar.title("🎬 IMDb AI System")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "🧠 Predict", "📊 Dashboard", "ℹ️ About"]
)

# ============================
# HOME PAGE
# ============================
if page == "🏠 Home":

    st.title("🎬 Sentiment Analysis AI App")

    st.markdown("""
    ### Welcome 👋

    This system predicts movie review sentiment using:
    - TF-IDF
    - Logistic Regression
    - NLP Text Processing

    ---
    """)

    st.success("Go to Predict page to test the model")

# ============================
# PREDICT PAGE
# ============================
elif page == "🧠 Predict":

    st.title("🧠 Movie Sentiment Prediction")

    st.markdown("### 🎯 Select a Sample Review")

    samples = {
        "😊 Positive": "This movie was absolutely amazing and I loved every moment",
        "😡 Negative": "Worst movie ever. Waste of time and boring plot",
        "😐 Neutral": "The acting was great but the story was average",
        "✍️ Custom": ""
    }

    choice = st.selectbox("Choose review type:", list(samples.keys()))

    if "review" not in st.session_state:
        st.session_state.review = ""

    if choice != "✍️ Custom":
        st.session_state.review = samples[choice]

    review = st.text_area(
        "Enter your review:",
        value=st.session_state.review,
        height=120
    )

    st.session_state.review = review

    # HISTORY
    if "history" not in st.session_state:
        st.session_state.history = []

    if st.button("Analyze 🎯"):

        if review.strip() == "":
            st.warning("Please enter a review")
        else:

            cleaned = clean_text(review)
            vector = tfidf.transform([cleaned])

            prediction = model.predict(vector)
            proba = model.predict_proba(vector)[0]
            confidence = float(np.max(proba))

            if prediction[0] == 1:
                result = "Positive 😊"
                st.success("POSITIVE REVIEW 😊")
            else:
                result = "Negative 😞"
                st.error("NEGATIVE REVIEW 😞")

            st.metric("Confidence", f"{confidence:.2f}")
            st.progress(confidence)

            st.session_state.history.append({
                "review": review,
                "result": result,
                "confidence": round(confidence, 2)
            })

    # HISTORY
    st.markdown("---")
    st.subheader("📌 Prediction History")

    if len(st.session_state.history) == 0:
        st.info("No predictions yet")
    else:
        for i, item in enumerate(reversed(st.session_state.history)):
            st.markdown(f"""
            **{i+1}. Review:** {item['review']}  
            **Result:** {item['result']}  
            **Confidence:** {item['confidence']}  
            ---
            """)

# ============================
# DASHBOARD PAGE
# ============================
elif page == "📊 Dashboard":

    st.title("📊 Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", "88%")
    col2.metric("Precision", "0.88")
    col3.metric("Recall", "0.88")
    col4.metric("F1 Score", "0.88")

    st.markdown("---")

    st.markdown("### Dataset Distribution")

    fig, ax = plt.subplots()
    ax.bar(["Positive", "Negative"], [25000, 25000], color=["green", "red"])
    st.pyplot(fig)

    st.markdown("### Confusion Matrix (Demo)")

    y_true = [0,1,0,1,1,0,1,0,1,1]
    y_pred = [0,1,0,1,0,0,1,0,1,1]

    cm = confusion_matrix(y_true, y_pred)

    fig2, ax2 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2)
    st.pyplot(fig2)

# ============================
# ABOUT PAGE
# ============================
elif page == "ℹ️ About":

    st.title("ℹ️ About This Project")

    st.markdown("""
    ### 🎯 Objective
    Build NLP sentiment analysis system

    ### 🧠 Model
    TF-IDF + Logistic Regression

    ### 📊 Dataset
    IMDb Reviews (Kaggle)

    ### ⚙️ Features
    - Sentiment Prediction
    - Confidence Score
    - Clean UI
    - Dashboard Visualization
    """)
