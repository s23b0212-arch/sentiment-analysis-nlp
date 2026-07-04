import streamlit as st
import joblib
import re
import nltk
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# =========================
# SETUP
# =========================
st.set_page_config(page_title="IMDb Sentiment AI", layout="wide")

nltk.download('stopwords')
nltk.download('wordnet')

# =========================
# LOAD MODEL
# =========================
model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf.pkl")

# =========================
# LOAD DATASET (FOR REAL METRICS)
# =========================
df = pd.read_csv("IMDB Dataset.csv")  # make sure file is uploaded in repo

df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# =========================
# TEXT CLEANING
# =========================
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

df['clean_review'] = df['review'].apply(clean_text)

# =========================
# FEATURES
# =========================
X = tfidf.transform(df['clean_review'])
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

y_pred = model.predict(X_test)

report = classification_report(y_test, y_pred, output_dict=True)
cm = confusion_matrix(y_test, y_pred)

# =========================
# AI TYPE EFFECT
# =========================
def type_writer(text):
    box = st.empty()
    out = ""
    for c in text:
        out += c
        box.markdown(f"### 🤖 {out}")
        time.sleep(0.01)

# =========================
# NLP CLEANING (UI USE)
# =========================
def clean_text_ui(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🎬 IMDb Sentiment AI")
page = st.sidebar.radio("Navigation", ["🏠 Home", "🧠 Predict", "📊 Dashboard", "ℹ️ About"])

# =========================
# HOME
# =========================
if page == "🏠 Home":

    st.title("🎬 NLP Sentiment Analysis System")

    st.markdown("""
### 📌 Project Overview
This system classifies movie reviews using NLP.

✔ Dataset: IMDb Reviews  
✔ Model: TF-IDF + Logistic Regression  
✔ Output: Positive / Negative  

---

### 🎯 Objective
Automate sentiment classification using machine learning.
""")

# =========================
# PREDICT PAGE
# =========================
elif page == "🧠 Predict":

    st.title("🧠 Sentiment Prediction")

    samples = [
        "Type your own review",
        "This movie was amazing and fantastic",
        "Worst movie ever, I hated it",
        "Acting was good but story was average"
    ]

    choice = st.selectbox("Choose sample review:", samples)

    if choice == "Type your own review":
        review = st.text_area("Enter your review:")
    else:
        review = choice
        st.info(f"Selected: {choice}")

    if "history" not in st.session_state:
        st.session_state.history = []

    if st.button("Analyze 🎯"):

        if review.strip() == "":
            st.warning("Please enter review")
        else:

            cleaned = clean_text_ui(review)
            vector = tfidf.transform([cleaned])

            pred = model.predict(vector)
            prob = model.predict_proba(vector)[0]

            confidence = float(np.max(prob))

            st.markdown("### ⚡ Live Sentiment Meter")
            st.progress(float(max(prob)))

            if pred[0] == 1:
                type_writer("Positive Sentiment Detected 😊")
                result = "Positive"
            else:
                type_writer("Negative Sentiment Detected 😞")
                result = "Negative"

            st.metric("Confidence", f"{confidence:.2%}")

            st.session_state.history.append({
                "review": review,
                "result": result,
                "confidence": round(confidence, 2)
            })

    st.markdown("---")
    st.subheader("📌 History")

    for i, item in enumerate(reversed(st.session_state.history)):
        st.write(f"{i+1}. {item['review']} → {item['result']} ({item['confidence']})")

# =========================
# DASHBOARD (REAL METRICS)
# =========================
elif page == "📊 Dashboard":

    st.title("📊 Model Evaluation Dashboard")

    acc = report["accuracy"]
    precision = report["1"]["precision"]
    recall = report["1"]["recall"]
    f1 = report["1"]["f1-score"]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{acc:.2f}")
    col2.metric("Precision", f"{precision:.2f}")
    col3.metric("Recall", f"{recall:.2f}")
    col4.metric("F1 Score", f"{f1:.2f}")

    st.markdown("---")

    st.subheader("📊 Confusion Matrix (REAL)")

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)

    st.pyplot(fig)

    st.markdown("""
### 📌 Analysis
Model performs well for binary sentiment classification using TF-IDF + Logistic Regression.
""")

# =========================
# ABOUT
# =========================
elif page == "ℹ️ About":

    st.title("ℹ️ Project Details")

    st.markdown("""
### 🎓 Title
Sentiment Analysis System using NLP

### ⚙️ Methodology
- Data Cleaning
- TF-IDF
- Logistic Regression
- Evaluation using classification report

### 📊 Result
Achieved ~88% accuracy on IMDb dataset.

### 🚀 Future Work
- Use BERT / Deep Learning
- Improve sarcasm detection
""")
