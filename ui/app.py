import streamlit as st
import pandas as pd
import pickle

# Page config
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="💬",
    layout="wide"
)

# Load model
model = pickle.load(open("models/sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

# Prediction function
def predict(text):
    vec = vectorizer.transform([text])
    result = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]

    confidence = max(prob) * 100
    label = "Positive 😊" if result == 1 else "Negative 😡"

    return label, round(confidence, 2)

# ---------------- UI ---------------- #

# Header
st.markdown("""
    <h1 style='text-align: center;'>💬 Multi-Domain Sentiment Analyzer</h1>
    <p style='text-align: center;'>Analyze reviews from products, movies, hospitals, apps & more</p>
""", unsafe_allow_html=True)

st.divider()

# Layout
col1, col2 = st.columns([2, 1])

# LEFT SIDE (INPUT)
with col1:
    st.subheader("📝 Enter Your Review")

    text = st.text_area("Type here...", height=150)

    if st.button("🚀 Analyze Sentiment"):
        if text.strip() == "":
            st.warning("Please enter some text")
        else:
            label, confidence = predict(text)

            st.success(f"Prediction: {label}")
            st.metric("Confidence", f"{confidence}%")

# RIGHT SIDE (INFO PANEL)
with col2:
    st.subheader("📌 Features")

    st.markdown("""
    - ✅ Multi-domain analysis  
    - 📊 Real-time prediction  
    - 📂 CSV bulk analysis  
    - 🤖 ML-powered model  
    """)

# Divider
st.divider()

# ---------------- CSV Upload ---------------- #

st.subheader("📂 Bulk Analysis (Upload CSV)")

file = st.file_uploader("Upload CSV with 'text' column")

if file:
    df = pd.read_csv(file)

    if "text" not in df.columns:
        st.error("CSV must contain 'text' column")
    else:
        df['prediction'] = df['text'].apply(lambda x: predict(x)[0])

        st.write("### 📄 Results")
        st.dataframe(df)

        # Charts
        st.subheader("📊 Sentiment Distribution")

        counts = df['prediction'].value_counts()
        st.bar_chart(counts)

# ---------------- Model Performance ---------------- #

st.divider()
st.subheader("📈 Model Performance")

try:
    results = pd.read_csv("models/results.csv")
    st.bar_chart(results.set_index("Model"))
    st.dataframe(results)
except:
    st.info("Train model to see performance")

# Footer
st.markdown("""
---
<p style='text-align: center;'>🚀 Built with Streamlit | Multi-Domain Sentiment Analyzer</p>
""", unsafe_allow_html=True)
