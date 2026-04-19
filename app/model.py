import pickle
from app.preprocess import clean_text

model = pickle.load(open("models/sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

def predict(text):
    text = clean_text(text)
    vec = vectorizer.transform([text])
    result = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]

    confidence = max(prob) * 100

    label = "Positive 😊" if result == 1 else "Negative 😡"

    return label, round(confidence, 2)