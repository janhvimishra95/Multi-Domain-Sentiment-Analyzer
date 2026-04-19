from fastapi import FastAPI
from app.model import predict

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Sentiment API is running 🚀"}

@app.post("/predict")
def get_sentiment(text: str):
    label, confidence = predict(text)
    return {
        "sentiment": label,
        "confidence": confidence
    }