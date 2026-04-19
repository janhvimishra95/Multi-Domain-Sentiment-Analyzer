import pandas as pd
import pickle
import os
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.preprocess import clean_text

# Load data
data = pd.read_csv("data/raw/data.csv")

# Preprocess
data['text'] = data['text'].apply(clean_text)
data['label'] = data['label'].map({"positive": 1, "negative": 0})

# Split (IMPORTANT: shuffle)
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42, shuffle=True
)

# Vectorize
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Naive Bayes": MultinomialNB(),
    "SVM": LinearSVC()
}

best_model = None
best_accuracy = 0
results = {}

print("\n📊 Model Performance:\n")

for name, model in models.items():
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)

    acc = accuracy_score(y_test, preds)
    results[name] = round(acc * 100, 2)

    print(f"{name}: {round(acc*100, 2)}%")

    print(classification_report(y_test, preds))
    print("-" * 40)

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

# Save best model
os.makedirs("models", exist_ok=True)

pickle.dump(best_model, open("models/sentiment_model.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))

# Save results for UI
pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"]).to_csv("models/results.csv", index=False)

print(f"\n✅ Best Model Accuracy: {round(best_accuracy*100,2)}%") 