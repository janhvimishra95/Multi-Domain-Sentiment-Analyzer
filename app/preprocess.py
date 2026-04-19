import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# keep important words like "not"
stop_words = set(stopwords.words('english')) - {"not", "no", "never"}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)