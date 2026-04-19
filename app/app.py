from flask import Flask, request, jsonify
import pickle
import re
import string
import numpy as np
import os
import nltk

from nltk.corpus import stopwords
from scipy.sparse import hstack

app = Flask(__name__)

# ==============================
# FIX NLTK (safe for deployment)
# ==============================
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
nltk.data.path.append(nltk_data_path)

try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# ==============================
# LOAD MODEL (FIXED FOR YOUR STRUCTURE)
# ==============================
# Your structure:
# SPAM_DETECTION_PROJECT/
#   ├── app/app.py
#   ├── model/*.pkl

base_dir = os.path.dirname(os.path.dirname(__file__))

model_path = os.path.join(base_dir, "model", "spam_model.pkl")
vectorizer_path = os.path.join(base_dir, "model", "vectorizer.pkl")

model = pickle.load(open(model_path, "rb"))
vectorizer = pickle.load(open(vectorizer_path, "rb"))

# ==============================
# HOME ROUTE (fixes Not Found)
# ==============================
@app.route("/")
def home():
    return "Spam Detection API is running 🚀"

# ==============================
# PREPROCESSING
# ==============================
def preprocess(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

# ==============================
# FEATURE ENGINEERING
# ==============================
def extract_features(text):
    return [
        1 if re.search(r'http|www|\.ly', text) else 0,
        1 if re.search(r'\d+', text) else 0,
        1 if re.search(r'₹|\$|rs', text.lower()) else 0,
        len(text)
    ]

# ==============================
# PREDICTION API
# ==============================
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json() or {}
    text = data.get("message", "")

    processed = preprocess(text)
    text_vec = vectorizer.transform([processed])

    extra = np.array([extract_features(text)])
    final_input = hstack([text_vec, extra])

    pred = model.predict(final_input)[0]

    # Confidence (if available)
    try:
        prob = model.predict_proba(final_input)[0][1]
    except:
        prob = None

    return jsonify({
        "prediction": "spam" if pred == 1 else "ham",
        "confidence": float(prob) if prob is not None else None
    })

# ==============================
# RUN LOCAL ONLY
# ==============================
if __name__ == "__main__":
    app.run(debug=True)