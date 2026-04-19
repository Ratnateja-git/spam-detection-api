import pandas as pd
import nltk
import string
import pickle
import re
import numpy as np

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from scipy.sparse import hstack

nltk.download('stopwords')

# ==============================
# 1. LOAD DATA
# ==============================
df = pd.read_csv("data/spam.csv", encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# ==============================
# 2. FEATURE ENGINEERING
# ==============================
def extract_features(text):
    return [
        1 if re.search(r'http|www|\.ly', text) else 0,
        1 if re.search(r'\d+', text) else 0,
        1 if re.search(r'₹|\$|rs', text.lower()) else 0,
        len(text)
    ]

extra_features = np.array([extract_features(t) for t in df['message']])

# ==============================
# 3. TEXT PREPROCESSING
# ==============================
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

df['clean'] = df['message'].apply(preprocess)

# ==============================
# 4. TF-IDF
# ==============================
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_text = tfidf.fit_transform(df['clean'])

# Combine features
X = hstack([X_text, extra_features])
y = df['label']

# ==============================
# 5. TRAIN TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# 6. HANDLE IMBALANCE
# ==============================
weights = compute_class_weight('balanced', classes=np.array([0,1]), y=y_train)
class_weights = {0: weights[0], 1: weights[1]}

# ==============================
# 7. MODEL COMPARISON
# ==============================
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight=class_weights),
    "SVM": LinearSVC(class_weight=class_weights)
}

best_model = None
best_f1 = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n{name}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    f1 = classification_report(y_test, y_pred, output_dict=True)['1']['f1-score']

    if f1 > best_f1:
        best_f1 = f1
        best_model = model

# ==============================
# 8. SAVE MODEL
# ==============================
pickle.dump(best_model, open("model/spam_model.pkl", "wb"))
pickle.dump(tfidf, open("model/vectorizer.pkl", "wb"))

print("\n🏆 Best Model Saved!")
print("🔥 Best Spam F1-score:", best_f1)