# train.py
# Train TF-IDF + Logistic Regression and save as model.pkl

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Tiny dataset
texts = [
    "I love this phone",
    "Amazing camera and great battery",
    "This is terrible",
    "I hate this device",
    "So good and fast",
    "Very bad experience"
]
labels = [1, 1, 0, 0, 1, 0]  # 1=positive, 0=negative

# Pipeline = vectorizer + classifier
clf = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2)),
    LogisticRegression(max_iter=1000)
)

# Train
clf.fit(texts, labels)

# Save
joblib.dump(clf, "model.pkl")
print("Saved model to model.pkl")