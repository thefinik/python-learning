# train.py
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# 1) Tiny training set
texts = [
    "I love this phone",
    "Amazing camera and fast performance",
    "Great value for the price",
    "The delivery was quick",
    "Very helpful customer service",
    "Works perfectly",
    "I hate this product",
    "Terrible quality and bad support",
    "Waste of money",
    "Not good at all",
    "Completely disappointing",
    "Worst experience ever"
]
labels = [1,1,1,1,1,1, 0,0,0,0,0,0]  # 1 = positive, 0 = negative

# 2) Vectorizer + model as a single pipeline
clf = make_pipeline(
    TfidfVectorizer(ngram_range=(1,2)),  # words + bigrams ("not good")
    MultinomialNB()
)

# 3) Train
clf.fit(texts, labels)

# 4) Save trained pipeline (vectorizer + model)
joblib.dump(clf, "model.pkl")
print("Saved model to model.pkl")