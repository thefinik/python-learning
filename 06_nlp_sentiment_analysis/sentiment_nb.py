from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1️⃣ Training data
texts = [
    "I love this phone",
    "This movie was amazing",
    "Great experience and fast delivery",
    "I hate this product",
    "Terrible quality, not good",
    "Very bad experience"
]
labels = [1, 1, 1, 0, 0, 0]  # 1=positive, 0=negative

# 2️⃣ TF-IDF + bigrams
vec = TfidfVectorizer(ngram_range=(1,2), lowercase=True)
X = vec.fit_transform(texts)

# 3️⃣ Train the model
model = MultinomialNB().fit(X, labels)

# 4️⃣ Test samples
tests = [
    "The phone is not good",
    "Amazing movie and great acting",
    "Terrible product, bad experience",
    "Fast delivery and good quality"
]

# 5️⃣ Predictions
X_test = vec.transform(tests)
preds = model.predict(X_test)

for t, p in zip(tests, preds):
    print(f"{t} -> {'positive' if p==1 else 'negative'}")