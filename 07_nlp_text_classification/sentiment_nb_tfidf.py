from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1) Data
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
    "Worst experience ever",
]
labels = [1,1,1,1,1,1, 0,0,0,0,0,0]   # 1 = positive, 0 = negative

# 2) Train/Test split (stratified to keep class balance)
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.25, random_state=42, stratify=labels
)

# 3) Vectorizer + Model
vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # unigrams + bigrams
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 4) Evaluation
preds = model.predict(X_test_vec)
print("Accuracy:", round(accuracy_score(y_test, preds), 3))
print("\nClassification report:\n", classification_report(y_test, preds, target_names=["negative","positive"]))

cm = confusion_matrix(y_test, preds)
print("Confusion matrix:\n", cm)

# 5) Inference on new sentences
demo = [
    "The service was great",
    "Terrible food, I hate it",
    "Amazing movie experience",
    "This phone is bad",
    "Fast delivery and helpful support",
]
demo_vec = vectorizer.transform(demo)
demo_preds = model.predict(demo_vec)

print("\n— Inference —")
for t, p in zip(demo, demo_preds):
    print(f"{t:<35} -> {'positive' if p==1 else 'negative'}")