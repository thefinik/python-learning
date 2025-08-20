import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# --- 1) Load data
HERE = Path(__file__).resolve().parent
data = pd.read_csv(HERE / "data.csv")

# Features (X) and target (y)
X = data[["hours", "sleep"]]
y = data["passed"]

# --- 2) Train/Test split (stratify keeps class ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# --- 3) Define models
logreg = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000, random_state=42))
])

tree = DecisionTreeClassifier(max_depth=3, random_state=42)

# Pick the best k for KNN via cross-validation (safe upper bound for k)
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
n_train_per_fold = len(X) * (cv.n_splits - 1) // cv.n_splits
max_k = min(10, max(1, n_train_per_fold))  # cap at 10 for neatness

best_k, best_cv = 1, -1.0
for k in range(1, max_k + 1):
    knn_cv = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=k))
    ])
    cv_mean = cross_val_score(knn_cv, X, y, cv=cv, scoring="accuracy").mean()
    if cv_mean > best_cv:
        best_cv, best_k = cv_mean, k

knn = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", KNeighborsClassifier(n_neighbors=best_k))
])

# --- 4) Train and evaluate on test set
models = {"LogReg": logreg, "Tree(max_depth=3)": tree, f"KNN(k={best_k})": knn}
test_acc = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    test_acc[name] = accuracy_score(y_test, preds)

print("Test accuracy per model:")
for name, acc in test_acc.items():
    print(f"  {name}: {acc:.3f}")
print(f"(KNN: best_k={best_k} by CV, mean CV acc={best_cv:.3f})")

# --- 5) Bar plot of results
plt.figure(figsize=(6, 4))
names = list(test_acc.keys())
vals = [test_acc[n] for n in names]
plt.bar(names, vals)
plt.ylim(0, 1.0)
plt.ylabel("Accuracy")
plt.title("Model comparison on test set")
for i, v in enumerate(vals):
    plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
plt.tight_layout()
plt.savefig(HERE / "results.png", dpi=150)

# --- 6) Optional: visualize the tree
plt.figure(figsize=(6, 4))
plot_tree(tree, feature_names=["hours", "sleep"], class_names=["fail", "pass"], filled=True)
plt.tight_layout()
plt.savefig(HERE / "tree.png", dpi=150)

print("Saved figures: results.png, tree.png")