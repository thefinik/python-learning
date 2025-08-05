# Kaggle Mini Project — Student Performance

**Goal:** predict if a student will pass (`passed` ∈ {0,1}) using:
- `hours` — study hours
- `sleep` — sleep hours

## Steps
1. Load data (`data/data.csv`) and inspect.
2. Split into train/test (stratified).
3. Train three models:
   - Logistic Regression (with StandardScaler)
   - Decision Tree (`max_depth=3`)
   - KNN — `k` chosen via 4-fold cross-validation
4. Evaluate on test set (Accuracy).
5. Visualize results (`results.png`) and the tree (`tree.png`).

## Results
Accuracy (LogReg): 0.75
Accuracy (DecisionTree): 0.75
Accuracy (KNN): 0.83

## How to run
```bash
pip install -r requirements.txt
python main.py
