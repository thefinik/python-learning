# NLP Text Classification (TF-IDF + Naive Bayes)

A minimal sentiment classifier that turns text into numeric features with **TF-IDF**, trains a **Multinomial Naive Bayes** model, and evaluates it with **Accuracy, Precision, Recall, F1**. Includes a confusion matrix and a simple inference demo.

## What this demonstrates
- Train/test split (stratified)
- Vectorization with `TfidfVectorizer` (unigrams + bigrams)
- Training `MultinomialNB`
- Evaluation (classification report + confusion matrix)
- Inference on custom phrases

## How to run (locally)
```bash
# from the repo root
cd 07_nlp_text_classification
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python sentiment_nb_tfidf.py