# app.py
import streamlit as st
import joblib

st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬")
st.title("💬 Sentiment Analyzer")
st.caption("TF-IDF + Naive Bayes • demo app")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# Single sentence
text = st.text_area("Enter a sentence:", height=120, placeholder="Type something like: I love this phone")

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        pred = model.predict([text])[0]
        label = "Positive 😊" if pred == 1 else "Negative 😞"

        conf = None
        if hasattr(model, "predict_proba"):
            conf = float(model.predict_proba([text])[0][pred])

        st.subheader(label)
        if conf is not None:
            st.write(f"Confidence: **{conf:.2f}**")

st.divider()

# Batch mode: one sentence per line
st.write("📄 **Batch analysis** (optional)")
file = st.file_uploader("Upload a .txt file (one sentence per line)", type=["txt"])

if file is not None:
    lines = [ln.strip() for ln in file.read().decode("utf-8", errors="ignore").splitlines() if ln.strip()]
    if not lines:
        st.warning("File is empty.")
    else:
        preds = model.predict(lines)
        out = []
        for s, p in zip(lines, preds):
            out.append(f"{s} → {'positive' if p==1 else 'negative'}")
        st.code("\n".join(out))