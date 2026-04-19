import numpy as np
import joblib
import streamlit as st

@st.cache_resource
def load_artifacts():
    model = joblib.load("models/email_multilabel_model.pkl")
    tfidf = joblib.load("models/tfidf_vectorizer.pkl")
    mlb = joblib.load("models/label_binarizer.pkl")
    return model, tfidf, mlb

def predict_email_labels_with_probs(subject, body, model, tfidf, mlb, threshold=0.35, top_k_fallback=1):
    text = str(subject).strip() + " " + str(body).strip()
    vector = tfidf.transform([text])

    probs = model.predict_proba(vector)[0]
    selected_indices = [i for i, p in enumerate(probs) if p >= threshold]

    if not selected_indices:
        selected_indices = np.argsort(probs)[-top_k_fallback:].tolist()

    predicted_labels = [mlb.classes_[i] for i in selected_indices]
    label_scores = {mlb.classes_[i]: float(probs[i]) for i in range(len(probs))}
    return predicted_labels, label_scores

st.title("Email Classifier")

subject = st.text_input("Subject")
body = st.text_area("Body", height=250)
threshold = st.slider("Threshold", 0.1, 0.9, 0.35, 0.05)

if st.button("Predict"):
    model, tfidf, mlb = load_artifacts()
    labels, scores = predict_email_labels_with_probs(subject, body, model, tfidf, mlb, threshold=threshold)

    st.subheader("Predicted Labels")
    for label in labels:
        st.write(f"- {label}")

    st.subheader("All Label Probabilities")
    for label, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        st.write(f"{label}: {score:.4f}")