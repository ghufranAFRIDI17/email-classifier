import joblib

def load_artifacts():
    model = joblib.load("models/email_multilabel_model.pkl")
    tfidf = joblib.load("models/tfidf_vectorizer.pkl")
    mlb = joblib.load("models/label_binarizer.pkl")
    return model, tfidf, mlb

def predict_email_labels(subject, body, model, tfidf, mlb):
    text = str(subject).strip() + " " + str(body).strip()
    vector = tfidf.transform([text])
    pred = model.predict(vector)
    labels = mlb.inverse_transform(pred)
    return list(labels[0])

if __name__ == "__main__":
    model, tfidf, mlb = load_artifacts()

    subject = "Flight Booking Confirmation"
    body = "Your ticket has been confirmed. Please find travel details attached."

    result = predict_email_labels(subject, body, model, tfidf, mlb)
    print("Predicted labels:", result)