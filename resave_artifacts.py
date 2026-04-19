# save as resave_artifacts.py
import warnings
import joblib
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

model = joblib.load("models/email_multilabel_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")
mlb = joblib.load("models/label_binarizer.pkl")

joblib.dump(model, "models/email_multilabel_model.pkl")
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
joblib.dump(mlb, "models/label_binarizer.pkl")

print("Artifacts re-saved.")