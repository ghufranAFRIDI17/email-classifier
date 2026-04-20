import base64
import os
from email.utils import parsedate_to_datetime

import joblib
import numpy as np
import requests
import streamlit as st
from streamlit_oauth import OAuth2Component


# =========================
# App config
# =========================
APP_TITLE = "Ghufran-email-classifier"
GMAIL_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"

AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
REVOKE_URL = "https://oauth2.googleapis.com/revoke"
GMAIL_API_BASE = "https://gmail.googleapis.com/gmail/v1"

DEFAULT_REDIRECT_URI = "http://localhost:8501"
# For deployment, set this in Streamlit secrets:
# GOOGLE_REDIRECT_URI = "https://ghufran-email-classifier.streamlit.app"


# =========================
# Basic helpers
# =========================
def get_config(env_name: str, secrets_key: str, default=None):
    try:
        if secrets_key in st.secrets:
            return str(st.secrets[secrets_key])
    except Exception:
        pass

    value = os.getenv(env_name)
    if value is None or str(value).strip() == "":
        return default
    return str(value)


def safe_rerun():
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()


def init_session_state():
    defaults = {
        "gmail_token": None,
        "gmail_emails": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# =========================
# ML model
# =========================
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/email_multilabel_model.pkl")
    tfidf = joblib.load("models/tfidf_vectorizer.pkl")
    mlb = joblib.load("models/label_binarizer.pkl")
    return model, tfidf, mlb


def predict_email_labels_with_probs(subject, body, model, tfidf, mlb, threshold=0.35, top_k_fallback=1):
    text = f"{str(subject).strip()} {str(body).strip()}".strip()
    vector = tfidf.transform([text])

    probs = model.predict_proba(vector)[0]
    selected_indices = [i for i, p in enumerate(probs) if p >= threshold]

    if not selected_indices:
        selected_indices = np.argsort(probs)[-top_k_fallback:].tolist()

    predicted_labels = [mlb.classes_[i] for i in selected_indices]
    label_scores = {mlb.classes_[i]: float(probs[i]) for i in range(len(probs))}
    return predicted_labels, label_scores


# =========================
# Gmail parsing helpers
# =========================
def gmail_api_get(access_token: str, endpoint: str, params=None):
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(
        f"{GMAIL_API_BASE}{endpoint}",
        headers=headers,
        params=params or {},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def gmail_list_messages(access_token: str, max_results: int = 5):
    data = gmail_api_get(
        access_token,
        "/users/me/messages",
        params={"maxResults": max_results, "labelIds": ["INBOX"]},
    )
    return data.get("messages", [])


def gmail_get_message(access_token: str, message_id: str):
    return gmail_api_get(
        access_token,
        f"/users/me/messages/{message_id}",
        params={"format": "full"},
    )


def decode_base64url(data: str) -> str:
    if not data:
        return ""
    padded = data + "=" * (-len(data) % 4)
    decoded = base64.urlsafe_b64decode(padded.encode("utf-8"))
    return decoded.decode("utf-8", errors="ignore")


def extract_headers(payload: dict) -> dict:
    headers = payload.get("headers", [])
    result = {}
    for header in headers:
        name = header.get("name", "")
        value = header.get("value", "")
        if name:
            result[name.lower()] = value
    return result


def extract_plain_text_from_payload(payload: dict) -> str:
    mime_type = payload.get("mimeType", "")
    body = payload.get("body", {}) or {}

    if mime_type == "text/plain" and body.get("data"):
        return decode_base64url(body["data"])

    for part in payload.get("parts", []):
        part_mime = part.get("mimeType", "")
        part_body = part.get("body", {}) or {}

        if part_mime == "text/plain" and part_body.get("data"):
            return decode_base64url(part_body["data"])

        if part.get("parts"):
            nested_text = extract_plain_text_from_payload(part)
            if nested_text.strip():
                return nested_text

    if body.get("data"):
        return decode_base64url(body["data"])

    return ""


def parse_gmail_message(message: dict) -> dict:
    payload = message.get("payload", {})
    headers = extract_headers(payload)

    subject = headers.get("subject", "(No Subject)")
    sender = headers.get("from", "Unknown Sender")
    date_str = headers.get("date", "")
    snippet = message.get("snippet", "")
    body_text = extract_plain_text_from_payload(payload).strip()

    try:
        parsed_date = parsedate_to_datetime(date_str) if date_str else None
    except Exception:
        parsed_date = None

    return {
        "id": message.get("id", ""),
        "thread_id": message.get("threadId", ""),
        "subject": subject,
        "from": sender,
        "date": parsed_date,
        "snippet": snippet,
        "body": body_text if body_text else snippet,
    }


def fetch_recent_emails(access_token: str, max_results: int = 5):
    messages = gmail_list_messages(access_token, max_results=max_results)
    parsed_emails = []

    for item in messages:
        msg_id = item.get("id")
        if not msg_id:
            continue
        raw_message = gmail_get_message(access_token, msg_id)
        parsed_emails.append(parse_gmail_message(raw_message))

    return parsed_emails


# =========================
# UI setup
# =========================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

init_session_state()
model, tfidf, mlb = load_artifacts()

tab1, tab2 = st.tabs(["Manual Prediction", "Gmail Inbox"])


# =========================
# Tab 1: Manual prediction
# =========================
with tab1:
    st.subheader("Manual Email Classification")

    subject = st.text_input("Subject")
    body = st.text_area("Body", height=250)
    threshold = st.slider("Threshold", 0.10, 0.90, 0.35, 0.05, key="manual_threshold")

    if st.button("Predict", use_container_width=True):
        labels, scores = predict_email_labels_with_probs(
            subject=subject,
            body=body,
            model=model,
            tfidf=tfidf,
            mlb=mlb,
            threshold=threshold,
        )

        st.markdown("### Predicted Labels")
        for label in labels:
            st.write(f"- {label}")

        st.markdown("### All Label Probabilities")
        for label, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            st.write(f"{label}: {score:.4f}")


# =========================
# Tab 2: Gmail inbox
# =========================
with tab2:
    st.subheader("Connect Gmail and Classify Inbox")

    client_id = get_config("GOOGLE_CLIENT_ID", "google_client_id")
    client_secret = get_config("GOOGLE_CLIENT_SECRET", "google_client_secret")
    redirect_uri = get_config("GOOGLE_REDIRECT_URI", "google_redirect_uri", DEFAULT_REDIRECT_URI)

    if not client_id or not client_secret:
        st.error(
            "Missing Google OAuth configuration. Add GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET "
            "to Streamlit secrets."
        )
        st.stop()

    oauth2 = OAuth2Component(
        client_id=client_id,
        client_secret=client_secret,
        authorize_endpoint=AUTHORIZE_URL,
        token_endpoint=TOKEN_URL,
        refresh_token_endpoint=TOKEN_URL,
        revoke_token_endpoint=REVOKE_URL,
    )

    token = st.session_state.get("gmail_token")

    if token is None:
        result = oauth2.authorize_button(
            name="Connect Gmail",
            icon="https://ssl.gstatic.com/ui/v1/icons/mail/rfr/gmail.ico",
            redirect_uri=redirect_uri,
            scope=GMAIL_SCOPE,
            key="gmail_oauth_button",
            extras_params={
                "access_type": "offline",
                "prompt": "consent",
                "include_granted_scopes": "true",
            },
            pkce="S256",
            use_container_width=True,
        )

        st.caption(f"Redirect URI in app: {redirect_uri}")

        if result and "token" in result:
            st.session_state["gmail_token"] = result["token"]
            safe_rerun()

        st.info("Connect Gmail first to fetch inbox emails.")

    else:
        st.success("Gmail connected.")

        access_token = token.get("access_token")
        if not access_token:
            st.error("No access token found. Disconnect and connect Gmail again.")
            st.stop()

        col1, col2 = st.columns([3, 1])

        with col1:
            email_limit = st.slider("Number of inbox emails to fetch", 1, 10, 5, 1)
            gmail_threshold = st.slider(
                "Classification threshold",
                0.10,
                0.90,
                0.35,
                0.05,
                key="gmail_threshold",
            )

            if st.button("Fetch and Classify Inbox Emails", use_container_width=True):
                try:
                    emails = fetch_recent_emails(access_token, max_results=email_limit)
                    st.session_state["gmail_emails"] = emails
                except requests.HTTPError as e:
                    try:
                        error_text = e.response.text
                    except Exception:
                        error_text = str(e)
                    st.error(f"Gmail API error: {error_text}")
                except Exception as e:
                    st.error(f"Unexpected Gmail processing error: {e}")

        with col2:
            if st.button("Disconnect Gmail", use_container_width=True):
                try:
                    oauth2.revoke_token(token)
                except Exception:
                    pass
                st.session_state["gmail_token"] = None
                st.session_state["gmail_emails"] = []
                safe_rerun()

        emails = st.session_state.get("gmail_emails", [])

        if emails:
            for idx, email_item in enumerate(emails, start=1):
                labels, scores = predict_email_labels_with_probs(
                    subject=email_item["subject"],
                    body=email_item["body"],
                    model=model,
                    tfidf=tfidf,
                    mlb=mlb,
                    threshold=gmail_threshold,
                )

                with st.expander(f"{idx}. {email_item['subject']}", expanded=(idx == 1)):
                    st.write(f"**From:** {email_item['from']}")
                    if email_item["date"]:
                        st.write(f"**Date:** {email_item['date']}")
                    st.write(f"**Snippet:** {email_item['snippet']}")

                    st.write("**Body Preview:**")
                    body_preview = email_item["body"][:1500] if email_item["body"] else ""
                    st.write(body_preview if body_preview else "No plain-text body found.")

                    st.write("**Predicted Labels:**")
                    for label in labels:
                        st.write(f"- {label}")

                    st.write("**Probabilities:**")
                    for label, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                        st.write(f"{label}: {score:.4f}")