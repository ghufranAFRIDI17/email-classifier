import base64
import os
import secrets
import time
from email.utils import parsedate_to_datetime
from urllib.parse import urlencode

import joblib
import numpy as np
import requests
import streamlit as st


# =========================
# Config
# =========================
GMAIL_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GMAIL_API_BASE = "https://gmail.googleapis.com/gmail/v1"


# =========================
# Helpers: config / query params
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


def get_query_params() -> dict:
    try:
        return st.query_params
    except Exception:
        try:
            return st.experimental_get_query_params()
        except Exception:
            return {}


def first_param(params: dict, key: str):
    value = params.get(key)
    if isinstance(value, list):
        return value[0] if value else None
    return value


def clear_query_params():
    try:
        st.query_params.clear()
    except Exception:
        try:
            st.experimental_set_query_params()
        except Exception:
            pass


def safe_rerun():
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()


# =========================
# Helpers: token/session state
# =========================
def init_session_state():
    defaults = {
        "oauth_state": secrets.token_urlsafe(24),
        "gmail_tokens": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_tokens():
    st.session_state["gmail_tokens"] = None


# =========================
# OAuth
# =========================
def build_google_auth_url(client_id: str, redirect_uri: str, state: str) -> str:
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": GMAIL_SCOPE,
        "access_type": "offline",
        "prompt": "consent",
        "include_granted_scopes": "true",
        "state": state,
    }
    return f"{GOOGLE_AUTH_URL}?{urlencode(params)}"


def exchange_code_for_tokens(code: str, client_id: str, client_secret: str, redirect_uri: str) -> dict:
    payload = {
        "code": code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
    }
    response = requests.post(GOOGLE_TOKEN_URL, data=payload, timeout=30)
    response.raise_for_status()
    data = response.json()

    if "expires_in" in data:
        data["expires_at"] = time.time() + float(data["expires_in"])
    return data


def refresh_access_token(refresh_token: str, client_id: str, client_secret: str) -> dict:
    payload = {
        "refresh_token": refresh_token,
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "refresh_token",
    }
    response = requests.post(GOOGLE_TOKEN_URL, data=payload, timeout=30)
    response.raise_for_status()
    data = response.json()

    if "expires_in" in data:
        data["expires_at"] = time.time() + float(data["expires_in"])
    return data


def get_valid_access_token(tokens: dict, client_id: str, client_secret: str):
    if not tokens:
        return None

    access_token = tokens.get("access_token")
    expires_at = tokens.get("expires_at")
    refresh_token = tokens.get("refresh_token")

    try:
        if access_token and expires_at and time.time() < float(expires_at) - 60:
            return str(access_token)
    except Exception:
        pass

    if refresh_token:
        refreshed = refresh_access_token(str(refresh_token), client_id, client_secret)
        tokens["access_token"] = refreshed.get("access_token")
        tokens["expires_at"] = refreshed.get("expires_at", time.time() + 3500)

        if refreshed.get("refresh_token"):
            tokens["refresh_token"] = refreshed["refresh_token"]

        st.session_state["gmail_tokens"] = tokens
        return tokens["access_token"]

    return None


# =========================
# Gmail API
# =========================
def gmail_api_get(access_token: str, endpoint: str, params: dict | None = None) -> dict:
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(f"{GMAIL_API_BASE}{endpoint}", headers=headers, params=params or {}, timeout=30)
    response.raise_for_status()
    return response.json()


def gmail_list_messages(access_token: str, max_results: int = 5) -> list:
    data = gmail_api_get(
        access_token,
        "/users/me/messages",
        params={"maxResults": max_results, "labelIds": "INBOX"},
    )
    return data.get("messages", [])


def gmail_get_message(access_token: str, message_id: str) -> dict:
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

    parts = payload.get("parts", [])
    for part in parts:
        part_mime = part.get("mimeType", "")
        part_body = part.get("body", {}) or {}

        if part_mime == "text/plain" and part_body.get("data"):
            return decode_base64url(part_body["data"])

        if part.get("parts"):
            nested = extract_plain_text_from_payload(part)
            if nested.strip():
                return nested

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
        "threadId": message.get("threadId", ""),
        "subject": subject,
        "from": sender,
        "date": parsed_date,
        "snippet": snippet,
        "body": body_text if body_text else snippet,
    }


def fetch_recent_emails(access_token: str, max_results: int = 5) -> list:
    messages = gmail_list_messages(access_token, max_results=max_results)
    parsed_emails = []

    for item in messages:
        msg_id = item.get("id")
        if not msg_id:
            continue
        raw_msg = gmail_get_message(access_token, msg_id)
        parsed_emails.append(parse_gmail_message(raw_msg))

    return parsed_emails


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
# UI: app
# =========================
st.set_page_config(page_title="Ghufran-email-classifier", layout="wide")
st.title("Ghufran-email-classifier")

model, tfidf, mlb = load_artifacts()

tab1, tab2 = st.tabs(["Manual Prediction", "Gmail Inbox"])

with tab1:
    st.subheader("Manual Email Classification")

    subject = st.text_input("Subject")
    body = st.text_area("Body", height=250)
    threshold = st.slider("Threshold", 0.10, 0.90, 0.35, 0.05, key="manual_threshold")

    if st.button("Predict", use_container_width=True):
        labels, scores = predict_email_labels_with_probs(
            subject, body, model, tfidf, mlb, threshold=threshold
        )

        st.markdown("### Predicted Labels")
        for label in labels:
            st.write(f"- {label}")

        st.markdown("### All Label Probabilities")
        for label, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            st.write(f"{label}: {score:.4f}")

with tab2:
    st.subheader("Connect Gmail and Classify Inbox")

    init_session_state()

    client_id = get_config("GOOGLE_CLIENT_ID", "google_client_id")
    client_secret = get_config("GOOGLE_CLIENT_SECRET", "google_client_secret")
    redirect_uri = get_config(
        "GOOGLE_REDIRECT_URI",
        "google_redirect_uri",
        default="http://localhost:8501/oauth2callback",
    )

    if not client_id or not client_secret or not redirect_uri:
        st.error(
            "Missing Google OAuth configuration. Add GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, "
            "and GOOGLE_REDIRECT_URI in Streamlit Secrets."
        )
        st.stop()

    params = get_query_params()
    code = first_param(params, "code")
    returned_state = first_param(params, "state")
    auth_error = first_param(params, "error")

    if auth_error:
        st.error(f"Google OAuth error: {auth_error}")
        clear_query_params()

    if code:
        expected_state = st.session_state.get("oauth_state")

        if expected_state and returned_state and returned_state != expected_state:
            st.error("OAuth state mismatch. Please try again.")
            clear_query_params()
            st.stop()

        try:
            token_data = exchange_code_for_tokens(
                str(code),
                str(client_id),
                str(client_secret),
                str(redirect_uri),
            )

            existing = st.session_state.get("gmail_tokens") or {}

            if "refresh_token" not in token_data and existing.get("refresh_token"):
                token_data["refresh_token"] = existing["refresh_token"]

            st.session_state["gmail_tokens"] = token_data
            st.success("Gmail connected successfully.")
        except requests.HTTPError as e:
            try:
                error_body = e.response.text
            except Exception:
                error_body = str(e)
            st.error(f"Token exchange failed: {error_body}")
        except Exception as e:
            st.error(f"Unexpected OAuth error: {e}")

        clear_query_params()
        safe_rerun()

    auth_url = build_google_auth_url(
        str(client_id),
        str(redirect_uri),
        str(st.session_state["oauth_state"]),
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        if st.session_state.get("gmail_tokens"):
            st.success("Gmail is connected.")
        else:
            st.markdown(
    f'''
    <a href="{auth_url}" target="_self">
        <button style="
            width: 100%;
            padding: 0.6rem 1rem;
            border-radius: 0.5rem;
            border: 1px solid #ccc;
            background-color: #f0f2f6;
            cursor: pointer;
            font-size: 16px;
        ">
            Connect Gmail
        </button>
    </a>
    ''',
    unsafe_allow_html=True,
)

    with col2:
        if st.button("Disconnect Gmail", use_container_width=True):
            clear_tokens()
            st.success("Gmail tokens cleared from this session.")
            safe_rerun()

    st.caption(f"Redirect URI: {redirect_uri}")

    tokens = st.session_state.get("gmail_tokens")
    access_token = get_valid_access_token(tokens, str(client_id), str(client_secret)) if tokens else None

    if access_token:
        email_limit = st.slider("Number of inbox emails to fetch", 1, 10, 5, 1)
        gmail_threshold = st.slider("Classification threshold", 0.10, 0.90, 0.35, 0.05, key="gmail_threshold")

        if st.button("Fetch and Classify Inbox Emails", use_container_width=True):
            try:
                emails = fetch_recent_emails(access_token, max_results=email_limit)

                if not emails:
                    st.info("No emails found.")
                else:
                    for idx, email_item in enumerate(emails, start=1):
                        labels, scores = predict_email_labels_with_probs(
                            email_item["subject"],
                            email_item["body"],
                            model,
                            tfidf,
                            mlb,
                            threshold=gmail_threshold,
                        )

                        with st.expander(f"{idx}. {email_item['subject']}", expanded=(idx == 1)):
                            st.write(f"**From:** {email_item['from']}")
                            if email_item["date"]:
                                st.write(f"**Date:** {email_item['date']}")
                            st.write(f"**Snippet:** {email_item['snippet']}")

                            preview_body = email_item["body"][:1500] if email_item["body"] else ""
                            st.write("**Body Preview:**")
                            st.write(preview_body if preview_body else "No plain-text body found.")

                            st.write("**Predicted Labels:**")
                            for label in labels:
                                st.write(f"- {label}")

                            st.write("**Probabilities:**")
                            for label, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                                st.write(f"{label}: {score:.4f}")

            except requests.HTTPError as e:
                try:
                    error_body = e.response.text
                except Exception:
                    error_body = str(e)
                st.error(f"Gmail API error: {error_body}")
            except Exception as e:
                st.error(f"Unexpected Gmail processing error: {e}")
    else:
        st.info("Connect Gmail first to fetch inbox emails.")