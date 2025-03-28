import streamlit as st
import os
import base64
import re
import nltk
from collections import defaultdict
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from transformers import pipeline, AutoTokenizer

nltk.download('punkt')

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# Base64 encoded credentials string (replace this with your actual Base64 string)
encoded_credentials = """
eyJpbnN0YWxsZWQiOnsiY2xpZW50X2lkIjoiMTQ1NjQzMzM2NjY0LWFvb2M2b3Qz
dHAzdTJzdjVjbWlybm85anBybmRnZzBpLmFwcHMuZ29vZ2xldXNlcmNvbnRlbnQu
Y29tIiwicHJvamVjdF9pZCI6ImdtYWlsdHJ1c3RhcHBmaW5hbCIsImF1dGhfdXJp
IjoiaHR0cHM6Ly9hY2NvdW50cy5nb29nbGUuY29tL28vb2F1dGgyL2F1dGgiLCJ0
b2tlbl91cmkiOiJodHRwczovL29hdXRoMi5nb29nbGVhcGlzLmNvbS90b2tlbiIs
ImF1dGhfcHJvdmlkZXJfeDUwOV9jZXJ0X3VybCI6Imh0dHBzOi8vd3d3Lmdvb2ds
ZWFwaXMuY29tL29hdXRoMi92MS9jZXJ0cyIsImNsaWVudF9zZWNyZXQiOiJHT0NT
UFgtTXhBUEFEbEtYRmZwUnJ5bHpXNWFtbmROdEdoNiIsInJlZGlyZWN0X3VyaXMi
OlsiaHR0cDovL2xvY2FsaG9zdCJdfX0=
"""

# Decode the Base64 credentials and write to a file
credentials_json = base64.b64decode(encoded_credentials).decode('utf-8')

# Write the credentials to a file
with open('credentials.json', 'w') as f:
    f.write(credentials_json)

@st.cache_resource
def gmail_authenticate():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=8765)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('gmail', 'v1', credentials=creds)

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

def extract_email_content(service, max_results=10):
    emails = []
    results = service.users().messages().list(userId='me', maxResults=max_results).execute()
    messages = results.get('messages', [])
    for msg in messages:
        msg_data = service.users().messages().get(userId='me', id=msg['id'], format='full').execute()
        headers = msg_data['payload'].get('headers', [])
        subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
        sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown')
        parts = msg_data['payload'].get('parts', [])
        body = ''
        for part in parts:
            if part.get('mimeType') == 'text/plain':
                body = part.get('body', {}).get('data', '')
                break
        try:
            body = base64.urlsafe_b64decode(body).decode('utf-8')
        except:
            body = ''
        emails.append({'sender': sender, 'subject': subject, 'content': body})
    return emails

def compute_trust_scores(emails, classifier, tokenizer, keyword_weights):
    sender_scores = defaultdict(list)
    for email in emails:
        sender = email['sender']
        text = email['content']
        tokens = tokenizer.encode(text, truncation=True, max_length=512)
        truncated_text = tokenizer.decode(tokens, skip_special_tokens=True)

        result = classifier(truncated_text)[0]
        sentiment_score = 2 * result['score'] - 1 if result['label'] == 'POSITIVE' else 1 - 2 * result['score']

        words = re.findall(r'\b\w+\b', truncated_text.lower())
        keyword_score = sum(
            keyword_weights['positive'].get(w, 0) + keyword_weights['negative'].get(w, 0)
            for w in words
        )
        keyword_score = min(max(keyword_score / 10, -1), 1)

        response_score = 0.5  # placeholder
        final_score = 0.5 * sentiment_score + 0.3 * keyword_score + 0.2 * response_score

        sender_scores[sender].append({
            'subject': email['subject'],
            'sentiment': result['label'],
            'score': round(final_score, 3)
        })

    return sender_scores

st.title("📬 Gmail Trust Classifier")
st.write("This app analyzes the sentiment and content of your Gmail emails and ranks senders by trust.")

credentials_file = 'credentials.json'  # We don't need to upload file now

if os.path.exists(credentials_file):
    service = gmail_authenticate()
    classifier = load_model()
    tokenizer = load_tokenizer()

    keyword_weights = {
        'positive': {'thank': 1, 'reliable': 2, 'trust': 2, 'great': 1, 'help': 1, 'appreciate': 1, 'excellent': 2},
        'negative': {'sorry': -1, 'delay': -1, 'fail': -2, 'issue': -1, 'problem': -1, 'mistake': -2, 'apologies': -1}
    }

    if st.button("📥 Analyze My Inbox"):
        emails = extract_email_content(service)
        results = compute_trust_scores(emails, classifier, tokenizer, keyword_weights)

        for sender, entries in results.items():
            avg_score = round(sum(e['score'] for e in entries) / len(entries), 3)
            st.subheader(sender)
            st.markdown(f"**Average Trust Score**: {avg_score}")
            for entry in entries:
                st.markdown(f"- ✉️ **{entry['subject']}** → _{entry['sentiment']}_ (Score: {entry['score']})")
            st.markdown("---")
else:
    st.warning("`credentials.json` file not found.")
