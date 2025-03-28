# 📬 Gmail Trustworthy Classifier

This is a Streamlit-based web app that connects to your Gmail account, analyzes the content of your emails, and classifies senders as **trustworthy**, **neutral**, or **not trustworthy** using sentiment analysis and custom keyword scoring.

---

## 🚀 Features

- 🔐 Google OAuth 2.0 login to access Gmail
- 📥 Fetches and analyzes recent emails from your inbox
- 🤖 Uses HuggingFace's DistilBERT model for sentiment classification
- 🧠 Calculates a custom trust score using:
  - Sentiment of the message
  - Presence of positive/negative keywords
  - (Future) responsiveness or reply behavior
- 📊 Displays sender-wise trust level with expandable email analysis

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Gmail API
- HuggingFace Transformers
- NLTK
- Google Cloud (OAuth setup)

---



## 🧰 Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/Likhil2k1/Gmail_Trustworthy_Classifier.git
cd Gmail_Trustworthy_Classifier
