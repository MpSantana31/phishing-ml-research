import re
import joblib
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix

def clean_text(text):
    """Limpa e padroniza o texto, removendo caracteres especiais e stopwords."""
    
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    return " ".join(tokens)

def extract_domain(sender):
    """Extrai o domínio do remetente."""
    
    return sender.split("@")[-1] if sender and "@" in sender else ""

def count_urls(urls):
    """Conta o número de URLs no texto."""
    
    return len(str(urls).split()) if urls else 0

def run_prediction():
    """Executa a predição de um e-mail de exemplo."""
    
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    encoder = joblib.load("models/onehot_encoder.pkl")
    model = joblib.load("models/spam_classifier.pkl")

    test_email = {
        "subject": "Urgent: Account Verification Needed",
        "body": "Dear user, please click the link below to verify your account immediately.",
        "sender": "alert@secure-bank.com",
        "urls": "http://fakebank.com/verify"
    }

    text = (test_email.get("subject") or "") + " " + (test_email.get("body") or "")
    cleaned_text = clean_text(text)
    sender_domain = extract_domain(test_email.get("sender", ""))
    url_count = count_urls(test_email.get("urls", ""))

    text_vector = vectorizer.transform([cleaned_text])
    domain_vector = encoder.transform([[sender_domain]])
    numeric_vector = csr_matrix([[url_count]])

    X = hstack([text_vector, domain_vector, numeric_vector])

    pred = model.predict(X)[0]
    print(f"\U0001f4e8 Previsão para o e-mail: {pred}")