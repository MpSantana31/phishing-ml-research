import os
import re
import joblib
import pandas as pd
from scipy.sparse import hstack, csr_matrix

# Dependências do NLTK para a função de limpeza
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- Bloco de Pré-requisitos ---
# Esta função de limpeza deve ser IDÊNTICA à usada no treinamento.

def _ensure_nltk_resources():
    """Garante que os recursos do NLTK (stopwords, punkt) estejam disponíveis."""
    try:
        stopwords.words("english")
        nltk.data.find('tokenizers/punkt')
    except (LookupError, OSError):
        print("📥 Baixando recursos necessários do NLTK (stopwords, punkt)...")
        nltk.download("stopwords", quiet=True)
        nltk.download("punkt", quiet=True)

def clean_text(text: str) -> str:
    """
    Aplica uma limpeza básica a uma string de texto.
    (Mesma função usada no pré-processamento).
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Mantém apenas letras e espaços
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# --- Fim do Bloco de Pré-requisitos ---


def predict_email(email_text: str, sender_domain: str, url_count: int) -> tuple[str, float]:
    """
    Carrega os modelos treinados e classifica um novo e-mail.

    Args:
        email_text: O corpo/assunto do e-mail.
        sender_domain: O domínio do remetente (ex: 'paypal.com', 'weird-domain.xyz').
        url_count: A quantidade de URLs no e-mail.

    Returns:
        Uma tupla contendo a classificação ('Phishing' ou 'Legit') e a confiança do modelo.
    """
    MODELS_DIR = "models"
    VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    ENCODER_PATH = os.path.join(MODELS_DIR, "onehot_encoder.pkl")
    MODEL_PATH = os.path.join(MODELS_DIR, "spam_classifier.pkl")

    # 1. Verificar se os modelos existem
    for path in [VECTORIZER_PATH, ENCODER_PATH, MODEL_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Arquivo de modelo não encontrado em: {path}. "
                "Execute o script de treinamento primeiro."
            )

    # 2. Carregar os componentes salvos
    vectorizer = joblib.load(VECTORIZER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    model = joblib.load(MODEL_PATH)
    
    # 3. Limpar e transformar os novos dados EXATAMENTE como no treino
    cleaned_text = clean_text(email_text)
    
    # É crucial usar '.transform()' e não '.fit_transform()'
    text_vector = vectorizer.transform([cleaned_text])
    
    # O encoder espera um DataFrame ou estrutura 2D
    domain_encoded = encoder.transform(pd.DataFrame({'sender_domain': [sender_domain]}))
    
    numerical_features = csr_matrix([[url_count]])

    # 4. Juntar todas as features na ordem correta
    features = hstack([text_vector, domain_encoded, numerical_features])

    # 5. Fazer a predição
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features).max()

    return prediction, probability


if __name__ == "__main__":
    # Garante que NLTK está pronto
    _ensure_nltk_resources()

    print("---" * 15)
    print("🧪 Executando testes de predição...")
    print("---" * 15)

    exemplos = [
        # PHISHING/SPAM
        {
            "tipo": "Phishing/Spam",
            "text": "Dear customer, your account is suspended. Please click here to update your details immediately.",
            "domain": "servlce-paypal-security.com",
            "urls": 1
        },
        {
            "tipo": "Phishing/Spam",
            "text": "Your account will be blocked! Click here to update your information immediately.",
            "domain": "itau.com.seguro-alerta.com",
            "urls": 1
        },
        {
            "tipo": "Phishing/Spam",
            "text": "Congratulations! You have won an iPhone. Click to claim your prize.",
            "domain": "promo-premios.com",
            "urls": 2
        },
        {
            "tipo": "Phishing/Spam",
            "text": "We detected suspicious activity. Log in to protect your account.",
            "domain": "security-update.com",
            "urls": 1
        },
        {
            "tipo": "Phishing/Spam",
            "text": "Urgent update: please confirm your banking details now.",
            "domain": "banking-alert.com",
            "urls": 1
        },
        # LEGITIMATE
        {
            "tipo": "Legitimate",
            "text": "Hi Maria, attached is the requested report. Let me know if you have any questions.",
            "domain": "empresa.com.br",
            "urls": 0
        },
        {
            "tipo": "Legitimate",
            "text": "Reminder: project meeting tomorrow at 3pm in room 2.",
            "domain": "universidade.edu",
            "urls": 0
        },
        {
            "tipo": "Legitimate",
            "text": "Your payment receipt is available in the employee portal.",
            "domain": "empresa.com",
            "urls": 0
        },
        {
            "tipo": "Legitimate",
            "text": "Hello, here is the link for our video call at 10am.",
            "domain": "google.com",
            "urls": 1
        },
        {
            "tipo": "Legitimate",
            "text": "Hi team, just a reminder about our meeting tomorrow at 10am. See you there. Best, John",
            "domain": "google.com",
            "urls": 0
        },
    ]

    resultados = []
    for idx, email in enumerate(exemplos, 1):
        print(f"\n📧 Testando E-mail {idx} ({email['tipo']}):")
        print(f"  Texto: \"{email['text']}\"")
        print(f"  Domínio: {email['domain']}")
        print(f"  Nº de URLs: {email['urls']}")
        try:
            pred, prob = predict_email(email["text"], email["domain"], email["urls"])
            print(f"\n✅ Classificação: {pred}")
            print(f"   Confiança do Modelo: {prob:.2%}")
            resultados.append({
                "tipo": email["tipo"],
                "pred": pred,
                "prob": prob,
                # Gabarito esperado: 1 para phishing/spam, 0 para legítimo
                "esperado": 1 if email["tipo"].lower().startswith("phishing") else 0
            })
        except FileNotFoundError as e:
            print(f"\n❌ ERRO: {e}")
        print("---" * 15)

    # Resumo final
    if resultados:
        media_confianca = sum(r["prob"] for r in resultados) / len(resultados)
        acertos = sum(1 for r in resultados if r["pred"] == r["esperado"])
        acuracia = acertos / len(resultados)
        print("\n===== RESUMO FINAL =====")
        print(f"Média da confiança dos exemplos: {media_confianca:.2%}")
        print(f"Acurácia geral nos exemplos: {acuracia:.2%} ({acertos}/{len(resultados)})")