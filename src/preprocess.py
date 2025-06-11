import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os

# Verificar e baixar recursos necessários do NLTK
resources = ['stopwords', 'punkt', 'wordnet', 'omw-1.4']

def check_and_download(resource):
    """Verifica e baixa recursos necessários do NLTK."""
    
    try:
        nltk.data.find(f'corpora/{resource}')
    except LookupError:
        print(f"Baixando recurso: {resource}")
        nltk.download(resource)

for resource in resources:
    check_and_download(resource)

# Função de limpeza de texto
def clean_text(text):
    """Limpa e padroniza o texto, removendo caracteres especiais e stopwords."""

    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    return " ".join(tokens)

# Mapas de renomeação por arquivo
column_mappings = {
    "Ling.csv":          {"subject": "subject", "body": "body", "label": "label"},
    "Nigerian_Fraud.csv":{"subject": "subject", "body": "body", "label": "label", "sender": "sender", "urls": "urls"},
    "SpamAssasin.csv":   {"subject": "subject", "body": "body", "label": "label", "sender": "sender", "urls": "urls"},
    "Enron.csv":         {"subject": "subject", "body": "body", "label": "label"},
    "Nazario.csv":       {"subject": "subject", "body": "body", "label": "label", "sender": "sender", "urls": "urls"},
    "phishing_email.csv":{"body": "text_combined", "label": "label"}
}

base_path = "data/raw"
output_path = "data/processed"
os.makedirs(output_path, exist_ok=True)

def run_preprocessing():
    """Executa o pré-processamento de todos os arquivos de dados."""

    for file_name, mapping in column_mappings.items():
        print(f"\nProcessando: {file_name}")
        file_path = os.path.join(base_path, file_name)

        try:
            df = pd.read_csv(file_path)

            # Renomear colunas
            df = df.rename(columns=mapping)

            # Garantir existência das colunas essenciais
            df["subject"] = df.get("subject", "")
            df["body"] = df["body"].fillna("")

            # Texto combinado
            df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")

            # Domínio do remetente, se existir
            df["sender_domain"] = df["sender"].apply(lambda x: x.split("@")[-1] if pd.notnull(x) else "") if "sender" in df else ""

            # Contar URLs, se existir
            df["url_count"] = df["urls"].apply(lambda x: len(str(x).split()) if pd.notnull(x) else 0) if "urls" in df else 0

            # Limpeza
            df["cleaned_text"] = df["text"].apply(clean_text)

            # Salvar
            out_file = file_name.replace(".csv", "_processed.csv")
            df.to_csv(os.path.join(output_path, out_file), index=False)
            print(f"Salvo em: {output_path}/{out_file}")

        except Exception as e:
            print(f"Erro ao processar {file_name}: {e}")
