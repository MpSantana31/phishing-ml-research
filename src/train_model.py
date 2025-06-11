import os
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from scipy.sparse import hstack, csr_matrix
import joblib

def ensure_models_directory():
    """Garante que a pasta de modelos exista."""
    
    models_path = "models"
    if not os.path.exists(models_path):
        os.makedirs(models_path)
        print(f"Pasta '{models_path}' criada.")
    else:
        print(f"Pasta '{models_path}' já existe.")

def run_training():
    """Executa o treinamento do modelo."""
    
    # Carregar e unificar todos os arquivos processados
    processed_path = "data/processed"
    dfs = []

    for file in os.listdir(processed_path):
        if file.endswith("_processed.csv"):
            print(f"\U0001f504 Lendo {file}...")
            df = pd.read_csv(os.path.join(processed_path, file))

            # Verifica se possui todas as colunas esperadas
            required_columns = ["cleaned_text", "label"]
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"O arquivo {file} não contém a coluna obrigatória: {col}")

            # Garante que colunas opcionais existam
            df["sender_domain"] = df.get("sender_domain", "")
            df["url_count"] = df.get("url_count", 0)
            dfs.append(df)

    ensure_models_directory()

    # Concatenar todos os DataFrames
    df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal de e-mails: {len(df)}")

    # Vetorização do texto
    vectorizer = TfidfVectorizer(max_features=5000)
    text_vectors = vectorizer.fit_transform(df["cleaned_text"])

    # One-hot encoding no domínio do remetente
    encoder = OneHotEncoder(handle_unknown="ignore")
    domain_encoded = encoder.fit_transform(df[["sender_domain"]])

    # Dados numéricos (url_count)
    numerical = csr_matrix(df[["url_count"]].fillna(0).values)

    # Juntar todas as features
    X = hstack([text_vectors, domain_encoded, numerical])
    y = df["label"]

    # Dividir em treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Tamanho do treino: {X_train.shape[0]}")
    print(f"Tamanho do teste: {X_test.shape[0]}")

    # Salvar vectorizer e encoder
    os.makedirs("models", exist_ok=True)
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
    joblib.dump(encoder, "models/onehot_encoder.pkl")

    # Treinar o modelo
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Avaliação
    y_pred = model.predict(X_test)
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))

    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")
    print("F1-scores na validação cruzada:", scores)
    print("Média F1:", scores.mean())

    # Salvar modelo
    joblib.dump(model, "models/spam_classifier.pkl")
    print("\nModelo salvo em: models/spam_classifier.pkl")
