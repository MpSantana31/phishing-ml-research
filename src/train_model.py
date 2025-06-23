import os
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
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
    """Executa o treinamento e comparação de múltiplos modelos."""
    
    # Carregar e unificar todos os arquivos processados
    processed_path = "data/processed"
    dfs = []

    for file in os.listdir(processed_path):
        if file.endswith("_processed.csv"):
            print(f"Lendo {file}...")
            df = pd.read_csv(os.path.join(processed_path, file))

            # Verifica se possui todas as colunas esperadas
            required_columns = ["cleaned_text", "label"]
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Arquivo {file} não contém coluna obrigatória: {col}")
            dfs.append(df)

    if not dfs:
        raise FileNotFoundError("Nenhum arquivo processado encontrado em 'data/processed'")

    full_df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal de registros: {len(full_df)}")

    # Vetorizar texto
    full_df["cleaned_text"] = full_df["cleaned_text"].fillna("")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_text = vectorizer.fit_transform(full_df["cleaned_text"])
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

    # Codificar domínios
    encoder = OneHotEncoder(handle_unknown='ignore')
    X_domain = encoder.fit_transform(full_df[["sender_domain"]])
    joblib.dump(encoder, "models/onehot_encoder.pkl")

    # Features numéricas
    X_numeric = csr_matrix(full_df[["url_count"]])

    # Juntar todas as features
    X = hstack([X_text, X_domain, X_numeric])
    y = full_df["label"]

    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Definir modelos para comparação (focados em velocidade)
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, n_jobs=-1),
        "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=5, n_jobs=-1),
        "SGDClassifier": SGDClassifier(
            loss='log_loss', 
            penalty='l2',
            max_iter=1000,
            n_jobs=-1,
            early_stopping=True
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=3,
            learning_rate=0.1
        )
    }

    # Se o dataset for muito grande, usar amostra de 10.000 registros
    if X_train.shape[0] > 10000:
        print("\nDataset grande detectado, usando amostra de 10.000 registros para treino...")
        X_train_sample, _, y_train_sample, _ = train_test_split(X_train, y_train, train_size=10000, stratify=y_train, random_state=42)
    else:
        X_train_sample, y_train_sample = X_train, y_train

    # Normalizar dados para SVM
    print("\nNormalizando dados para SVM...")
    scaler = StandardScaler(with_mean=False)  
    X_train_scaled = scaler.fit_transform(X_train_sample)
    X_test_scaled = scaler.transform(X_test)

    # Treinar e avaliar cada modelo
    results = {}
    for name, model in models.items():
        print(f"\nTreinando {name}...")
        
        # Treinar modelo
        if name == "SGDClassifier":
            model.fit(X_train_scaled, y_train_sample)
        elif name == "HistGradientBoosting":
            model.fit(X_train_sample.toarray(), y_train_sample)
        else:
            model.fit(X_train_sample, y_train_sample)
        
        # Avaliar no teste
        if name == "SGDClassifier":
            y_pred = model.predict(X_test_scaled)
        elif name == "HistGradientBoosting":
            y_pred = model.predict(X_test.toarray())
        else:
            y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Validação cruzada
        if name == "SGDClassifier":
            cv_scores = cross_val_score(model, X_train_scaled, y_train_sample, cv=5, scoring="f1")
        elif name == "HistGradientBoosting":
            cv_scores = cross_val_score(model, X_train_sample.toarray(), y_train_sample, cv=5, scoring="f1")
        else:
            cv_scores = cross_val_score(model, X_train_sample, y_train_sample, cv=5, scoring="f1")
        
        # Armazenar resultados
        results[name] = {
            "model": model,
            "test_accuracy": report['accuracy'],
            "test_f1": report['weighted avg']['f1-score'],
            "cv_mean_f1": cv_scores.mean(),
            "cv_std_f1": cv_scores.std()
        }
        
        # Imprimir resultados
        print(f"\n{name} Results:")
        print(f"- Test Accuracy: {results[name]['test_accuracy']:.4f}")
        print(f"- Test F1: {results[name]['test_f1']:.4f}")
        print(f"- CV Mean F1: {results[name]['cv_mean_f1']:.4f} (±{results[name]['cv_std_f1']:.4f})")

    # Selecionar melhor modelo baseado no F1
    best_model_name = max(results.items(), key=lambda x: x[1]['test_f1'])[0]
    best_model = results[best_model_name]['model']
    print(f"\nMelhor modelo: {best_model_name} (F1={results[best_model_name]['test_f1']:.4f})")

    # Salvar todos os modelos individualmente
    ensure_models_directory()
    for name in models.keys():
        model_path = f"models/{name.lower().replace(' ', '_')}.pkl"
        joblib.dump(results[name]['model'], model_path)
        print(f"Modelo {name} salvo em: {model_path}")

    # Salvar melhor modelo
    joblib.dump(best_model, "models/spam_classifier.pkl")
    print(f"\nMelhor modelo ({best_model_name}) salvo em: models/spam_classifier.pkl")
    
    # Retornar resultados para possível análise posterior
    return results