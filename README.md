# Phishing Email Detection Pipeline

Este projeto implementa um pipeline completo de detecção de e-mails de phishing utilizando técnicas de processamento de linguagem natural (NLP) e machine learning. Ele foi desenvolvido para um artigo acadêmico de faculdade, com foco em automação, reprodutibilidade e clareza de etapas.

## Objetivo
Detectar automaticamente e-mails de phishing a partir de diferentes bases públicas, utilizando pré-processamento de texto, extração de features e treinamento de modelos de classificação.

## Modelos Implementados

- **SGDClassifier**: Versão rápida de regressão logística
- **HistGradientBoosting**: Alternativa rápida ao XGBoost
- **Naive Bayes**: Mantido com parâmetros otimizados para velocidade
- **Random Forest**: Mantido com parâmetros otimizados para velocidade

## Como funciona
O pipeline executa as seguintes etapas, todas automatizadas pelo arquivo `src/main.py`:

1. **Verificação e download dos dados:** Garante que as pastas e os datasets necessários estejam presentes.
2. **Pré-processamento:** Limpa e padroniza os dados, combinando colunas relevantes e extraindo features como domínio do remetente e contagem de URLs.
3. **Treinamento do modelo:** Realiza a vetorização dos textos, codificação das features e treina um modelo Naive Bayes, salvando o modelo e os transformadores.
4. **Predição de exemplo:** Realiza uma predição de teste com um e-mail exemplo, mostrando o resultado no console.

## Como executar

### Pré-requisitos
- Python 3.8+
- Instalar dependências:

```bash
pip install -r requirements.txt
```

### Execução
Na raiz do projeto, execute:

```bash
python3 src/main.py
```

O pipeline irá baixar (se necessário) e preparar os dados, treinar o modelo e mostrar uma predição de exemplo.

## Estrutura dos arquivos principais

- `src/data_loader.py`: Funções para checagem de pastas e download dos datasets
- `src/preprocess.py`: Função para pré-processamento dos dados (`run_preprocessing`)
- `src/train_model.py`: Função para treinamento e avaliação do modelo (`run_training`)
- `src/predict_email.py`: Função para predição de exemplo (`run_prediction`)
- `src/main.py`: Arquivo principal que executa todo o fluxo em ordem

## Bases de dados utilizadas
- [Phishing Email Dataset (Kaggle)](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset)

## Créditos e referências
- Projeto desenvolvido para fins acadêmicos.
- Bases públicas de e-mails de phishing e spam.
- Bibliotecas: pandas, scikit-learn, nltk, joblib, scipy.

## Otimizações

- Substituição dos modelos lentos (SVM e XGBoost) por alternativas mais rápidas
- Parâmetros ajustados para melhor equilíbrio entre performance e velocidade

## Resultados

### Métricas dos Modelos (última execução)

| Modelo | Acurácia (Teste) | F1-Score (Teste) | F1-Score (CV Mean ± Std) |
|--------|------------------|------------------|--------------------------|
| Naive Bayes | 0.9725 | 0.9725 | 0.9682 ± 0.0059 |
| Logistic Regression | 0.9807 | 0.9807 | 0.9759 ± 0.0040 |
| Random Forest | 0.8702 | 0.8686 | 0.8586 ± 0.0164 |
| SGDClassifier | 0.9580 | 0.9580 | 0.9532 ± 0.0053 |
| HistGradientBoosting | 0.9326 | 0.9326 | 0.9296 ± 0.0051 |

**Melhor modelo**: Logistic Regression (F1=0.9807)

### Visualização de Resultados

O projeto agora inclui:
1. Salvamento automático de métricas em `data/processed/results/`
2. Script `generate_plots.py` para gerar gráficos comparativos
3. Gráficos salvos em `data/processed/plots/`

Para gerar os gráficos:
```bash
python src/generate_plots.py
```