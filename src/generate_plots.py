import json
import os
import matplotlib.pyplot as plt
import pandas as pd

def load_results():
    """Carrega todos os resultados salvos"""
    results_dir = os.path.join('..', 'data', 'processed', 'results')
    all_results = []
    
    for filename in os.listdir(results_dir):
        if filename.endswith('.json'):
            with open(os.path.join(results_dir, filename)) as f:
                data = json.load(f)
                all_results.append(data)
    
    return pd.DataFrame(all_results)

def plot_metrics_comparison(df):
    """Gera gráfico comparativo das métricas"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Agrupa por modelo e pega a média
    grouped = df.groupby('model').mean()
    
    # Plot
    grouped[['test_accuracy', 'test_f1']].plot(
        kind='bar', 
        ax=ax,
        title='Comparação de Desempenho dos Modelos'
    )
    
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Salva figura
    plots_dir = os.path.join('..', 'data', 'processed', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'model_comparison.png'))
    plt.close()

if __name__ == '__main__':
    results_df = load_results()
    plot_metrics_comparison(results_df)
    print("Gráficos gerados com sucesso!")
