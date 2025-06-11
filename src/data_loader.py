import os
import pandas as pd
import zipfile

def check_folders():
    """Verifica e cria as pastas necessárias se não existirem"""
    required_folders = [
        "data/raw",
        "data/processed"
    ]
    
    for folder in required_folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Pasta verificada/criada: {folder}")

def check_datasets():
    """Verifica se todos os datasets necessários existem e retorna o status."""
    raw_dir = "data/raw"
    required_files = [
        "Ling.csv",
        "Nigerian_Fraud.csv",
        "SpamAssasin.csv",
        "Enron.csv",
        "Nazario.csv",
        "phishing_email.csv"
    ]
    
    # Verificar quais arquivos estão faltando
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(raw_dir, f))]
    
    if not missing_files:
        print("\nTodos os datasets foram encontrados em:", raw_dir)
        return True
    else:
        print("\nArquivos faltando:", ", ".join(missing_files))
        print("\nBaixando dataset do Kaggle...")
        
        # Baixar o arquivo zip
        download_path = os.path.join(raw_dir, "phishing-email-dataset.zip")
        os.system(f'curl -L -o "{download_path}" '
                 'https://www.kaggle.com/api/v1/datasets/download/naserabdullahalam/phishing-email-dataset')
        
        # Extrair o arquivo zip
        try:
            with zipfile.ZipFile(download_path, 'r') as zip_ref:
                zip_ref.extractall(raw_dir)
            print("Arquivo extraído com sucesso!")
            os.remove(download_path)
            
            # Verificar novamente após extração
            still_missing = [f for f in missing_files if not os.path.exists(os.path.join(raw_dir, f))]
            if still_missing:
                print("\nAinda faltam arquivos após extração:", ", ".join(still_missing))    
                return False
            else:
                print("\nTodos os arquivos foram restaurados com sucesso!")
                return True
                
        except Exception as e:
            print(f"Erro ao extrair arquivo: {e}")
            print("\nComo baixar manualmente:")
            print("1. Acesse: https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset")
            print(f"2. Extraia os arquivos para: {raw_dir}")
            return False
