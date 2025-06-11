import sys

try:
    from data_loader import check_folders, check_datasets
    check_folders()
    check_datasets()
except Exception as e:
    print(f"[ERRO] data_loader: {e}")
    sys.exit(1)

try:
    from preprocess import run_preprocessing
    run_preprocessing()
except Exception as e:
    print(f"[ERRO] preprocess: {e}")
    sys.exit(1)

try:
    from train_model import run_training
    run_training()
except Exception as e:
    print(f"[ERRO] train_model: {e}")
    sys.exit(1)
    
try:
    from predict_email import run_prediction
    run_prediction()
except Exception as e:
    print(f"[ERRO] predict_email: {e}")
    # Não finaliza o script, pois é opcional
