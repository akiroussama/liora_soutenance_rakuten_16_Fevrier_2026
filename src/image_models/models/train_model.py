# src/models/train_model.py
import xgboost as xgb
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# je definis les chemins
BASE_DIR = r"C:\Users\amisf\Desktop\datascientest_projet"
OUT_DIR = os.path.join(BASE_DIR, "implementation", "outputs")

def train_xgboost():
    print("debut train xgboost")
    
    # chargement features (supposées existantes sur disque)
    # on utilise celles du drive si besoin
    x_path = os.path.join(OUT_DIR, 'train_features_resnet50_augmented.npy')
    y_path = os.path.join(OUT_DIR, 'train_labels_augmented.npy')
    
    if not os.path.exists(x_path):
        print("err: features introuvables")
        return

    x_data = np.load(x_path)
    y_data = np.load(y_path)
    
    # encodage
    le = LabelEncoder()
    y_enc = le.fit_transform(y_data)
    
    # save encodeur pour la prediction plus tard
    joblib.dump(le, os.path.join(OUT_DIR, "label_encoder_xgb.pkl"))
    
    # split
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_enc, test_size=0.2)
    
    # config
    params = {
        'objective': 'multi:softmax',
        'num_class': len(le.classes_),
        'n_estimators': 3000,
        'max_depth': 8,
        'learning_rate': 0.05,
        'tree_method': 'hist',
        'device': 'cuda' # ou cpu
    }
    
    # fit
    model = xgb.XGBClassifier(**params)
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=100)
    
    # save
    save_path = os.path.join(OUT_DIR, "best_xgboost_gpu_model.json")
    model.save_model(save_path)
    print(f"model saved: {save_path}")

if __name__ == "__main__":
    train_xgboost()