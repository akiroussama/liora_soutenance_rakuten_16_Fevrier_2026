# src/models/predict_model.py
import torch
import timm
import xgboost as xgb
import joblib
import os
import numpy as np
# j importe mon pre processing
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.image_models.build_features_image import load_and_process_image

# config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = r"C:\Users\amisf\Desktop\datascientest_projet\implementation\outputs"

class Predictor:
    def __init__(self, model_type='dino'):
        self.type = model_type
        self.model = None
        self.le = None
        self.load_model()
        
    def load_model(self):
        print(f"chargement modele {self.type}...")
        
        if self.type == 'dino':
            # load architecture
            self.model = timm.create_model(
                'vit_large_patch14_reg4_dinov2.lvd142m',
                pretrained=False,
                num_classes=27
            ).to(DEVICE)
            
            # load poids
            path = os.path.join(OUT_DIR, "model_DINOv3_BEST.pth")
            if os.path.exists(path):
                self.model.load_state_dict(torch.load(path, map_location=DEVICE))
                self.model.eval()
            else:
                print("err: poids dino introuvables")

        elif self.type == 'xgboost':
            # load xgb
            self.model = xgb.XGBClassifier()
            self.model.load_model(os.path.join(OUT_DIR, "best_xgboost_gpu_model.json"))
            # load labels
            self.le = joblib.load(os.path.join(OUT_DIR, "label_encoder_xgb.pkl"))

    def predict(self, image_path):
        # logique de prediction
        if self.type == 'dino':
            img_t = load_and_process_image(image_path, 'dino')
            if img_t is None: return None
            
            with torch.no_grad():
                out = self.model(img_t.to(DEVICE))
                probs = torch.softmax(out, dim=1)
                top_p, top_class = probs.topk(1, dim=1)
                return int(top_class.item())
        
        elif self.type == 'xgboost':
            # attention xgboost a besoin de features pas d image brute
            # ici il faudrait extraire les features d abord
            # c est un placeholder
            print("todo: extraction feature resnet avant predict xgb")
            return 0

# test rapide
if __name__ == "__main__":
    pred = Predictor('dino')
    # mettre un chemin reel pour tester
    # res = pred.predict("chemin/vers/image.jpg")
    # print(f"classe predite: {res}")