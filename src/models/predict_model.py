import torch
import torch.nn.functional as F
import xgboost as xgb
import timm, joblib, numpy as np
from pathlib import Path
from torchvision import models
import torch.nn as nn
from src.features.build_features import preprocess_image

class VotingPredictor:
    def __init__(self, models_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mdir = Path(models_dir)
        self.loaded = False

    def load_models(self):
        if self.loaded: return
        self.m1 = timm.create_model('vit_large_patch14_reg4_dinov2.lvd142m', pretrained=False, num_classes=27)
        self.m1.load_state_dict(torch.load(self.mdir/"M1_IMAGE_DeepLearning_DINOv3.pth", map_location=self.device))
        self.m2 = xgb.XGBClassifier()
        self.m2.load_model(str(self.mdir/"M2_IMAGE_Classic_XGBoost.json"))
        self.le = joblib.load(self.mdir/"M2_IMAGE_XGBoost_Encoder.pkl")
        res = models.resnet50(weights=None)
        self.ext = nn.Sequential(*list(res.children())[:-1])
        self.m3 = models.efficientnet_b0(weights=None)
        self.m3.classifier[1] = nn.Linear(1280, 27)
        self.m3.load_state_dict(torch.load(self.mdir/"M3_IMAGE_Classic_EfficientNetB0.pth", map_location=self.device))
        for m in [self.m1, self.m3, self.ext]: m.to(self.device).eval()
        self.loaded = True

    def predict(self, img_p):
        if not self.loaded: self.load_models()
        with torch.no_grad():
            # nous calculons dino le patron
            p1 = F.softmax(self.m1(preprocess_image(img_p, "dino").to(self.device)), dim=1).cpu().numpy()[0]
            
            # nous preparons input standard
            i2 = preprocess_image(img_p, "standard").to(self.device)
            
            # nous calculons efficientnet
            p3 = F.softmax(self.m3(i2), dim=1).cpu().numpy()[0]
            
            # nous calculons xgboost avec correction
            f = self.ext(i2).squeeze().cpu().numpy().reshape(1, -1)
            raw_p2 = self.m2.predict_proba(f)[0]
            
            # nous appliquons le sharpening puissance 3
            # cela force xgboost a prendre position et evite la dilution
            sharp_p2 = np.power(raw_p2, 3)
            p2 = sharp_p2 / sharp_p2.sum() # renormalisation

        # nous appliquons le vote pondere mis a jour
        # dino=4, effnet=2, xgboost=1
        f_p = (4.0 * p1 + 1.0 * p2 + 2.0 * p3) / 7.0
        
        # nous recuperons le top 5
        ids = np.argsort(f_p)[-5:][::-1]
        return [{"label": str(self.le.inverse_transform([i])[0]), "confidence": float(f_p[i])} for i in ids]