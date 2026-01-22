import sys
import os
import torch
import cv2
import numpy as np

# ajout du dossier src au chemin pour que python trouve tes modules
sys.path.append(os.path.join(os.getcwd(), 'src'))

# import des classes de prediction qu'on vient de creer
try:
    from image_models.predict.predict_M1_dino import PredictorM1
    from image_models.predict.predict_M2_xgboost import PredictorM2
    from image_models.predict.predict_M3_effnet import PredictorM3
    from image_models.predict.predict_M4_overfit import PredictorM4
    print("✅ imports python : ok")
except ImportError as e:
    print(f"❌ erreur import : {e}")
    print("verifie que tu as bien cree les fichiers __init__.py dans src/image_models/predict/")
    sys.exit(1)

# config des chemins
OUTPUT_DIR = r"C:\Users\amisf\Desktop\datascientest_projet\implementation\outputs"
IMG_TEST_DIR = r"C:\Users\amisf\Desktop\datascientest_projet\data\raw\images\images\image_train"

# on cherche une image au pif pour tester
test_img_path = None
if os.path.exists(IMG_TEST_DIR):
    for f in os.listdir(IMG_TEST_DIR):
        if f.endswith(".jpg"):
            test_img_path = os.path.join(IMG_TEST_DIR, f)
            break

if not test_img_path:
    print("⚠️ attention : pas d'image trouvee pour le test technique.")
    # on cree une fausse image noire pour tester quand meme le code
    test_img_path = "dummy_test.jpg"
    cv2.imwrite(test_img_path, np.zeros((500,500,3), dtype=np.uint8))
    print("   -> image factice créée pour le test.")

print(f"\n--- lancement du crash test sur : {os.path.basename(test_img_path)} ---")

# --- test m1 (dino) ---
print("\n[test m1 : deep learning dinov3]")
try:
    path_m1 = os.path.join(OUTPUT_DIR, "M1_IMAGE_DeepLearning_DINOv3.pth")
    if not os.path.exists(path_m1): raise FileNotFoundError(f"fichier manquant : {path_m1}")
    
    predictor = PredictorM1(path_m1)
    res = predictor.predict(test_img_path)
    print(f"✅ m1 succes | prediction classe : {res}")
except Exception as e:
    print(f"❌ m1 echec : {e}")

# --- test m2 (xgboost) ---
print("\n[test m2 : classic xgboost]")
try:
    path_json = os.path.join(OUTPUT_DIR, "M2_IMAGE_Classic_XGBoost.json")
    path_pkl = os.path.join(OUTPUT_DIR, "M2_IMAGE_XGBoost_Encoder.pkl")
    
    if not os.path.exists(path_json): raise FileNotFoundError(f"manque json : {path_json}")
    if not os.path.exists(path_pkl): raise FileNotFoundError(f"manque encoder : {path_pkl}")
    
    # note : predict_m2 a besoin de 2 arguments
    predictor = PredictorM2(path_json, path_pkl)
    res = predictor.predict(test_img_path)
    print(f"✅ m2 succes | prediction classe : {res}")
except Exception as e:
    print(f"❌ m2 echec : {e}")

# --- test m3 (efficientnet) ---
print("\n[test m3 : classic efficientnet]")
try:
    path_m3 = os.path.join(OUTPUT_DIR, "M3_IMAGE_Classic_EfficientNetB0.pth")
    if not os.path.exists(path_m3): raise FileNotFoundError(f"fichier manquant : {path_m3}")
    
    predictor = PredictorM3(path_m3)
    res = predictor.predict(test_img_path)
    print(f"✅ m3 succes | prediction classe : {res}")
except Exception as e:
    print(f"❌ m3 echec : {e}")

# --- test m4 (phoenix overfit) ---
print("\n[test m4 : deep learning overfit phoenix]")
try:
    path_m4 = os.path.join(OUTPUT_DIR, "M4_IMAGE_DeepLearning_OVERFIT_ResNet.pth")
    if not os.path.exists(path_m4): raise FileNotFoundError(f"fichier manquant : {path_m4}")
    
    predictor = PredictorM4(path_m4)
    res = predictor.predict(test_img_path)
    print(f"✅ m4 succes | prediction classe : {res}")
except Exception as e:
    print(f"❌ m4 echec : {e}")

print("\n--- fin du test ---")
