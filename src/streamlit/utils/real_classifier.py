import sys
import os
import joblib
import json
import numpy as np
from pathlib import Path

# --- CONFIGURATION DES CHEMINS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from config import MODELS_DIR, TEXT_MODEL_PATH, CATEGORY_MAPPING_PATH, FUSION_W_IMAGE, FUSION_W_TEXT
from src.models.predict_model import VotingPredictor

class MultimodalClassifier:
    def __init__(self):
        self.w_text = FUSION_W_TEXT
        self.w_image = FUSION_W_IMAGE
        
        # 1. Chargement du Mapping
        try:
            with open(CATEGORY_MAPPING_PATH, 'r', encoding='utf-8') as f:
                self.mapping = json.load(f)
        except:
            try:
                with open(CATEGORY_MAPPING_PATH, 'r') as f: self.mapping = json.load(f)
            except: self.mapping = {}

        # 2. Chargement Image (Voting)
        try:
            self.voting = VotingPredictor(MODELS_DIR)
            self.voting.load_models()
        except Exception as e:
            print(f"Erreur Image: {e}")
            self.voting = None

        # 3. Chargement Texte (Pipeline)
        try:
            self.text_model = joblib.load(TEXT_MODEL_PATH)
        except Exception as e:
            print(f"Erreur Texte: {e}")
            self.text_model = None

    def _format_result(self, label, score):
        """Helper pour formater joliment un résultat"""
        return {
            "label": str(label),
            "name": self.mapping.get(str(label), f"Produit Type {label}"),
            "confidence": float(score)
        }

    def predict_image(self, image_path):
        """Prédiction Image pure"""
        if not self.voting: return []
        try:
            # Le voting renvoie déjà une liste triée
            raw_res = self.voting.predict(image_path)
            # On formate
            return [self._format_result(r['label'], r['confidence']) for r in raw_res]
        except Exception as e:
            print(f"Bug Image: {e}")
            return []

    def predict_text(self, text):
        """Prédiction Texte pure (avec gestion LinearSVC)"""
        if not self.text_model: return []
        try:
            if isinstance(text, str): text = [text]

            # Calcul des probabilités (même si LinearSVC)
            if hasattr(self.text_model, "predict_proba"):
                probs = self.text_model.predict_proba(text)[0]
            elif hasattr(self.text_model, "decision_function"):
                scores = self.text_model.decision_function(text)[0]
                exp_scores = np.exp(scores - np.max(scores))
                probs = exp_scores / exp_scores.sum()
            else:
                return []

            # On récupère TOUTES les classes pour la fusion
            results = []
            for i, class_id in enumerate(self.text_model.classes_):
                results.append(self._format_result(class_id, probs[i]))
            
            # On trie pour l'affichage direct (Top 5)
            return sorted(results, key=lambda x: x['confidence'], reverse=True)
        except Exception as e:
            print(f"Bug Texte: {e}")
            return []

    def predict_fusion(self, text, image_path):
        """LA FUSION : Combine les scores Texte et Image"""
        
        # 1. Obtenir les scores bruts (tous les scores, pas juste le top 5)
        res_text = self.predict_text(text)
        res_image = self.predict_image(image_path)
        
        # Dictionnaire pour fusionner : {label: score_fusionné}
        fusion_scores = {}
        
        # On remplit avec le Texte
        for item in res_text:
            label = item['label']
            fusion_scores[label] = item['confidence'] * self.w_text
            
        # On ajoute l'Image
        for item in res_image:
            label = item['label']
            current_score = fusion_scores.get(label, 0.0)
            fusion_scores[label] = current_score + (item['confidence'] * self.w_image)
            
        # 2. On transforme en liste propre et on trie
        final_results = []
        for label, score in fusion_scores.items():
            final_results.append(self._format_result(label, score))
            
        return sorted(final_results, key=lambda x: x['confidence'], reverse=True)
        