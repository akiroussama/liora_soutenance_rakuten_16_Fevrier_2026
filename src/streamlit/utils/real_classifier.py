import os, sys, joblib, json
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import MODELS_DIR, IMAGE_MODEL_PATH, TEXT_MODEL_PATH, TFIDF_VECTORIZER_PATH, CATEGORY_MAPPING_PATH
from src.models.predict_model import VotingPredictor

class MultimodalClassifier:
    def __init__(self):
        # charge mon voting
        try:
            self.voting = VotingPredictor(MODELS_DIR)
            self.voting.load_models()
        except: self.voting = None
        # charge texte oussama
        try:
            self.text_model = joblib.load(TEXT_MODEL_PATH)
            self.tfidf = joblib.load(TFIDF_VECTORIZER_PATH)
        except:
            self.text_model = None
            self.tfidf = None
        # charge mapping
        with open(CATEGORY_MAPPING_PATH, 'r') as f: self.mapping = json.load(f)

    def predict_image(self, p):
        if not self.voting: return []
        res = self.voting.predict(p)
        for r in res: r['name'] = self.mapping.get(str(r['label']), r['label'])
        return res

    def predict_text(self, t):
        if not self.text_model: return []
        v = self.tfidf.transform([t])
        probs = self.text_model.predict_proba(v)[0]
        ids = probs.argsort()[-5:][::-1]
        return [{"label": str(self.text_model.classes_[i]), 
                 "name": self.mapping.get(str(self.text_model.classes_[i]), str(self.text_model.classes_[i])),
                 "confidence": float(probs[i])} for i in ids]
