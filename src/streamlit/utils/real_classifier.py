"""
Production multimodal classifier — uses real trained models.

Supports three prediction modes:
  - Text only:  TF-IDF + LinearSVC pipeline (83% accuracy)
  - Image only: Voting System with 3 models (92% accuracy)
  - Fusion:     Weighted average of both (60% image + 40% text = ~94%)

The text model uses decision_function + softmax to produce probabilities
from LinearSVC (which doesn't natively support predict_proba).
"""
import sys
import os
import joblib
import json
import numpy as np
from pathlib import Path

# Add project root to path for cross-module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from config import MODELS_DIR, TEXT_MODEL_PATH, CATEGORY_MAPPING_PATH, FUSION_W_IMAGE, FUSION_W_TEXT
from src.models.predict_model import VotingPredictor


class MultimodalClassifier:
    """Loads all models once, exposes predict_text / predict_image / predict_fusion."""

    def __init__(self):
        # Fusion weights from config (single source of truth)
        self.w_text = FUSION_W_TEXT
        self.w_image = FUSION_W_IMAGE

        # 1. Category mapping (code -> human-readable name)
        try:
            with open(CATEGORY_MAPPING_PATH, 'r', encoding='utf-8') as f:
                self.mapping = json.load(f)
        except Exception:
            try:
                with open(CATEGORY_MAPPING_PATH, 'r') as f:
                    self.mapping = json.load(f)
            except Exception:
                self.mapping = {}

        # 2. Image model — Voting System (DINOv3 + XGBoost + EfficientNet)
        try:
            self.voting = VotingPredictor(MODELS_DIR)
            self.voting.load_models()
        except Exception as e:
            print(f"Image model error: {e}")
            self.voting = None

        # 3. Text model — TF-IDF FeatureUnion + LinearSVC
        try:
            self.text_model = joblib.load(TEXT_MODEL_PATH)
        except Exception as e:
            print(f"Text model error: {e}")
            self.text_model = None

    def _format_result(self, label, score):
        """Format a single prediction as {label, name, confidence}."""
        return {
            "label": str(label),
            "name": self.mapping.get(str(label), f"Produit Type {label}"),
            "confidence": float(score)
        }

    def predict_image(self, image_path):
        """Run image-only classification through the Voting System."""
        if not self.voting:
            return []
        try:
            raw_res = self.voting.predict(image_path)
            return [self._format_result(r['label'], r['confidence']) for r in raw_res]
        except Exception as e:
            print(f"Image prediction error: {e}")
            return []

    def predict_text(self, text):
        """
        Run text-only classification through LinearSVC.

        LinearSVC uses decision_function (not predict_proba), so we convert
        raw scores to probabilities via softmax: exp(s - max) / sum(exp(s - max)).
        """
        if not self.text_model:
            return []
        try:
            if isinstance(text, str):
                text = [text]

            # Get probabilities from the sklearn pipeline
            if hasattr(self.text_model, "predict_proba"):
                probs = self.text_model.predict_proba(text)[0]
            elif hasattr(self.text_model, "decision_function"):
                scores = self.text_model.decision_function(text)[0]
                # Softmax conversion for LinearSVC raw scores
                exp_scores = np.exp(scores - np.max(scores))
                probs = exp_scores / exp_scores.sum()
            else:
                return []

            # Build results for all 27 classes, sorted by confidence
            results = []
            for i, class_id in enumerate(self.text_model.classes_):
                results.append(self._format_result(class_id, probs[i]))
            return sorted(results, key=lambda x: x['confidence'], reverse=True)

        except Exception as e:
            print(f"Text prediction error: {e}")
            return []

    def predict_fusion(self, text, image_path):
        """
        Late fusion: combine text and image scores with configurable weights.

        For each class, the fused score = w_text * text_score + w_image * image_score.
        This allows classes missed by one modality to be rescued by the other.
        """
        res_text = self.predict_text(text)
        res_image = self.predict_image(image_path)

        # Merge scores by label
        fusion_scores = {}
        for item in res_text:
            fusion_scores[item['label']] = item['confidence'] * self.w_text
        for item in res_image:
            label = item['label']
            fusion_scores[label] = fusion_scores.get(label, 0.0) + (item['confidence'] * self.w_image)

        # Sort and return
        final_results = [self._format_result(label, score) for label, score in fusion_scores.items()]
        return sorted(final_results, key=lambda x: x['confidence'], reverse=True)
