from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
IMAGES_DIR = RAW_DATA_DIR / "images"
MODELS_DIR = PROJECT_ROOT / "models"
IMAGE_MODEL_PATH = MODELS_DIR / "M1_IMAGE_DeepLearning_DINOv3.pth"
XGB_MODEL_PATH = MODELS_DIR / "M2_IMAGE_Classic_XGBoost.json"
XGB_ENC_PATH = MODELS_DIR / "M2_IMAGE_XGBoost_Encoder.pkl"
EFF_MODEL_PATH = MODELS_DIR / "M3_IMAGE_Classic_EfficientNetB0.pth"
TEXT_MODEL_PATH = MODELS_DIR / "text_classifier.joblib"
TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
CATEGORY_MAPPING_PATH = MODELS_DIR / "category_mapping.json"
IMPLEMENTATION_DIR = PROJECT_ROOT / "implementation"
FEATURES_DIR = IMPLEMENTATION_DIR / "outputs"
METADATA_PATH = FEATURES_DIR / "metadata_augmented.json"
STREAMLIT_DIR = Path(__file__).parent
ASSETS_DIR = STREAMLIT_DIR / "assets"
APP_CONFIG = {"title": "Rakuten Product Classifier", "icon": "🛒", "layout": "wide", "initial_sidebar_state": "expanded"}
FUSION_W_IMAGE = 0.6
FUSION_W_TEXT = 0.4
MODEL_CONFIG = {"use_mock": False, "fusion_weights": (FUSION_W_IMAGE, FUSION_W_TEXT), "top_k": 5, "confidence_threshold": 0.1}
IMAGE_CONFIG = {"target_size": (224, 224), "allowed_formats": ["jpg", "jpeg", "png", "webp"], "max_size_mb": 10}
THEME = {
    "primary_color": "#BF0000", "secondary_color": "#FFFFFF", "background_color": "#F5F5F5",
    "text_color": "#333333", "success_color": "#28A745", "warning_color": "#FFC107", "error_color": "#DC3545"
}
