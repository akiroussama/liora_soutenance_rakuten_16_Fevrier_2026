"""
Utilitaires pour l'application Streamlit Rakuten.
"""
from .model_interface import BaseClassifier, ClassificationResult
from .mock_classifier import MockClassifier, DemoClassifier
from .category_mapping import (
    CATEGORY_NAMES,
    CATEGORY_CODES,
    CATEGORY_MAPPING,
    get_category_name,
    get_category_info,
    get_category_emoji,
    get_all_categories,
)
from .image_utils import (
    load_image_from_upload,
    validate_image,
    resize_image,
    preprocess_for_resnet,
    create_thumbnail,
    get_image_info,
)
from .preprocessing import (
    clean_text,
    preprocess_product_text,
    validate_text_input,
    detect_language_simple,
)
