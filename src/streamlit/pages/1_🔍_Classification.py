"""
Page de classification de produits Rakuten.

Cette page permet à l'utilisateur de:
- Uploader une image de produit
- Saisir la désignation et description du produit
- Obtenir une prédiction de catégorie avec scores de confiance
- Visualiser les top-K prédictions
"""
import streamlit as st
import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import APP_CONFIG, MODEL_CONFIG, THEME
from utils.category_mapping import get_category_info, get_category_emoji
from utils.mock_classifier import DemoClassifier
from utils.image_utils import load_image_from_upload, validate_image, get_image_info
from utils.preprocessing import preprocess_product_text, validate_text_input

# =============================================================================
# Configuration de la page
# =============================================================================
st.set_page_config(
    page_title=f"Classification - {APP_CONFIG['title']}",
    page_icon="🔍",
    layout=APP_CONFIG["layout"],
)

# =============================================================================
# CSS personnalisé
# =============================================================================
st.markdown(f"""
<style>
    .result-card {{
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }}
    .confidence-high {{
        color: {THEME["success_color"]};
        font-weight: bold;
    }}
    .confidence-medium {{
        color: {THEME["warning_color"]};
        font-weight: bold;
    }}
    .confidence-low {{
        color: {THEME["error_color"]};
    }}
    .category-badge {{
        display: inline-block;
        background-color: {THEME["primary_color"]};
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 1.2rem;
        font-weight: bold;
    }}
    .prediction-bar {{
        height: 25px;
        border-radius: 5px;
        margin-bottom: 8px;
    }}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Initialisation
# =============================================================================
def init_page_state():
    """Initialise l'état de la page."""
    if "classifier" not in st.session_state:
        st.session_state.classifier = DemoClassifier()

    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None

    if "uploaded_image" not in st.session_state:
        st.session_state.uploaded_image = None


init_page_state()


# =============================================================================
# Fonctions d'affichage
# =============================================================================
def display_prediction_result(result, image=None):
    """
    Affiche le résultat de classification de manière visuelle.

    Args:
        result: ClassificationResult avec les prédictions
        image: Image PIL optionnelle à afficher
    """
    # Informations sur la catégorie principale
    main_category = result.category
    main_confidence = result.confidence
    name, full_name, emoji = get_category_info(main_category)

    # Layout en deux colonnes
    col_img, col_result = st.columns([1, 2])

    with col_img:
        if image is not None:
            st.image(image, caption="Image analysée", use_container_width=True)

    with col_result:
        # Badge de la catégorie principale
        st.markdown(f"""
        <div style='text-align: center; margin-bottom: 1rem;'>
            <span style='font-size: 3rem;'>{emoji}</span>
            <h2 style='margin: 0.5rem 0; color: {THEME["primary_color"]};'>{name}</h2>
            <p style='color: #666;'>{full_name}</p>
            <p><strong>Code:</strong> {main_category}</p>
        </div>
        """, unsafe_allow_html=True)

        # Score de confiance principal
        confidence_pct = main_confidence * 100
        if confidence_pct >= 70:
            confidence_class = "confidence-high"
            confidence_color = THEME["success_color"]
        elif confidence_pct >= 40:
            confidence_class = "confidence-medium"
            confidence_color = THEME["warning_color"]
        else:
            confidence_class = "confidence-low"
            confidence_color = THEME["error_color"]

        st.markdown(f"""
        <div style='text-align: center;'>
            <p style='font-size: 0.9rem; color: #888;'>Confiance</p>
            <p class='{confidence_class}' style='font-size: 2rem;'>{confidence_pct:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

        # Source de la prédiction
        source_labels = {
            "mock_image": "🖼️ Image (Mock)",
            "mock_text": "📝 Texte (Mock)",
            "mock_multimodal": "🔀 Multimodal (Mock)",
            "demo": "🎯 Démonstration",
            "image": "🖼️ Image",
            "text": "📝 Texte",
            "multimodal": "🔀 Multimodal",
        }
        source_label = source_labels.get(result.source, result.source)
        st.info(f"**Source:** {source_label}")

    # Top-K prédictions avec barres de progression
    st.markdown("---")
    st.markdown("### 📊 Top 5 des Prédictions")

    for i, (cat_code, score) in enumerate(result.top_k_predictions[:5]):
        cat_name, _, cat_emoji = get_category_info(cat_code)
        score_pct = score * 100

        # Déterminer la couleur de la barre
        if i == 0:
            bar_color = THEME["primary_color"]
        elif score_pct >= 20:
            bar_color = THEME["success_color"]
        else:
            bar_color = "#CCC"

        # Afficher la prédiction
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            st.markdown(f"{cat_emoji} **{cat_name}**")
        with col2:
            st.progress(score, text=None)
        with col3:
            st.markdown(f"**{score_pct:.1f}%**")


def display_input_summary(designation, description, image):
    """Affiche un résumé des entrées utilisateur."""
    with st.expander("📋 Résumé des entrées", expanded=False):
        if image is not None:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(image, width=150)
            with col2:
                info = get_image_info(image)
                st.markdown(f"""
                **Image:**
                - Dimensions: {info['size_str']}
                - Mode: {info['mode']}
                """)

        if designation:
            st.markdown(f"**Désignation:** {designation}")

        if description:
            st.markdown(f"**Description:** {description[:200]}{'...' if len(description) > 200 else ''}")


# =============================================================================
# Interface principale
# =============================================================================
st.title("🔍 Classification de Produits")

st.markdown("""
Uploadez une image et/ou saisissez la description d'un produit pour obtenir
sa catégorie prédite parmi les **27 catégories** Rakuten.
""")

# Indicateur de mode
if "use_mock" in st.session_state and st.session_state.use_mock:
    st.warning("⚠️ **Mode Démonstration** - Les prédictions sont simulées pour tester l'interface.")

# =============================================================================
# Formulaire d'entrée
# =============================================================================
st.markdown("### 📤 Entrées du Produit")

# Tabs pour les différentes méthodes d'entrée
tab_combined, tab_image, tab_text = st.tabs([
    "🔀 Multimodal (Recommandé)",
    "🖼️ Image seule",
    "📝 Texte seul"
])

with tab_combined:
    col_upload, col_text = st.columns(2)

    with col_upload:
        st.markdown("#### 🖼️ Image du Produit")
        uploaded_file = st.file_uploader(
            "Glissez-déposez ou sélectionnez une image",
            type=["jpg", "jpeg", "png", "webp"],
            key="upload_combined",
            help="Formats acceptés: JPG, PNG, WEBP. Taille max: 10 MB"
        )

        image = None
        if uploaded_file is not None:
            try:
                image = load_image_from_upload(uploaded_file)
                is_valid, message = validate_image(image)
                if is_valid:
                    st.image(image, caption="Aperçu", use_container_width=True)
                else:
                    st.error(message)
                    image = None
            except ValueError as e:
                st.error(str(e))
                image = None

    with col_text:
        st.markdown("#### 📝 Description du Produit")
        designation = st.text_input(
            "Désignation (titre du produit)",
            placeholder="Ex: Livre Harry Potter à l'école des sorciers",
            key="designation_combined",
            help="Le titre ou nom du produit"
        )

        description = st.text_area(
            "Description (optionnel)",
            placeholder="Ex: Roman fantastique pour enfants, édition française...",
            height=150,
            key="description_combined",
            help="Description détaillée du produit"
        )

    # Bouton de classification
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        classify_button = st.button(
            "🚀 Classifier le Produit",
            use_container_width=True,
            type="primary",
            key="classify_combined"
        )

    if classify_button:
        # Validation des entrées
        has_image = image is not None
        has_text = bool(designation and designation.strip())

        if not has_image and not has_text:
            st.error("❌ Veuillez fournir au moins une image ou une désignation.")
        else:
            with st.spinner("🔄 Classification en cours..."):
                # Prétraiter le texte si présent
                processed_text = None
                if has_text:
                    processed_text = preprocess_product_text(designation, description)

                # Effectuer la prédiction
                try:
                    result = st.session_state.classifier.predict(
                        image=image,
                        text=processed_text,
                        top_k=5
                    )

                    # Stocker le résultat
                    st.session_state.last_prediction = result

                    # Afficher les résultats
                    st.markdown("---")
                    st.markdown("## 🎯 Résultat de la Classification")
                    display_prediction_result(result, image)

                    # Résumé des entrées
                    display_input_summary(designation, description, image)

                except Exception as e:
                    st.error(f"❌ Erreur lors de la classification: {e}")

with tab_image:
    st.markdown("#### 🖼️ Classification par Image Seule")
    uploaded_file_img = st.file_uploader(
        "Sélectionnez une image de produit",
        type=["jpg", "jpeg", "png", "webp"],
        key="upload_image_only",
    )

    if uploaded_file_img is not None:
        try:
            image_only = load_image_from_upload(uploaded_file_img)
            is_valid, message = validate_image(image_only)
            if is_valid:
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image_only, caption="Image à classifier", use_container_width=True)
                with col2:
                    if st.button("🚀 Classifier", key="classify_image_only", use_container_width=True):
                        with st.spinner("Classification..."):
                            result = st.session_state.classifier.predict(image=image_only, top_k=5)
                            st.session_state.last_prediction = result

                if st.session_state.last_prediction and uploaded_file_img:
                    st.markdown("---")
                    display_prediction_result(st.session_state.last_prediction, image_only)
            else:
                st.error(message)
        except ValueError as e:
            st.error(str(e))

with tab_text:
    st.markdown("#### 📝 Classification par Texte Seul")

    designation_text = st.text_input(
        "Désignation du produit",
        placeholder="Ex: Console PlayStation 5",
        key="designation_text_only",
    )

    description_text = st.text_area(
        "Description (optionnel)",
        placeholder="Ex: Console de jeux vidéo nouvelle génération...",
        height=100,
        key="description_text_only",
    )

    if st.button("🚀 Classifier", key="classify_text_only", use_container_width=True):
        if not designation_text or not designation_text.strip():
            st.error("❌ Veuillez saisir au moins la désignation du produit.")
        else:
            with st.spinner("Classification..."):
                processed_text = preprocess_product_text(designation_text, description_text)
                result = st.session_state.classifier.predict(text=processed_text, top_k=5)
                st.session_state.last_prediction = result

            st.markdown("---")
            display_prediction_result(result)


# =============================================================================
# Section d'exemples
# =============================================================================
st.markdown("---")
st.markdown("### 💡 Exemples à Essayer")

example_col1, example_col2, example_col3 = st.columns(3)

with example_col1:
    st.markdown("**📚 Livre**")
    if st.button("Essayer 'Livre Harry Potter'", key="example_book"):
        with st.spinner("Classification..."):
            result = st.session_state.classifier.predict(
                text="livre harry potter à l'école des sorciers roman fantastique",
                top_k=5
            )
        st.session_state.last_prediction = result
        display_prediction_result(result)

with example_col2:
    st.markdown("**🎮 Jeux Vidéo**")
    if st.button("Essayer 'Console PlayStation'", key="example_console"):
        with st.spinner("Classification..."):
            result = st.session_state.classifier.predict(
                text="console playstation 5 jeux vidéo nouvelle génération",
                top_k=5
            )
        st.session_state.last_prediction = result
        display_prediction_result(result)

with example_col3:
    st.markdown("**🏊 Piscine**")
    if st.button("Essayer 'Piscine gonflable'", key="example_pool"):
        with st.spinner("Classification..."):
            result = st.session_state.classifier.predict(
                text="piscine gonflable ronde pour jardin été enfants",
                top_k=5
            )
        st.session_state.last_prediction = result
        display_prediction_result(result)


# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    st.markdown("### 🔍 Classification")
    st.markdown("---")

    # Statistiques de session
    st.markdown("#### 📊 Session")
    if "classification_history" not in st.session_state:
        st.session_state.classification_history = []

    st.metric("Classifications", len(st.session_state.classification_history))

    # Bouton de réinitialisation
    if st.button("🔄 Réinitialiser", use_container_width=True):
        st.session_state.last_prediction = None
        st.session_state.classification_history = []
        st.rerun()

    st.markdown("---")

    # Informations sur le mode
    st.markdown("#### ℹ️ Mode Actuel")
    if st.session_state.get("use_mock", True):
        st.info("**Démonstration**\n\nPrédictions simulées")
    else:
        st.success("**Production**\n\nModèles ML actifs")
