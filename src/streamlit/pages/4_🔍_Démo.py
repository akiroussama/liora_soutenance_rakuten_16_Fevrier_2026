"""
Page de classification de produits Rakuten.

Cette page permet à l'utilisateur de:
- Uploader une image de produit
- Saisir la désignation et description du produit
- Obtenir une prédiction de catégorie avec scores de confiance
- Visualiser les top-K prédictions
- Historique des classifications de la session
- Galerie d'exemples pré-définis
- Mode comparaison des modalités
"""
import streamlit as st
import plotly.graph_objects as go
import sys
from pathlib import Path
from datetime import datetime

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import APP_CONFIG, MODEL_CONFIG, THEME, ASSETS_DIR
from utils.category_mapping import get_category_info, get_category_emoji
from utils.mock_classifier import (
    DemoClassifier,
    TEXT_MODELS,
    IMAGE_MODELS,
    get_available_text_models,
    get_available_image_models,
)
from utils.image_utils import load_image_from_upload, validate_image, get_image_info
from utils.preprocessing import preprocess_product_text, validate_text_input
from utils.ui_utils import load_css

# =============================================================================
# Configuration de la page
# =============================================================================
st.set_page_config(
    page_title=f"Démo - {APP_CONFIG['title']}",
    page_icon="🔍",
    layout=APP_CONFIG["layout"],
)

# =============================================================================
# CSS personnalisé
# =============================================================================
load_css(ASSETS_DIR / "style.css")


# =============================================================================
# Initialisation
# =============================================================================
def init_page_state():
    """Initialise l'état de la page."""
    # Sélection des modèles
    if "selected_text_model" not in st.session_state:
        st.session_state.selected_text_model = "camembert"  # Meilleur modèle par défaut

    if "selected_image_model" not in st.session_state:
        st.session_state.selected_image_model = "resnet50_svm"  # Meilleur modèle par défaut

    # Créer les classifieurs avec les modèles sélectionnés
    if "text_classifier" not in st.session_state:
        text_config = TEXT_MODELS[st.session_state.selected_text_model]
        st.session_state.text_classifier = DemoClassifier(model_config=text_config)

    if "image_classifier" not in st.session_state:
        image_config = IMAGE_MODELS[st.session_state.selected_image_model]
        st.session_state.image_classifier = DemoClassifier(model_config=image_config)

    # Classifieur par défaut (pour compatibilité)
    if "classifier" not in st.session_state:
        st.session_state.classifier = DemoClassifier()

    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None

    if "uploaded_image" not in st.session_state:
        st.session_state.uploaded_image = None

    # État pour chaque onglet
    if "image_only_result" not in st.session_state:
        st.session_state.image_only_result = None

    if "text_only_result" not in st.session_state:
        st.session_state.text_only_result = None

    # Historique des classifications
    if "classification_history" not in st.session_state:
        st.session_state.classification_history = []

    # État pour le mode comparaison
    if "comparison_results" not in st.session_state:
        st.session_state.comparison_results = None


def update_classifiers():
    """Met à jour les classifieurs après changement de modèle."""
    text_config = TEXT_MODELS[st.session_state.selected_text_model]
    st.session_state.text_classifier = DemoClassifier(model_config=text_config)

    image_config = IMAGE_MODELS[st.session_state.selected_image_model]
    st.session_state.image_classifier = DemoClassifier(model_config=image_config)


def add_to_history(result, designation="", description="", has_image=False, source_tab="multimodal"):
    """Ajoute une classification à l'historique."""
    cat_name, cat_full, cat_emoji = get_category_info(result.category)
    history_entry = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "category": result.category,
        "category_name": cat_name,
        "emoji": cat_emoji,
        "confidence": result.confidence,
        "designation": designation[:50] + "..." if len(designation) > 50 else designation,
        "has_image": has_image,
        "source": source_tab,
    }
    st.session_state.classification_history.insert(0, history_entry)
    # Garder les 10 dernières classifications
    st.session_state.classification_history = st.session_state.classification_history[:10]


init_page_state()


# =============================================================================
# Galerie d'exemples pré-définis
# =============================================================================
EXAMPLE_PRODUCTS = [
    {
        "id": "book",
        "emoji": "📚",
        "name": "Livre",
        "designation": "Harry Potter à l'école des sorciers",
        "description": "Roman fantastique pour enfants de J.K. Rowling, édition française",
        "expected_category": "2403",
    },
    {
        "id": "console",
        "emoji": "🎮",
        "name": "Console",
        "designation": "Console PlayStation 5",
        "description": "Console de jeux vidéo nouvelle génération Sony, 825 Go SSD",
        "expected_category": "2462",
    },
    {
        "id": "pool",
        "emoji": "🏊",
        "name": "Piscine",
        "designation": "Piscine gonflable ronde",
        "description": "Piscine autoportante pour jardin, été, enfants et adultes, 3m diamètre",
        "expected_category": "2582",
    },
    {
        "id": "figurine",
        "emoji": "🦸",
        "name": "Figurine",
        "designation": "Figurine Pop Marvel Spider-Man",
        "description": "Figurine de collection Funko Pop, vinyle, 10cm, super héros",
        "expected_category": "1281",
    },
    {
        "id": "phone",
        "emoji": "📱",
        "name": "Téléphone",
        "designation": "iPhone 15 Pro Max 256GB",
        "description": "Smartphone Apple dernière génération, écran OLED, 5G",
        "expected_category": "2583",
    },
    {
        "id": "toy",
        "emoji": "🧸",
        "name": "Jouet",
        "designation": "LEGO Star Wars Millennium Falcon",
        "description": "Jeu de construction LEGO, vaisseau spatial, 1351 pièces",
        "expected_category": "1280",
    },
    {
        "id": "clothes",
        "emoji": "👗",
        "name": "Vêtement",
        "designation": "Robe d'été fleurie femme",
        "description": "Robe légère en coton, motif floral, taille M, rouge",
        "expected_category": "1920",
    },
    {
        "id": "garden",
        "emoji": "🌱",
        "name": "Jardin",
        "designation": "Tondeuse à gazon électrique",
        "description": "Tondeuse électrique Bosch 1400W, coupe 37cm, bac 40L",
        "expected_category": "2585",
    },
    {
        "id": "makeup",
        "emoji": "💄",
        "name": "Maquillage",
        "designation": "Palette de maquillage Urban Decay",
        "description": "Palette fards à paupières Naked, 12 teintes, beauté",
        "expected_category": "1301",
    },
]


# =============================================================================
# CSS pour les résultats WOW
# =============================================================================
st.markdown("""
<style>
/* Rakuten official top bar */
.rakuten-top-bar {
    background-color: #BF0000;
    height: 4px;
    width: 100%;
    position: fixed;
    top: 0;
    left: 0;
    z-index: 1000;
}

/* Header container */
.rakuten-header {
    background: white;
    padding: 1.2rem 0;
    border-bottom: 1px solid #f0f0f0;
    display: flex;
    align-items: center;
    border-radius: 12px;
    margin-bottom: 1.5rem;
}

.rakuten-logo-container {
    padding-left: 2rem;
    display: flex;
    align-items: center;
}

.rakuten-logo {
    height: 28px;
}

.app-title {
    color: #333;
    font-size: 1.2rem;
    font-weight: 600;
    margin-left: 1.2rem;
    border-left: 1.5px solid #eaeaea;
    padding-left: 1.2rem;
    font-family: 'Inter', sans-serif;
}

/* Result card refined - PRO & Sobere */
.result-card {
    background: white;
    padding: 2.5rem;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
    margin-bottom: 2rem;
    border-top: 4px solid #BF0000;
    position: relative;
    overflow: hidden;
}

.result-emoji {
    font-size: 3.5rem;
    display: block;
    margin-bottom: 1rem;
}

.result-category {
    color: #333 !important;
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    margin: 0.5rem 0 !important;
    font-family: 'Inter', sans-serif;
}

.result-description {
    color: #666 !important;
    font-size: 1.1rem !important;
    margin-bottom: 1.5rem !important;
    font-weight: 300;
}

.result-code {
    display: inline-block;
    background: #f8f8f8;
    color: #888 !important;
    padding: 0.4rem 1.2rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 500;
    border: 1px solid #eee;
}

/* Confidence gauge */
.confidence-container {
    background: white;
    padding: 1.5rem;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    text-align: center;
}

.confidence-value {
    font-size: 3rem !important;
    font-weight: 800 !important;
    margin: 0.5rem 0 !important;
}

.confidence-label {
    color: #666 !important;
    font-size: 0.9rem !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.confidence-bar-bg {
    width: 100%;
    height: 12px;
    background: #E0E0E0;
    border-radius: 6px;
    margin-top: 1rem;
    overflow: hidden;
}

.confidence-bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.5s ease;
}

/* Alternative predictions */
.alt-prediction {
    background: white;
    padding: 1rem 1.2rem;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    display: flex;
    align-items: center;
    margin-bottom: 0.8rem;
    transition: all 0.2s ease;
    border-left: 4px solid #E0E0E0;
}

.alt-prediction:hover {
    transform: translateX(5px);
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}

.alt-prediction.rank-1 {
    border-left-color: #BF0000;
    background: linear-gradient(90deg, #FFF5F5 0%, white 100%);
}

.alt-prediction.rank-2 {
    border-left-color: #FF6B6B;
}

.alt-prediction.rank-3 {
    border-left-color: #FFB4B4;
}

.alt-emoji {
    font-size: 1.8rem;
    margin-right: 1rem;
}

.alt-info {
    flex-grow: 1;
}

.alt-name {
    color: #333 !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    margin: 0 !important;
}

.alt-score {
    color: #666 !important;
    font-size: 0.85rem !important;
}

.alt-percent {
    font-size: 1.3rem;
    font-weight: 700;
    color: #BF0000;
}

/* Source badge */
.source-badge {
    display: inline-block;
    background: #F0F0F0;
    color: #333;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-size: 0.85rem;
    margin-top: 0.5rem;
}

/* Image preview */
.image-preview-container {
    background: white;
    padding: 1rem;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}

/* =========================================== */
/* EXAMPLE GALLERY STYLES */
/* =========================================== */
.example-card {
    background: white;
    padding: 1.2rem;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    cursor: pointer;
    transition: all 0.3s ease;
    border: 2px solid transparent;
    min-height: 140px;
}

.example-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 25px rgba(191,0,0,0.2);
    border-color: #BF0000;
}

.example-emoji {
    font-size: 2.5rem;
    display: block;
    margin-bottom: 0.5rem;
}

.example-name {
    color: #333 !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    margin: 0.3rem 0 !important;
}

.example-desc {
    color: #888 !important;
    font-size: 0.75rem !important;
    line-height: 1.3;
}

/* =========================================== */
/* HISTORY STYLES */
/* =========================================== */
.history-card {
    background: white;
    padding: 0.8rem 1rem;
    border-radius: 10px;
    margin-bottom: 0.6rem;
    border-left: 4px solid #BF0000;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    transition: all 0.2s ease;
}

.history-card:hover {
    transform: translateX(3px);
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.history-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.history-emoji {
    font-size: 1.5rem;
    margin-right: 0.8rem;
}

.history-category {
    color: #333 !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    flex-grow: 1;
}

.history-confidence {
    font-size: 0.85rem;
    font-weight: 700;
    color: #BF0000;
}

.history-time {
    color: #999 !important;
    font-size: 0.7rem !important;
    margin-top: 0.3rem;
}

.history-source {
    display: inline-block;
    background: #F5F5F5;
    color: #666;
    padding: 0.15rem 0.5rem;
    border-radius: 10px;
    font-size: 0.65rem;
    margin-left: 0.5rem;
}

/* =========================================== */
/* COMPARISON MODE STYLES */
/* =========================================== */
.comparison-card {
    background: white;
    padding: 1.5rem;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    text-align: center;
    border-top: 4px solid #BF0000;
}

.comparison-title {
    color: #BF0000 !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    margin-bottom: 1rem !important;
}

.comparison-result {
    background: linear-gradient(135deg, #FFF5F5 0%, white 100%);
    padding: 1rem;
    border-radius: 10px;
    margin-top: 1rem;
}

.comparison-emoji {
    font-size: 2rem;
    display: block;
    margin-bottom: 0.5rem;
}

.comparison-category {
    color: #333 !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
}

.comparison-conf {
    font-size: 1.5rem;
    font-weight: 800;
    color: #BF0000;
    margin-top: 0.5rem;
}

.comparison-winner {
    border: 3px solid #28A745 !important;
    box-shadow: 0 4px 20px rgba(40, 167, 69, 0.3) !important;
}

.winner-badge {
    background: #28A745;
    color: white;
    padding: 0.3rem 0.8rem;
    border-radius: 15px;
    font-size: 0.75rem;
    font-weight: 600;
    display: inline-block;
    margin-bottom: 0.5rem;
}

/* Gallery grid */
.gallery-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
}

@media (max-width: 768px) {
    .gallery-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Fonctions d'affichage
# =============================================================================
def display_prediction_result(result, image=None):
    """
    Affiche le résultat de classification avec effet WOW.
    """
    main_category = result.category
    main_confidence = result.confidence
    name, full_name, emoji = get_category_info(main_category)
    confidence_pct = main_confidence * 100

    # Déterminer couleur confiance
    if confidence_pct >= 70:
        conf_color = "#28A745"
        conf_text = "Haute"
    elif confidence_pct >= 40:
        conf_color = "#FF9800"
        conf_text = "Moyenne"
    else:
        conf_color = "#DC3545"
        conf_text = "Faible"

    # Source label
    source_labels = {
        "mock_image": "🖼️ Image",
        "mock_text": "📝 Texte",
        "mock_multimodal": "🔀 Multimodal",
        "demo": "🎯 Démo",
        "image": "🖼️ Image",
        "text": "📝 Texte",
        "multimodal": "🔀 Multimodal",
    }
    source_label = source_labels.get(result.source, result.source)

    # =========================================================================
    # RESULT CARD refined
    # =========================================================================
    st.markdown(f"""
    <div class="result-card">
        <span class="result-emoji">{emoji}</span>
        <h2 class="result-category">{name}</h2>
        <p class="result-description">{full_name}</p>
        <span class="result-code">Code catégorie : {main_category}</span>
    </div>
    """, unsafe_allow_html=True)

    # =========================================================================
    # CONFIDENCE + IMAGE ROW
    # =========================================================================
    if image is not None:
        col_conf, col_img = st.columns([1, 1])
    else:
        col_conf, col_img = st.columns([2, 1])

    with col_conf:
        st.markdown(f"""
        <div class="confidence-container">
            <p class="confidence-label">Niveau de confiance</p>
            <p class="confidence-value" style="color: {conf_color};">{confidence_pct:.1f}%</p>
            <div class="confidence-bar-bg">
                <div class="confidence-bar-fill" style="width: {confidence_pct}%; background: {conf_color};"></div>
            </div>
            <p style="color: #888; font-size: 0.85rem; margin-top: 0.8rem;">Confiance {conf_text.lower()}</p>
            <span class="source-badge">{source_label}</span>
        </div>
        """, unsafe_allow_html=True)

    with col_img:
        if image is not None:
            st.markdown('<div class="image-preview-container">', unsafe_allow_html=True)
            st.image(image, caption="Image analysée", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =========================================================================
    # TOP 5 ALTERNATIVES
    # =========================================================================
    st.markdown("### 🏆 Classement des Prédictions")

    col_list, col_chart = st.columns([1, 1])

    with col_list:
        for i, (cat_code, score) in enumerate(result.top_k_predictions[:5]):
            cat_name, cat_full, cat_emoji = get_category_info(cat_code)
            score_pct = score * 100
            rank_class = f"rank-{i+1}" if i < 3 else ""

            st.markdown(f"""
            <div class="alt-prediction {rank_class}">
                <span class="alt-emoji">{cat_emoji}</span>
                <div class="alt-info">
                    <p class="alt-name">{cat_name}</p>
                    <p class="alt-score">{cat_full}</p>
                </div>
                <span class="alt-percent">{score_pct:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

    with col_chart:
        # Graphique Plotly
        top_5 = result.top_k_predictions[:5]
        categories = []
        scores = []
        colors = []

        for i, (cat_code, score) in enumerate(reversed(top_5)):
            cat_name, _, cat_emoji = get_category_info(cat_code)
            categories.append(f"{cat_emoji} {cat_name}")
            scores.append(score * 100)

            if i == len(top_5) - 1:
                colors.append("#BF0000")
            elif i >= len(top_5) - 2:
                colors.append("#FF6B6B")
            else:
                colors.append("#CCCCCC")

        fig = go.Figure(go.Bar(
            x=scores,
            y=categories,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='#990000', width=1)
            ),
            text=[f"{s:.1f}%" for s in scores],
            textposition='outside',
            textfont=dict(color='#333333', size=12)
        ))

        fig.update_layout(
            height=300,
            margin=dict(l=0, r=60, t=10, b=10),
            xaxis=dict(
                title="Confiance (%)",
                range=[0, max(scores) * 1.25],
                showgrid=True,
                gridcolor='#E0E0E0',
                tickfont=dict(color='#333333')
            ),
            yaxis=dict(
                tickfont=dict(color='#333333', size=11)
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )

        st.plotly_chart(fig, use_container_width=True)


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
# =============================================================================
# HEADER & BREADCRUMBS
# =============================================================================
st.markdown("""
<div class="rakuten-top-bar"></div>
<div class="rakuten-header">
    <div class="rakuten-logo-container">
        <img src="https://fr.shopping.rakuten.com/visuels/0_content_square/autres/rakuten-logo7.svg" class="rakuten-logo">
        <span class="app-title">Product Classifier</span>
    </div>
</div>
""", unsafe_allow_html=True)

col_title, col_empty = st.columns([2, 1])
with col_title:
    st.markdown("## 🔍 Classification de Produits")
    st.markdown("""
    <p style="color: #666; font-size: 1.1rem; font-weight: 300; margin-bottom: 2rem;">
        Identifiez instantanément la catégorie Rakuten de vos produits grâce à notre intelligence artificielle.
    </p>
    """, unsafe_allow_html=True)

# Indicateur de mode
if "use_mock" in st.session_state and st.session_state.use_mock:
    st.warning("⚠️ **Mode Démonstration** - Les prédictions sont simulées pour tester l'interface.")

# =============================================================================
# Formulaire d'entrée
# =============================================================================
st.markdown("### 📤 Entrées du Produit")

# Tabs pour les différentes méthodes d'entrée
tab_combined, tab_image, tab_text, tab_compare, tab_gallery = st.tabs([
    "🔀 Multimodal",
    "🖼️ Image seule",
    "📝 Texte seul",
    "⚔️ Comparaison",
    "🎯 Galerie d'exemples"
])

with tab_combined:
    st.markdown("""
    Fournissez **texte OU image** (ou les deux). Le système utilisera le modèle approprié
    selon les données fournies.
    """)

    # Afficher les modèles sélectionnés
    col_models1, col_models2 = st.columns(2)
    with col_models1:
        txt_cfg = TEXT_MODELS[st.session_state.selected_text_model]
        st.markdown(f"📝 **Texte**: {txt_cfg.name}")
    with col_models2:
        img_cfg = IMAGE_MODELS[st.session_state.selected_image_model]
        st.markdown(f"🖼️ **Image**: {img_cfg.name}")

    st.markdown("---")

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
            # Déterminer quel(s) modèle(s) utiliser
            results_to_show = []

            try:
                # Classification par texte si texte fourni
                if has_text:
                    with st.spinner(f"🔄 Classification texte avec {txt_cfg.short_name}..."):
                        processed_text = preprocess_product_text(designation, description)
                        text_result = st.session_state.text_classifier.predict(text=processed_text, top_k=5)
                        results_to_show.append(("texte", text_result, txt_cfg))
                        add_to_history(text_result, designation, description, False, "texte")

                # Classification par image si image fournie
                if has_image:
                    with st.spinner(f"🔄 Classification image avec {img_cfg.short_name}..."):
                        image_result = st.session_state.image_classifier.predict(image=image, top_k=5)
                        results_to_show.append(("image", image_result, img_cfg))
                        add_to_history(image_result, "Image", "", True, "image")

                # Afficher les résultats
                st.markdown("---")

                if len(results_to_show) == 2:
                    # Afficher les deux résultats côte à côte
                    st.markdown("## 🎯 Résultats de Classification")
                    st.info("📊 Deux modèles ont été utilisés : un pour le texte, un pour l'image")

                    col_res1, col_res2 = st.columns(2)

                    with col_res1:
                        st.markdown(f"### 📝 Modèle Texte")
                        st.caption(f"*{txt_cfg.name}*")
                        display_prediction_result(results_to_show[0][1], None)

                    with col_res2:
                        st.markdown(f"### 🖼️ Modèle Image")
                        st.caption(f"*{img_cfg.name}*")
                        display_prediction_result(results_to_show[1][1], image)

                else:
                    # Un seul résultat
                    mode, result, config = results_to_show[0]
                    st.markdown(f"## 🎯 Résultat ({config.name})")
                    st.session_state.last_prediction = result
                    display_prediction_result(result, image if mode == "image" else None)

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
                    # Afficher le modèle utilisé
                    img_config = IMAGE_MODELS[st.session_state.selected_image_model]
                    st.info(f"🧠 Modèle: **{img_config.name}**")

                    if st.button("🚀 Classifier", key="classify_image_only", use_container_width=True, type="primary"):
                        with st.spinner(f"Classification avec {img_config.short_name}..."):
                            result = st.session_state.image_classifier.predict(image=image_only, top_k=5)
                            st.session_state.image_only_result = result
                            st.session_state.image_only_image = image_only
                            add_to_history(result, "Image uploadée", "", True, "image")

                # Afficher le résultat s'il existe pour cet onglet
                if "image_only_result" in st.session_state and st.session_state.image_only_result is not None:
                    st.markdown("---")
                    display_prediction_result(st.session_state.image_only_result, st.session_state.get("image_only_image"))
            else:
                st.error(message)
        except ValueError as e:
            st.error(str(e))

with tab_text:
    st.markdown("#### 📝 Classification par Texte Seul")

    # Afficher le modèle utilisé
    txt_config = TEXT_MODELS[st.session_state.selected_text_model]
    st.info(f"🧠 Modèle: **{txt_config.name}** - {txt_config.description}")

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

    if st.button("🚀 Classifier", key="classify_text_only", use_container_width=True, type="primary"):
        if not designation_text or not designation_text.strip():
            st.error("❌ Veuillez saisir au moins la désignation du produit.")
        else:
            with st.spinner(f"Classification avec {txt_config.short_name}..."):
                processed_text = preprocess_product_text(designation_text, description_text)
                result = st.session_state.text_classifier.predict(text=processed_text, top_k=5)
                st.session_state.text_only_result = result
                add_to_history(result, designation_text, description_text, False, "texte")

    # Afficher le résultat s'il existe pour cet onglet
    if "text_only_result" in st.session_state and st.session_state.text_only_result is not None:
        st.markdown("---")
        display_prediction_result(st.session_state.text_only_result)


# =============================================================================
# ONGLET COMPARAISON
# =============================================================================
with tab_compare:
    st.markdown("#### ⚔️ Mode Comparaison des Modalités")
    st.markdown("""
    Comparez les prédictions entre **Texte seul**, **Image seule** et **Multimodal**
    pour un même produit.
    """)

    # Input pour la comparaison
    col_compare_input, col_compare_image = st.columns(2)

    with col_compare_input:
        compare_designation = st.text_input(
            "Désignation du produit",
            placeholder="Ex: Console PlayStation 5",
            key="compare_designation"
        )
        compare_description = st.text_area(
            "Description",
            placeholder="Console de jeux vidéo...",
            height=100,
            key="compare_description"
        )

    with col_compare_image:
        st.markdown("**Image (optionnel)**")
        compare_file = st.file_uploader(
            "Ajouter une image pour comparer",
            type=["jpg", "jpeg", "png", "webp"],
            key="compare_upload"
        )
        compare_image = None
        if compare_file:
            compare_image = load_image_from_upload(compare_file)
            st.image(compare_image, caption="Image", use_container_width=True)

    # Bouton de comparaison
    if st.button("⚔️ Lancer la Comparaison", use_container_width=True, type="primary", key="btn_compare"):
        if not compare_designation or not compare_designation.strip():
            st.error("❌ Veuillez saisir au moins la désignation du produit.")
        else:
            with st.spinner("Comparaison en cours..."):
                processed_text = preprocess_product_text(compare_designation, compare_description)

                # Prédiction texte seul
                result_text = st.session_state.classifier.predict(text=processed_text, top_k=5)

                # Prédiction image seule (si image fournie)
                result_image = None
                if compare_image:
                    result_image = st.session_state.classifier.predict(image=compare_image, top_k=5)

                # Prédiction multimodale
                result_multi = st.session_state.classifier.predict(
                    image=compare_image,
                    text=processed_text,
                    top_k=5
                )

                st.session_state.comparison_results = {
                    "text": result_text,
                    "image": result_image,
                    "multimodal": result_multi
                }

    # Afficher les résultats de comparaison
    if st.session_state.comparison_results:
        st.markdown("---")
        st.markdown("### 📊 Résultats de la Comparaison")

        results = st.session_state.comparison_results

        # Trouver le gagnant
        confidences = []
        confidences.append(("text", results["text"].confidence))
        if results["image"]:
            confidences.append(("image", results["image"].confidence))
        confidences.append(("multimodal", results["multimodal"].confidence))

        winner = max(confidences, key=lambda x: x[1])[0]

        # Afficher les cartes de comparaison
        if results["image"]:
            cols = st.columns(3)
            modes = [
                ("📝 Texte", "text", results["text"]),
                ("🖼️ Image", "image", results["image"]),
                ("🔀 Multimodal", "multimodal", results["multimodal"]),
            ]
        else:
            cols = st.columns(2)
            modes = [
                ("📝 Texte", "text", results["text"]),
                ("🔀 Multimodal", "multimodal", results["multimodal"]),
            ]

        for col, (mode_name, mode_key, result) in zip(cols, modes):
            with col:
                is_winner = mode_key == winner
                cat_name, _, cat_emoji = get_category_info(result.category)
                conf_pct = result.confidence * 100

                winner_class = "comparison-winner" if is_winner else ""
                winner_badge = '<span class="winner-badge">🏆 MEILLEUR</span>' if is_winner else ""

                st.markdown(f"""
                <div class="comparison-card {winner_class}">
                    {winner_badge}
                    <h4 class="comparison-title">{mode_name}</h4>
                    <div class="comparison-result">
                        <span class="comparison-emoji">{cat_emoji}</span>
                        <p class="comparison-category">{cat_name}</p>
                        <p class="comparison-conf">{conf_pct:.1f}%</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Graphique de comparaison
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📈 Comparaison des Confiances")

        chart_data = []
        colors = []
        for mode_name, mode_key, result in modes:
            chart_data.append({
                "Mode": mode_name,
                "Confiance": result.confidence * 100
            })
            colors.append("#28A745" if mode_key == winner else "#BF0000")

        import pandas as pd
        df_compare = pd.DataFrame(chart_data)

        fig_compare = go.Figure(go.Bar(
            x=df_compare["Mode"],
            y=df_compare["Confiance"],
            marker_color=colors,
            text=[f"{c:.1f}%" for c in df_compare["Confiance"]],
            textposition="outside",
            textfont=dict(color="#333", size=14, weight="bold")
        ))

        fig_compare.update_layout(
            height=350,
            xaxis_title="",
            yaxis_title="Confiance (%)",
            yaxis=dict(range=[0, 110]),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#333333'),
            margin=dict(t=30)
        )

        st.plotly_chart(fig_compare, use_container_width=True)


# =============================================================================
# ONGLET GALERIE D'EXEMPLES
# =============================================================================
with tab_gallery:
    st.markdown("#### 🎯 Galerie d'Exemples")
    st.markdown("""
    Cliquez sur un exemple pour tester instantanément notre modèle de classification.
    Ces exemples couvrent diverses catégories de produits Rakuten.
    """)

    # Afficher la galerie en grille 3x3
    for row_start in range(0, len(EXAMPLE_PRODUCTS), 3):
        row_examples = EXAMPLE_PRODUCTS[row_start:row_start + 3]
        cols = st.columns(3)

        for col, example in zip(cols, row_examples):
            with col:
                st.markdown(f"""
                <div class="example-card">
                    <span class="example-emoji">{example['emoji']}</span>
                    <p class="example-name">{example['name']}</p>
                    <p class="example-desc">{example['designation'][:35]}...</p>
                </div>
                """, unsafe_allow_html=True)

                if st.button(
                    f"Tester {example['emoji']}",
                    key=f"gallery_{example['id']}",
                    use_container_width=True
                ):
                    with st.spinner("Classification..."):
                        full_text = f"{example['designation']} {example['description']}"
                        result = st.session_state.classifier.predict(text=full_text, top_k=5)
                        st.session_state.gallery_result = result
                        st.session_state.gallery_example = example
                        add_to_history(result, example['designation'], example['description'], False, "galerie")

    # Afficher le résultat de la galerie
    if "gallery_result" in st.session_state and st.session_state.gallery_result:
        st.markdown("---")
        example = st.session_state.get("gallery_example", {})
        st.markdown(f"### Résultat pour: {example.get('emoji', '')} {example.get('name', '')}")
        st.caption(f"*{example.get('designation', '')}*")
        display_prediction_result(st.session_state.gallery_result)


# =============================================================================
# Sidebar avec Sélection de Modèles et Historique
# =============================================================================
with st.sidebar:
    st.markdown("### 🔍 Classification")
    st.markdown("---")

    # =========================================================================
    # SÉLECTION DES MODÈLES
    # =========================================================================
    st.markdown("#### 🧠 Modèles")

    # Modèle texte
    text_model_options = {k: v.name for k, v in TEXT_MODELS.items()}
    selected_text = st.selectbox(
        "📝 Modèle Texte",
        options=list(text_model_options.keys()),
        format_func=lambda x: text_model_options[x],
        index=list(text_model_options.keys()).index(st.session_state.selected_text_model),
        key="sidebar_text_model"
    )

    # Modèle image
    image_model_options = {k: v.name for k, v in IMAGE_MODELS.items()}
    selected_image = st.selectbox(
        "🖼️ Modèle Image",
        options=list(image_model_options.keys()),
        format_func=lambda x: image_model_options[x],
        index=list(image_model_options.keys()).index(st.session_state.selected_image_model),
        key="sidebar_image_model"
    )

    # Mettre à jour si changement
    if selected_text != st.session_state.selected_text_model or selected_image != st.session_state.selected_image_model:
        st.session_state.selected_text_model = selected_text
        st.session_state.selected_image_model = selected_image
        update_classifiers()
        st.rerun()

    # Afficher les infos du modèle sélectionné
    text_config = TEXT_MODELS[st.session_state.selected_text_model]
    image_config = IMAGE_MODELS[st.session_state.selected_image_model]

    with st.expander("ℹ️ Détails des modèles", expanded=False):
        st.markdown(f"""
        **Texte: {text_config.name}**
        - {text_config.description}
        - Confiance moy.: {text_config.base_confidence*100:.0f}%

        **Image: {image_config.name}**
        - {image_config.description}
        - Confiance moy.: {image_config.base_confidence*100:.0f}%
        """)

    st.markdown("---")

    # =========================================================================
    # STATISTIQUES DE SESSION
    # =========================================================================
    st.markdown("#### 📊 Session")
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.metric("Total", len(st.session_state.classification_history))
    with col_stat2:
        if st.session_state.classification_history:
            avg_conf = sum(h["confidence"] for h in st.session_state.classification_history) / len(st.session_state.classification_history)
            st.metric("Conf. moy.", f"{avg_conf*100:.0f}%")
        else:
            st.metric("Conf. moy.", "-")

    # Bouton de réinitialisation
    if st.button("🔄 Réinitialiser", use_container_width=True):
        st.session_state.last_prediction = None
        st.session_state.image_only_result = None
        st.session_state.text_only_result = None
        st.session_state.classification_history = []
        st.session_state.comparison_results = None
        if "gallery_result" in st.session_state:
            del st.session_state.gallery_result
        st.rerun()

    st.markdown("---")

    # =========================================================================
    # HISTORIQUE
    # =========================================================================
    st.markdown("#### 🕐 Historique")

    if st.session_state.classification_history:
        for entry in st.session_state.classification_history[:5]:
            source_icons = {"multimodal": "🔀", "image": "🖼️", "texte": "📝", "galerie": "🎯"}
            source_icon = source_icons.get(entry["source"], "📋")
            conf_color = "#28A745" if entry["confidence"] >= 0.7 else "#FF9800" if entry["confidence"] >= 0.4 else "#DC3545"

            st.markdown(f"""
            <div class="history-card">
                <div class="history-header">
                    <span class="history-emoji">{entry['emoji']}</span>
                    <span class="history-category">{entry['category_name']}</span>
                    <span class="history-confidence" style="color: {conf_color};">{entry['confidence']*100:.0f}%</span>
                </div>
                <p class="history-time">{entry['timestamp']} <span class="history-source">{source_icon} {entry['source']}</span></p>
            </div>
            """, unsafe_allow_html=True)

        if len(st.session_state.classification_history) > 5:
            st.caption(f"... et {len(st.session_state.classification_history) - 5} autres")
    else:
        st.info("Aucune classification")

    st.markdown("---")

    # =========================================================================
    # LIEN VERS COMPARAISON
    # =========================================================================
    st.markdown("#### 🔬 Comparer les modèles")
    if st.button("⚔️ Page Comparaison", use_container_width=True, type="secondary"):
        st.switch_page("pages/3_🧠_Modèles.py")

    st.markdown("---")

    # Informations sur le mode
    st.markdown("#### ℹ️ Mode Actuel")
    if st.session_state.get("use_mock", True):
        st.info("**Démonstration**\n\nPrédictions simulées")
    else:
        st.success("**Production**\n\nModèles ML actifs")
