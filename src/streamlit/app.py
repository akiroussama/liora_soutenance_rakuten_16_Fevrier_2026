"""
Application Streamlit pour la classification de produits Rakuten.

Page d'accueil avec effet WOW pour la soutenance.
"""
import streamlit as st
from config import APP_CONFIG, MODEL_CONFIG, THEME, ASSETS_DIR
from utils.ui_utils import load_css
from utils.category_mapping import get_all_categories

# =============================================================================
# Configuration de la page
# =============================================================================
st.set_page_config(
    page_title=APP_CONFIG["title"],
    page_icon=APP_CONFIG["icon"],
    layout=APP_CONFIG["layout"],
    initial_sidebar_state=APP_CONFIG["initial_sidebar_state"],
)

# =============================================================================
# CSS personnalisé + animations
# =============================================================================
load_css(ASSETS_DIR / "style.css")

# CSS additionnel pour effets WOW
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
    padding: 1.5rem 0;
    border-bottom: 1px solid #f0f0f0;
    margin-top: 4px;
    display: flex;
    align-items: center;
    border-radius: 12px 12px 0 0;
}

.rakuten-logo-container {
    padding-left: 2.5rem;
    display: flex;
    align-items: center;
}

.rakuten-logo {
    height: 32px;
}

.app-title {
    color: #333;
    font-size: 1.4rem;
    font-weight: 600;
    margin-left: 1.5rem;
    border-left: 1.5px solid #eaeaea;
    padding-left: 1.5rem;
    font-family: 'Outfit', 'Inter', sans-serif;
    letter-spacing: -0.5px;
}

/* Hero section refined - Thinner & More Sobere */
.hero-container {
    background: white;
    padding: 1.5rem 2.5rem;
    border-radius: 0 0 12px 12px;
    margin-bottom: 2.5rem;
    border-bottom: 3.5px solid #BF0000;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
}

.hero-subtitle {
    color: #555 !important;
    font-size: 1.1rem !important;
    font-weight: 300 !important;
    margin: 0 !important;
    font-family: 'Inter', sans-serif;
}

/* Metric cards */
.metric-card {
    background: white;
    padding: 1.5rem;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    border-left: 4px solid #BF0000;
}

.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
}

.metric-value {
    font-size: 2.5rem !important;
    font-weight: 800 !important;
    color: #BF0000 !important;
    margin: 0 !important;
}

.metric-label {
    font-size: 0.9rem !important;
    color: #666 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Pipeline section */
.pipeline-step {
    background: white;
    padding: 1.5rem;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    position: relative;
    min-height: 180px;
}

.pipeline-icon {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
}

.pipeline-title {
    color: #BF0000 !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
}

.pipeline-desc {
    color: #666 !important;
    font-size: 0.85rem !important;
}

.pipeline-arrow {
    font-size: 2rem;
    color: #BF0000;
    display: flex;
    align-items: center;
    justify-content: center;
}

/* Category grid */
.category-card {
    background: white;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    transition: all 0.3s ease;
    cursor: default;
}

.category-card:hover {
    transform: scale(1.05);
    box-shadow: 0 5px 20px rgba(191,0,0,0.2);
    border: 1px solid #BF0000;
}

.category-emoji {
    font-size: 1.8rem;
}

.category-name {
    color: #333 !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    margin-top: 0.3rem;
}

/* Business impact */
.impact-box {
    background: linear-gradient(135deg, #FFF5F5 0%, #FFE5E5 100%);
    padding: 1.5rem;
    border-radius: 15px;
    border-left: 5px solid #BF0000;
}

/* Tech stack */
.tech-badge {
    display: inline-block;
    background: #F0F0F0;
    color: #333;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    margin: 0.3rem;
    font-size: 0.85rem;
    font-weight: 500;
}

/* CTA Button */
.cta-button {
    background: linear-gradient(135deg, #BF0000 0%, #8B0000 100%);
    color: white !important;
    padding: 1rem 2.5rem;
    border-radius: 30px;
    font-size: 1.1rem;
    font-weight: 600;
    border: none;
    cursor: pointer;
    box-shadow: 0 5px 20px rgba(191,0,0,0.4);
    transition: all 0.3s ease;
}

.cta-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(191,0,0,0.5);
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Initialisation
# =============================================================================
def init_session_state():
    if "classifier" not in st.session_state:
        from utils.mock_classifier import DemoClassifier
        st.session_state.classifier = DemoClassifier()
    if "use_mock" not in st.session_state:
        st.session_state.use_mock = MODEL_CONFIG["use_mock"]

init_session_state()

# =============================================================================
# HEADER & HERO SECTION
# =============================================================================
st.markdown("""
<div class="rakuten-top-bar"></div>
<div class="rakuten-header">
    <div class="rakuten-logo-container">
        <img src="https://fr.shopping.rakuten.com/visuels/0_content_square/autres/rakuten-logo7.svg" class="rakuten-logo">
        <span class="app-title">Product Classifier</span>
    </div>
</div>
<div class="hero-container">
    <p class="hero-subtitle">Classification automatique de produits par Intelligence Artificielle</p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# KEY METRICS - Big visual impact
# =============================================================================
col1, col2, col3, col4 = st.columns(4)

metrics = [
    ("84 916", "Produits analysés", "📦"),
    ("27", "Catégories", "🏷️"),
    ("2", "Modalités (Texte + Image)", "🔀"),
    ("85%+", "Précision visée", "🎯"),
]

for col, (value, label, icon) in zip([col1, col2, col3, col4], metrics):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{icon}</div>
            <p class="metric-value">{value}</p>
            <p class="metric-label">{label}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =============================================================================
# BUSINESS CONTEXT
# =============================================================================
st.markdown("## 💼 Le Défi Business")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    <div class="impact-box">
        <h4 style="color: #BF0000; margin-top: 0;">Pourquoi ce projet ?</h4>
        <p style="color: #333; font-size: 1rem; line-height: 1.7;">
            Rakuten France traite <strong>des millions de produits</strong> chaque année.
            La classification manuelle est coûteuse et source d'erreurs.
        </p>
        <p style="color: #333; font-size: 1rem; line-height: 1.7;">
            Notre solution utilise le <strong>Deep Learning multimodal</strong> pour automatiser
            ce processus avec une précision supérieure à la classification humaine.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("#### 📈 Impact attendu")
    st.markdown("""
    - ⏱️ **Temps** : -90% de traitement
    - 💰 **Coûts** : Réduction significative
    - ✅ **Qualité** : Moins d'erreurs
    - 📊 **Scale** : Millions de produits/jour
    """)

st.markdown("<br>", unsafe_allow_html=True)

# =============================================================================
# ML PIPELINE VISUAL
# =============================================================================
st.markdown("## 🔬 Notre Approche Multimodale")

col1, col2, col3, col4, col5 = st.columns([2, 1, 2, 1, 2])

with col1:
    st.markdown("""
    <div class="pipeline-step">
        <div class="pipeline-icon">📷</div>
        <p class="pipeline-title">Image</p>
        <p class="pipeline-desc">Extraction de features avec ResNet50 pré-entraîné sur ImageNet</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="pipeline-arrow">→</div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="pipeline-step">
        <div class="pipeline-icon">🧠</div>
        <p class="pipeline-title">Fusion</p>
        <p class="pipeline-desc">Combinaison intelligente des features texte et image</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="pipeline-arrow">→</div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div class="pipeline-step">
        <div class="pipeline-icon">🎯</div>
        <p class="pipeline-title">Prédiction</p>
        <p class="pipeline-desc">Classification parmi 27 catégories avec score de confiance</p>
    </div>
    """, unsafe_allow_html=True)

# Text pipeline below
st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 3, 2])
with col2:
    st.markdown("""
    <div class="pipeline-step">
        <div class="pipeline-icon">📝</div>
        <p class="pipeline-title">Texte</p>
        <p class="pipeline-desc">NLP : Nettoyage, détection de langue, traduction, TF-IDF vectorization</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =============================================================================
# CATEGORIES SHOWCASE
# =============================================================================
st.markdown("## 🏷️ 27 Catégories de Produits")

categories = get_all_categories()

# Display in grid of 9 columns (3 rows of 9)
rows = [list(categories.items())[i:i+9] for i in range(0, 27, 9)]

for row in rows:
    cols = st.columns(9)
    for col, (code, (name, full_name, emoji)) in zip(cols, row):
        with col:
            st.markdown(f"""
            <div class="category-card" title="{full_name}">
                <div class="category-emoji">{emoji}</div>
                <p class="category-name">{name}</p>
            </div>
            """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =============================================================================
# TECH STACK
# =============================================================================
st.markdown("## 🛠️ Technologies Utilisées")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Machine Learning")
    st.markdown("""
    <div>
        <span class="tech-badge">🐍 Python</span>
        <span class="tech-badge">🔥 TensorFlow</span>
        <span class="tech-badge">🧠 ResNet50</span>
        <span class="tech-badge">📊 Scikit-learn</span>
        <span class="tech-badge">📝 NLTK</span>
        <span class="tech-badge">🌐 LangID</span>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("#### Interface & Visualisation")
    st.markdown("""
    <div>
        <span class="tech-badge">🎨 Streamlit</span>
        <span class="tech-badge">📈 Plotly</span>
        <span class="tech-badge">🐼 Pandas</span>
        <span class="tech-badge">🖼️ Pillow</span>
        <span class="tech-badge">📉 Matplotlib</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =============================================================================
# CALL TO ACTION
# =============================================================================
st.markdown("---")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h2 style="color: #333; margin-bottom: 1rem;">🚀 Prêt à tester ?</h2>
        <p style="color: #666; margin-bottom: 1.5rem;">
            Uploadez une image ou décrivez un produit pour voir l'IA en action
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🔍 Classifier un Produit", use_container_width=True, type="primary"):
            st.switch_page("pages/1_🔍_Classification.py")
    with col_btn2:
        if st.button("📊 Explorer les Données", use_container_width=True, type="primary"):
            st.switch_page("pages/2_📊_Exploration.py")

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")

st.markdown("""
<div style="text-align: center; padding: 1rem 0; color: #666;">
    <p style="margin-bottom: 0.5rem;">
        <strong>Projet DataScientest</strong> — Formation BMLE Octobre 2025
    </p>
    <p style="font-size: 0.85rem; color: #888;">
        Machine Learning Engineer | Classification Multimodale | Deep Learning
    </p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("## 🛒 Rakuten")
    st.markdown("**Product Classifier**")
    st.markdown("---")

    # Mode indicator with icon
    if st.session_state.use_mock:
        st.warning("⚠️ **Mode Démo**")
    else:
        st.success("✅ **Production**")

    st.markdown("---")
    st.markdown("### 📑 Navigation")
    st.markdown("""
    - 🏠 **Accueil**
    - 🔍 Classification
    - 📊 Exploration
    """)

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; font-size: 0.75rem; color: #888;">
        v1.0 — Janvier 2025
    </div>
    """, unsafe_allow_html=True)
