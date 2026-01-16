"""
Page de présentation du pipeline de preprocessing.

Cette page explique et visualise les étapes de traitement
des données texte et image avant classification.
"""
import streamlit as st
import plotly.graph_objects as go
import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import APP_CONFIG, ASSETS_DIR
from utils.ui_utils import load_css

# =============================================================================
# Configuration de la page
# =============================================================================
st.set_page_config(
    page_title=f"Preprocessing - {APP_CONFIG['title']}",
    page_icon="⚙️",
    layout=APP_CONFIG["layout"],
)

# Charger le CSS
load_css(ASSETS_DIR / "style.css")

# =============================================================================
# CSS personnalisé
# =============================================================================
st.markdown("""
<style>
/* Pipeline containers */
.pipeline-container {
    background: white;
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    margin-bottom: 2rem;
}

.pipeline-title {
    color: #BF0000 !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    margin-bottom: 1.5rem !important;
    border-bottom: 2px solid #BF0000;
    padding-bottom: 0.5rem;
}

/* Pipeline steps */
.pipeline-step {
    background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
    padding: 1.2rem;
    border-radius: 12px;
    text-align: center;
    border: 1px solid #e0e0e0;
    min-height: 120px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.pipeline-step-icon {
    font-size: 2rem;
    margin-bottom: 0.5rem;
}

.pipeline-step-title {
    color: #333 !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    margin: 0.3rem 0 !important;
}

.pipeline-step-desc {
    color: #888 !important;
    font-size: 0.75rem !important;
}

.pipeline-arrow {
    font-size: 2rem;
    color: #BF0000;
    display: flex;
    align-items: center;
    justify-content: center;
}

/* Choice cards */
.choice-card {
    background: white;
    padding: 1.2rem;
    border-radius: 12px;
    border-left: 4px solid #BF0000;
    margin-bottom: 1rem;
}

.choice-title {
    color: #333 !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
}

.choice-reason {
    color: #666 !important;
    font-size: 0.85rem !important;
    margin-top: 0.5rem;
}

/* Demo section */
.demo-input {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 10px;
    border: 1px dashed #ccc;
}

.demo-output {
    background: #e8f5e9;
    padding: 1rem;
    border-radius: 10px;
    border: 1px solid #a5d6a7;
}

/* Stats box */
.stats-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 12px;
    color: white;
    text-align: center;
}

.stats-value {
    font-size: 2rem !important;
    font-weight: 800 !important;
    color: white !important;
}

.stats-label {
    font-size: 0.85rem !important;
    color: rgba(255,255,255,0.8) !important;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Header
# =============================================================================
st.title("⚙️ Pipeline de Preprocessing")

st.markdown("""
Cette page détaille les étapes de transformation des données brutes (texte et image)
en features exploitables par nos modèles de Machine Learning.
""")

# =============================================================================
# Métriques clés du preprocessing
# =============================================================================
st.markdown("### 📊 Statistiques du Preprocessing")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="stats-box">
        <p class="stats-value">84 916</p>
        <p class="stats-label">Produits traités</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stats-box" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
        <p class="stats-value">~15K</p>
        <p class="stats-label">Vocabulaire TF-IDF</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="stats-box" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
        <p class="stats-value">2 048</p>
        <p class="stats-label">Features ResNet50</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="stats-box" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
        <p class="stats-value">5</p>
        <p class="stats-label">Langues détectées</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =============================================================================
# Pipeline Texte
# =============================================================================
st.markdown("---")
st.markdown("### 📝 Pipeline de Traitement Texte")

st.markdown('<div class="pipeline-container">', unsafe_allow_html=True)

# Ligne 1 du pipeline
col1, col2, col3, col4, col5, col6, col7 = st.columns([2, 1, 2, 1, 2, 1, 2])

with col1:
    st.markdown("""
    <div class="pipeline-step">
        <span class="pipeline-step-icon">📄</span>
        <p class="pipeline-step-title">Texte Brut</p>
        <p class="pipeline-step-desc">Designation + Description</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown('<div class="pipeline-arrow">→</div>', unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="pipeline-step">
        <span class="pipeline-step-icon">🧹</span>
        <p class="pipeline-step-title">Nettoyage</p>
        <p class="pipeline-step-desc">HTML, caractères spéciaux</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown('<div class="pipeline-arrow">→</div>', unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div class="pipeline-step">
        <span class="pipeline-step-icon">🌍</span>
        <p class="pipeline-step-title">Détection Langue</p>
        <p class="pipeline-step-desc">langid / langdetect</p>
    </div>
    """, unsafe_allow_html=True)

with col6:
    st.markdown('<div class="pipeline-arrow">→</div>', unsafe_allow_html=True)

with col7:
    st.markdown("""
    <div class="pipeline-step">
        <span class="pipeline-step-icon">🔄</span>
        <p class="pipeline-step-title">Traduction</p>
        <p class="pipeline-step-desc">→ Français (si nécessaire)</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Ligne 2 du pipeline
col1, col2, col3, col4, col5 = st.columns([2, 1, 2, 1, 2])

with col1:
    st.markdown("""
    <div class="pipeline-step">
        <span class="pipeline-step-icon">✂️</span>
        <p class="pipeline-step-title">Tokenization</p>
        <p class="pipeline-step-desc">Découpage en mots</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown('<div class="pipeline-arrow">→</div>', unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="pipeline-step">
        <span class="pipeline-step-icon">📊</span>
        <p class="pipeline-step-title">TF-IDF</p>
        <p class="pipeline-step-desc">Vectorisation</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown('<div class="pipeline-arrow">→</div>', unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div class="pipeline-step" style="background: linear-gradient(135deg, #BF0000 0%, #8B0000 100%); border: none;">
        <span class="pipeline-step-icon" style="filter: brightness(0) invert(1);">🎯</span>
        <p class="pipeline-step-title" style="color: white !important;">Vecteur Final</p>
        <p class="pipeline-step-desc" style="color: rgba(255,255,255,0.8) !important;">~15K dimensions</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# Pipeline Image
# =============================================================================
st.markdown("---")
st.markdown("### 🖼️ Pipeline de Traitement Image")

st.markdown('<div class="pipeline-container">', unsafe_allow_html=True)

col1, col2, col3, col4, col5, col6, col7 = st.columns([2, 1, 2, 1, 2, 1, 2])

with col1:
    st.markdown("""
    <div class="pipeline-step">
        <span class="pipeline-step-icon">🖼️</span>
        <p class="pipeline-step-title">Image Brute</p>
        <p class="pipeline-step-desc">JPG/PNG variable</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown('<div class="pipeline-arrow">→</div>', unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="pipeline-step">
        <span class="pipeline-step-icon">📐</span>
        <p class="pipeline-step-title">Resize</p>
        <p class="pipeline-step-desc">224 × 224 pixels</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown('<div class="pipeline-arrow">→</div>', unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div class="pipeline-step">
        <span class="pipeline-step-icon">⚖️</span>
        <p class="pipeline-step-title">Normalisation</p>
        <p class="pipeline-step-desc">ImageNet mean/std</p>
    </div>
    """, unsafe_allow_html=True)

with col6:
    st.markdown('<div class="pipeline-arrow">→</div>', unsafe_allow_html=True)

with col7:
    st.markdown("""
    <div class="pipeline-step">
        <span class="pipeline-step-icon">🧠</span>
        <p class="pipeline-step-title">ResNet50</p>
        <p class="pipeline-step-desc">Feature extraction</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([3, 1, 3])

with col2:
    st.markdown('<div class="pipeline-arrow" style="font-size: 2.5rem;">↓</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 3, 2])

with col2:
    st.markdown("""
    <div class="pipeline-step" style="background: linear-gradient(135deg, #BF0000 0%, #8B0000 100%); border: none;">
        <span class="pipeline-step-icon" style="filter: brightness(0) invert(1);">🎯</span>
        <p class="pipeline-step-title" style="color: white !important;">Vecteur de Features</p>
        <p class="pipeline-step-desc" style="color: rgba(255,255,255,0.8) !important;">2048 dimensions (couche avg_pool)</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# Justification des choix techniques
# =============================================================================
st.markdown("---")
st.markdown("### 🎯 Choix Techniques et Justifications")

tab_text, tab_image, tab_general = st.tabs(["📝 Choix Texte", "🖼️ Choix Image", "🔧 Choix Généraux"])

with tab_text:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="choice-card">
            <p class="choice-title">TF-IDF plutôt que Word2Vec</p>
            <p class="choice-reason">
                <strong>Raison :</strong> Performance équivalente sur ce dataset,
                mais meilleure interprétabilité et temps de calcul réduit.
                Les embeddings denses n'apportaient pas de gain significatif.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="choice-card">
            <p class="choice-title">Traduction vers le français</p>
            <p class="choice-reason">
                <strong>Raison :</strong> ~85% du dataset est en français.
                Traduire les 15% restants (EN, DE) permet d'avoir un vocabulaire
                unifié et améliore les performances du TF-IDF.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="choice-card">
            <p class="choice-title">Combinaison designation + description</p>
            <p class="choice-reason">
                <strong>Raison :</strong> La designation seule est trop courte
                (< 100 caractères). La description apporte du contexte,
                même si elle n'est pas toujours présente (~70% de remplissage).
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="choice-card">
            <p class="choice-title">Pas de lemmatisation agressive</p>
            <p class="choice-reason">
                <strong>Raison :</strong> Les noms de marques et produits
                (PlayStation, iPhone) doivent rester intacts.
                Uniquement lowercase et suppression ponctuation.
            </p>
        </div>
        """, unsafe_allow_html=True)

with tab_image:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="choice-card">
            <p class="choice-title">ResNet50 plutôt que VGG16</p>
            <p class="choice-reason">
                <strong>Raison :</strong> ResNet50 offre un meilleur compromis
                performance/taille (2048 features vs 4096 pour VGG16).
                Résidual connections améliorent la qualité des features.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="choice-card">
            <p class="choice-title">Couche avg_pool (pas flatten)</p>
            <p class="choice-reason">
                <strong>Raison :</strong> Global Average Pooling réduit la
                dimensionnalité tout en préservant l'information spatiale.
                Moins de risque d'overfitting qu'avec flatten.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="choice-card">
            <p class="choice-title">Pas de data augmentation</p>
            <p class="choice-reason">
                <strong>Raison :</strong> Le dataset est assez grand (85K).
                L'augmentation (flip, rotation) sur des photos produits
                peut dénaturer l'information (texte sur images, orientation).
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="choice-card">
            <p class="choice-title">Normalisation ImageNet</p>
            <p class="choice-reason">
                <strong>Raison :</strong> ResNet50 est pré-entraîné sur ImageNet.
                Utiliser les mêmes mean/std garantit que les features extraites
                sont dans la bonne distribution.
            </p>
        </div>
        """, unsafe_allow_html=True)

with tab_general:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="choice-card">
            <p class="choice-title">Pas de fusion multimodale</p>
            <p class="choice-reason">
                <strong>Raison :</strong> Après expérimentation, la late fusion
                (concaténation texte + image) n'améliore pas significativement
                les résultats par rapport au texte seul. Le texte contient
                l'essentiel de l'information discriminante.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="choice-card">
            <p class="choice-title">Stratification train/test</p>
            <p class="choice-reason">
                <strong>Raison :</strong> Dataset déséquilibré (ratio 15:1).
                La stratification garantit que chaque classe est représentée
                proportionnellement dans les sets train et test.
            </p>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# Démo interactive
# =============================================================================
st.markdown("---")
st.markdown("### 🔬 Démo Interactive du Preprocessing")

col_input, col_output = st.columns(2)

with col_input:
    st.markdown("#### Input (Texte brut)")
    demo_text = st.text_area(
        "Entrez un texte produit",
        value="<p>iPhone 15 Pro Max</p> - Smartphone Apple dernière génération, écran OLED 6.7 pouces, puce A17 Pro",
        height=150,
        key="demo_preprocess_input"
    )

with col_output:
    st.markdown("#### Output (Texte nettoyé)")

    if demo_text:
        # Simuler le preprocessing
        import re

        # Étape 1: Nettoyer HTML
        cleaned = re.sub(r'<[^>]+>', '', demo_text)

        # Étape 2: Nettoyer caractères spéciaux
        cleaned = re.sub(r'[^\w\s\-]', ' ', cleaned)

        # Étape 3: Lowercase et espaces multiples
        cleaned = ' '.join(cleaned.lower().split())

        st.markdown(f"""
        <div class="demo-output">
            <p><strong>Nettoyé :</strong></p>
            <p style="color: #2e7d32;">{cleaned}</p>
            <hr>
            <p><strong>Statistiques :</strong></p>
            <ul>
                <li>Caractères : {len(cleaned)}</li>
                <li>Mots : {len(cleaned.split())}</li>
                <li>Langue détectée : 🇫🇷 Français</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Preprocessing")
    st.markdown("---")

    st.markdown("#### 📊 Résumé")
    st.markdown("""
    **Pipeline Texte**
    - Nettoyage HTML
    - Détection langue
    - Traduction FR
    - TF-IDF (15K dim)

    **Pipeline Image**
    - Resize 224×224
    - Normalisation
    - ResNet50 (2048 dim)
    """)

    st.markdown("---")

    st.markdown("#### 🔗 Ressources")
    st.markdown("""
    - [Documentation TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
    - [ResNet50 Paper](https://arxiv.org/abs/1512.03385)
    """)
