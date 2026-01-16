"""
Application Streamlit pour la classification de produits Rakuten.

Point d'entrée principal de l'application de démonstration.
Cette application utilise une approche multimodale (texte + image)
pour classifier automatiquement les produits en 27 catégories.

Lancement:
    cd src/streamlit
    streamlit run app.py
"""
import streamlit as st
from config import APP_CONFIG, MODEL_CONFIG, THEME

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
# CSS personnalisé (thème Rakuten)
# =============================================================================
st.markdown(f"""
<style>
    /* Couleur principale Rakuten */
    .stApp {{
        background-color: {THEME["background_color"]};
    }}

    /* Headers */
    h1, h2, h3 {{
        color: {THEME["primary_color"]};
    }}

    /* Boutons */
    .stButton > button {{
        background-color: {THEME["primary_color"]};
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }}
    .stButton > button:hover {{
        background-color: #990000;
        color: white;
    }}

    /* Cards */
    .prediction-card {{
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: #F8F8F8;
    }}

    /* Mode indicateur */
    .mode-indicator {{
        background-color: #FFF3CD;
        border: 1px solid #FFE69C;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Initialisation du state
# =============================================================================
def init_session_state():
    """Initialise les variables de session."""
    if "classifier" not in st.session_state:
        from utils.mock_classifier import DemoClassifier
        st.session_state.classifier = DemoClassifier()

    if "use_mock" not in st.session_state:
        st.session_state.use_mock = MODEL_CONFIG["use_mock"]

    if "classification_history" not in st.session_state:
        st.session_state.classification_history = []


init_session_state()


# =============================================================================
# Sidebar - Navigation et informations
# =============================================================================
with st.sidebar:
    # Logo et titre
    st.markdown("# 🛒 Rakuten")
    st.markdown("### Classification de Produits")
    st.markdown("---")

    # Indicateur de mode
    if st.session_state.use_mock:
        st.warning("⚠️ **Mode Démonstration**\n\nLes prédictions sont simulées.")
    else:
        st.success("✅ **Modèles Actifs**\n\nPrédictions en temps réel.")

    st.markdown("---")

    # Navigation info
    st.markdown("### 📑 Pages")
    st.markdown("""
    - **Accueil** - Présentation du projet
    - **Classification** - Classifier un produit
    - **Exploration** - Explorer les données
    - **Performance** - Métriques du modèle
    """)

    st.markdown("---")

    # Footer
    st.markdown("""
    <div style='text-align: center; color: #888; font-size: 0.8rem;'>
        Projet DataScientest<br>
        BMLE - Octobre 2025
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# Page d'accueil principale
# =============================================================================
st.title("🛒 Classification de Produits Rakuten")

st.markdown("""
Bienvenue dans l'application de **classification automatique de produits** développée
dans le cadre du projet Rakuten - DataScientest.
""")

# Présentation du projet
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### 🎯 Objectif du Projet

    Cette application utilise une approche **multimodale** combinant:
    - 📷 **Analyse d'images** (features ResNet50)
    - 📝 **Analyse de texte** (TF-IDF + NLP)

    Pour classifier automatiquement les produits en **27 catégories** distinctes.

    ### 📊 Le Dataset

    Le dataset Rakuten France contient:
    - **84 916 produits** d'entraînement
    - **Images** de produits
    - **Textes** (désignation + description)
    - **27 catégories** de produits

    ### 🚀 Comment utiliser l'application

    1. **Naviguez** vers la page "Classification" via le menu latéral
    2. **Uploadez** une image de produit et/ou saisissez sa description
    3. **Cliquez** sur "Classifier" pour obtenir la prédiction
    4. **Explorez** les résultats et les scores de confiance
    """)

with col2:
    st.markdown("### 📈 Statistiques Clés")

    # Métriques en colonnes
    st.metric("Catégories", "27", help="Nombre de classes de produits")
    st.metric("Produits", "84 916", help="Taille du dataset d'entraînement")
    st.metric("Approche", "Multimodal", help="Texte + Image")

    # Indicateur de mode
    st.markdown("---")
    if st.session_state.use_mock:
        st.info("🔧 **Mode actuel**: Démonstration")
    else:
        st.success("🚀 **Mode actuel**: Production")

# Section des catégories
st.markdown("---")
st.markdown("### 🏷️ Les 27 Catégories de Produits")

from utils.category_mapping import get_all_categories

categories = get_all_categories()

# Afficher les catégories en grille
cols = st.columns(4)
for i, (code, (name, full_name, emoji)) in enumerate(categories.items()):
    with cols[i % 4]:
        st.markdown(f"{emoji} **{name}**")
        st.caption(f"Code: {code}")

# Appel à l'action
st.markdown("---")
st.markdown("### 🎬 Prêt à commencer ?")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🔍 Accéder à la Classification", use_container_width=True):
        st.switch_page("pages/1_🔍_Classification.py")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    <p>Développé avec ❤️ par l'équipe BMLE - DataScientest</p>
    <p style='font-size: 0.8rem;'>Streamlit | Python | Machine Learning</p>
</div>
""", unsafe_allow_html=True)
