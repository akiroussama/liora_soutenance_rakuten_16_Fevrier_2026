import streamlit as st
import time
import sys
from pathlib import Path

# Hack pour trouver les modules du projet
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.real_classifier import MultimodalClassifier

st.set_page_config(page_title="Démo Rakuten", page_icon="🔍", layout="wide")

st.title("🔍 Démonstration Interactive & Explicabilité")
st.markdown("---")

# Chargement unique du cerveau
@st.cache_resource
def get_clf():
    return MultimodalClassifier()

clf = get_clf()

# --- FONCTIONS UTILITAIRES ---

def show_results(results, title="Résultats"):
    """Affiche le gagnant et le top 5 avec des barres"""
    if not results:
        st.error("⚠️ Le modèle n'a renvoyé aucun résultat. Vérifiez que les fichiers modèles sont bien chargés.")
        return
    
    # 1. Le Vainqueur
    top = results[0]
    st.success(f"🏆 **Prédiction : {top['name']}** (Code: {top['label']})")
    st.metric("Confiance Globale", f"{top['confidence']:.1%}")
    
    st.markdown("#### 📊 Détails des probabilités (Top 5)")
    
    # 2. Le Podium
    for r in results[:5]:
        col_lbl, col_bar, col_pct = st.columns([3, 5, 1])
        with col_lbl: 
            st.write(f"**{r['name']}**")
        with col_bar: 
            st.progress(r['confidence'])
        with col_pct: 
            st.write(f"{r['confidence']:.1%}")

def show_pipeline_steps(mode="text"):
    """Affiche les étapes techniques pour l'explicabilité"""
    with st.expander(f"🛠️ Comprendre le traitement ({mode.upper()})", expanded=True):
        if mode == "text":
            st.info("""
            1. **Nettoyage** : Minuscules, suppression balises HTML.
            2. **Tokenization** : Découpage en mots (TF-IDF Word + Char).
            3. **Modèle** : LinearSVC (Support Vector Machine).
            4. **Calibration** : Conversion du score en probabilité (Softmax).
            """)
        elif mode == "image":
            st.info("""
            1. **Preprocessing** : Redimensionnement (224x224) et normalisation.
            2. **Extraction** : Analyse par DINOv3 et EfficientNet.
            3. **Décision** : XGBoost analyse les vecteurs caractéristiques.
            4. **Voting** : Consensus entre les différents experts.
            """)
        elif mode == "fusion":
            st.info("""
            1. **Analyse Parallèle** : Texte (40%) et Image (60%) travaillent séparément.
            2. **Alignement** : Les scores sont normalisés par catégorie.
            3. **Fusion** : Addition pondérée des vecteurs de probabilité.
            4. **Décision Finale** : La catégorie avec le score combiné le plus haut l'emporte.
            """)

# --- INTERFACE PRINCIPALE ---

tabs = st.tabs(["📝 Analyse Texte", "🖼️ Analyse Image", "🚀 FUSION Multimodale"])

# ==========================================
# ONGLET 1 : TEXTE
# ==========================================
with tabs[0]:
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("Entrée Texte")
        txt_input = st.text_area("Description du produit", height=200, 
                                 placeholder="Ex: Piscine gonflable pour enfants intex, résistante et colorée...")
        btn_txt = st.button("Analyser le Texte", type="primary")
    
    with col2:
        st.subheader("Résultats")
        if btn_txt and txt_input:
            with st.spinner("Lecture et analyse sémantique..."):
                time.sleep(0.5) 
                res = clf.predict_text(txt_input)
                show_pipeline_steps("text")
                st.divider()
                show_results(res)

# ==========================================
# ONGLET 2 : IMAGE (CORRIGÉ STABLE)
# ==========================================
with tabs[1]:
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("Entrée Image")
        img_file = st.file_uploader("Image du produit", type=['jpg', 'png', 'jpeg', 'webp'])
        
        if img_file:
            # RETOUR A LA VERSION STABLE (use_container_width)
            st.image(img_file, caption="Aperçu", use_container_width=True)
            
            with open("temp_demo.jpg", "wb") as f: 
                f.write(img_file.getbuffer())
    
    with col2:
        st.subheader("Résultats")
        if img_file:
            if st.button("Analyser l'Image", type="primary"):
                with st.spinner("Analyse visuelle (DINOv3 + Voting)..."):
                    res = clf.predict_image("temp_demo.jpg")
                    show_pipeline_steps("image")
                    st.divider()
                    show_results(res)

# ==========================================
# ONGLET 3 : FUSION
# ==========================================
with tabs[2]:
    st.markdown("### 🧬 La puissance du Multimodal")
    st.info("💡 La fusion combine les forces du texte et de l'image pour corriger les erreurs de l'un ou l'autre.")
    
    c1, c2 = st.columns(2, gap="large")
    
    with c1:
        f_txt = st.text_area("1. Description", height=100, key="fusion_txt")
    with c2:
        f_img = st.file_uploader("2. Image", type=['jpg', 'png'], key="fusion_img")
        
    if st.button("Lancer la FUSION 🔥", type="primary", help="Cliquez pour lancer l'analyse"):
        if f_txt and f_img:
            with open("temp_fusion.jpg", "wb") as f: f.write(f_img.getbuffer())
            
            with st.spinner("Fusion des intelligences en cours..."):
                res = clf.predict_fusion(f_txt, "temp_fusion.jpg")
                res_col1, res_col2 = st.columns([1, 2])
                with res_col1:
                    show_pipeline_steps("fusion")
                with res_col2:
                    show_results(res, title="Résultat Fusionné")
        else:
            st.warning("⚠️ Merci de remplir le texte ET l'image pour tester la fusion.")