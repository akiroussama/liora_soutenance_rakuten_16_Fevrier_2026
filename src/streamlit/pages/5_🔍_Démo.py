"""
Page 4 — Interactive Demo (PoC).

Three tabs for live classification using real trained models:
  - Text:   Enter a product description -> LinearSVC prediction
  - Image:  Upload a product photo -> Voting System (DINOv3+EfficientNet+XGBoost)
  - Fusion: Both inputs + adjustable weight slider (default 60% image / 40% text)

Business context: automates product classification from ~5min/product (manual)
to <1s (AI) with 88% full automation rate at 80% confidence threshold.
"""
import streamlit as st
import time
import os
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import APP_CONFIG, ASSETS_DIR, MODELS_DIR
from utils.ui_utils import load_css
from utils.real_classifier import MultimodalClassifier
from utils.model_downloader import ensure_models

st.set_page_config(
    page_title=f"Démo - {APP_CONFIG['title']}",
    page_icon="🔍",
    layout=APP_CONFIG["layout"],
)

load_css(ASSETS_DIR / "style.css")

st.title("Démonstration Interactive")

# Contexte métier PoC
st.info("""
**Proof of Concept — Classificateur de produits Rakuten France**

Ce PoC simule le processus de classification automatique des produits sur une marketplace e-commerce.
**Avant** : un opérateur classifie manuellement chaque produit (~5 min, 10-15% d'erreur).
**Après** : le système IA classifie en <1 seconde avec ~6% d'erreur, permettant de traiter **100K+ produits/jour**.
Le taux d'automatisation avec seuil de confiance à 80% est de **88%** (les 12% restants partent en revue humaine).
""")
st.markdown("---")

# Architecture visuelle de la demo
with st.expander("Architecture du systeme de classification", expanded=False):
    img_explain = str(ASSETS_DIR / "explainability_drive.png")
    if os.path.exists(img_explain):
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image(img_explain, width="stretch")
            st.caption("Les 3 modeles du Voting analysent chaque image differemment : "
                       "DINOv3 (attention globale), EfficientNet (details), XGBoost (features statistiques).")

    img_accuracy = str(ASSETS_DIR / "model_accuracy_comparison.png")
    if os.path.exists(img_accuracy):
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image(img_accuracy, width="stretch")
            st.caption("Le VOTING combine 3 architectures complementaires pour une accuracy image de 79.28%.")

# Download models from HF Hub if not present locally (runs once, cached)
@st.cache_resource
def get_clf():
    ensure_models(MODELS_DIR)
    return MultimodalClassifier()

clf = get_clf()

# --- FONCTIONS UTILITAIRES ---

def show_results(results, title="Résultats"):
    # affiche le gagnant et le top 5 avec des barres
    if not results:
        st.error("⚠️ Le modèle n'a renvoyé aucun résultat.")
        return
    
    # 1. le vainqueur
    top = results[0]
    c1, c2 = st.columns([1, 3])
    with c1:
        st.metric("🏆 Prédiction", top['label'])
    with c2:
        st.success(f"**{top['name']}**")
        st.progress(top['confidence'])
        st.caption(f"Confiance: {top['confidence']:.1%}")

    st.markdown("#### 📊 Détails des probabilités (Top 5)")
    
    # 2. le podium
    chart_data = []
    for r in results[:5]:
        chart_data.append({"Produit": r['name'], "Confiance": r['confidence']})
    
    df_chart = pd.DataFrame(chart_data)
    st.dataframe(
        df_chart.style.background_gradient(cmap="Greens", subset=["Confiance"]).format({"Confiance": "{:.1%}"}),
        width="stretch",
        hide_index=True
    )

def explain_voting_system():
    # explicabilite du vote image
    st.info("🧠 **Architecture du Conseil des Sages (Voting)**")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("🦖 DINOv3 (Le Patron)", "Poids 4", "Vision Globale")
        st.progress(0.57) # 4/7
    with c2:
        st.metric("👁️ EffNet (L'Expert)", "Poids 2", "Détails Fins")
        st.progress(0.28) # 2/7
    with c3:
        st.metric("⚡ XGBoost (Le Statisticien)", "Poids 1", "Correction")
        st.progress(0.14) # 1/7
    st.caption("Le système combine ces 3 avis. XGBoost est 'Sharpened' (au cube) pour trancher net.")

def explain_text_tokens(text):
    # simulation nettoyage texte
    tokens = text.lower().split()
    # on garde que les mots longs pour le show
    kept = [w for w in tokens if len(w) > 3]
    st.write("🔠 **Mots-clés captés par le modèle :**")
    st.markdown(" ".join([f"`{w}`" for w in kept]))

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
                                 placeholder="Ex: Piscine gonflable pour enfants intex...")
        btn_txt = st.button("Analyser le Texte", type="primary")
    
    with col2:
        st.subheader("Résultats")
        if btn_txt and txt_input:
            with st.spinner("Lecture et analyse sémantique..."):
                time.sleep(0.3) 
                res = clf.predict_text(txt_input)
                
                # explicabilite texte
                explain_text_tokens(txt_input)
                st.divider()
                show_results(res)

# ==========================================
# ONGLET 2 : IMAGE
# ==========================================
with tabs[1]:
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("Entrée Image")
        img_file = st.file_uploader("Image du produit", type=['jpg', 'png', 'jpeg', 'webp'])
        
        if img_file:
            st.image(img_file, caption="Aperçu", width="stretch")
            with open("temp_demo.jpg", "wb") as f: 
                f.write(img_file.getbuffer())
    
    with col2:
        st.subheader("Analyse Visuelle")
        if img_file:
            if st.button("Lancer le Voting Image", type="primary"):
                with st.spinner("Réunion du Conseil (DINO + XGBoost + EffNet)..."):
                    res = clf.predict_image("temp_demo.jpg")
                    
                    # explicabilite image
                    explain_voting_system()
                    st.divider()
                    show_results(res)

# ==========================================
# ONGLET 3 : FUSION (SLIDER INTERACTIF)
# ==========================================
with tabs[2]:
    st.markdown("### 🧬 Cockpit de Fusion")
    st.info("Ajustez le curseur pour voir comment le Texte et l'Image s'influencent mutuellement.")
    
    # slider interactif poids
    fusion_weight = st.slider("⚖️ Équilibre de la Décision", 0.0, 1.0, 0.6,
                              format="Image: %d%%")
    
    # mise a jour dynamique des poids du classifieur
    clf.w_image = fusion_weight
    clf.w_text = 1.0 - fusion_weight
    
    # affichage visuel des poids
    c_txt, c_mid, c_img = st.columns([1, 6, 1])
    with c_txt: st.write(f"📜 Texte **{int((1-fusion_weight)*100)}%**")
    with c_img: st.write(f"🖼️ Image **{int(fusion_weight*100)}%**")
    with c_mid: st.progress(fusion_weight)
    
    st.divider()

    c1, c2 = st.columns(2, gap="large")
    with c1:
        f_txt = st.text_area("1. Description", height=100, key="fusion_txt")
        f_img = st.file_uploader("2. Image", type=['jpg', 'png'], key="fusion_img")
        
        launch = st.button("Calculer la Fusion 🔥", type="primary")

    with c2:
        if launch and f_txt and f_img:
            with open("temp_fusion.jpg", "wb") as f: f.write(f_img.getbuffer())
            
            with st.spinner("Fusion pondérée en cours..."):
                res = clf.predict_fusion(f_txt, "temp_fusion.jpg")
                show_results(res, title="Résultat Fusionné")
        elif launch:
            st.warning("Remplissez les deux champs (Texte et Image) !")

# Sidebar
with st.sidebar:
    st.markdown("### Démo Interactive")
    st.divider()
    st.markdown("**Poids de fusion actuels**")
    st.metric("Image", f"{int(clf.w_image*100)}%")
    st.metric("Texte", f"{int(clf.w_text*100)}%")
    st.divider()
    st.markdown("**Modèles chargés**")
    st.success("Voting (3 modèles)" if clf.voting else "Voting indisponible")
    st.success("LinearSVC" if clf.text_model else "Texte indisponible")