"""
Page 3 — Model Architecture.

Tab 1 (Vision): Graphviz voting pipeline diagram (DINOv3 x4, EfficientNet x2, XGBoost x1),
radar chart, calibration plot, and inter-model correlation matrix.
Tab 2 (Text): NLP pipeline code block, benchmark table (LinearSVC 83% > CamemBERT 81%).
"""
import streamlit as st
import pandas as pd
import graphviz
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import APP_CONFIG, ASSETS_DIR
from utils.ui_utils import load_css

st.set_page_config(
    page_title=f"Modèles - {APP_CONFIG['title']}",
    page_icon="🧠",
    layout=APP_CONFIG["layout"],
)

load_css(ASSETS_DIR / "style.css")

st.title("Architecture & Modèles")
st.markdown("---")

tabs = st.tabs(["🖼️ Vision (Image)", "📝 Sémantique (Texte)"])

# ==========================================
# ONGLET 1 : VISION (IMAGE) - MODE DASHBOARD
# ==========================================
with tabs[0]:
    # --- ZONE 1 : LE PIPELINE (Haut de page) ---
    st.markdown("#### 1. Pipeline de Décision : Le 'Voting'")
    
    graph = graphviz.Digraph()
    # rankdir LR = gauche a droite. ratio compress = plus compact
    graph.attr(rankdir='LR', size='12,3', ratio='compress', margin='0')
    
    graph.node('I', 'Input', shape='oval', style='filled', fillcolor='#e0e0e0', fontsize='10')
    graph.node('P', 'Preproc', shape='box', style='rounded', fontsize='10')
    graph.node('D', 'DINOv3', style='filled', fillcolor='#d1c4e9', fontsize='10')
    graph.node('E', 'EffNet', style='filled', fillcolor='#b3e5fc', fontsize='10')
    graph.node('X', 'XGBoost', style='filled', fillcolor='#c8e6c9', fontsize='10')
    graph.node('V', 'VOTING', shape='Mdiamond', style='filled', fillcolor='#ff8a80', fontsize='12')
    graph.node('O', 'Sortie', shape='oval', style='filled', fillcolor='gold', fontsize='10')
    
    graph.edge('I', 'P')
    graph.edge('P', 'D')
    graph.edge('P', 'E')
    graph.edge('D', 'X', style='dashed')
    graph.edge('E', 'X', style='dashed')
    graph.edge('D', 'V', label='x4', color='#7e57c2', penwidth='2')
    graph.edge('E', 'V', label='x2', color='#039be5', penwidth='2')
    graph.edge('X', 'V', label='x1', color='#2e7d32', penwidth='1')
    graph.edge('V', 'O')
    
    # on centre le graphe et on limite sa taille
    c_g1, c_g2, c_g3 = st.columns([1, 4, 1])
    with c_g2:
        st.graphviz_chart(graph, use_container_width=True)
    
    st.markdown("---")
    
    # --- ZONE 2 : LE TABLEAU DE BORD (3 Colonnes alignées) ---
    # c'est ici qu'on regle le probleme de taille
    # on divise l'ecran en 3 parts egales pour forcer les images a etre petites
    c1, c2, c3 = st.columns([1, 1, 1], gap="medium")
    
    # COLONNE 1 : LE RADAR
    with c1:
        st.markdown("##### 🎯 Synthèse (Radar)")
        img_radar = str(ASSETS_DIR / "comparaison profil modele.png")
        if os.path.exists(img_radar):
            st.image(img_radar, use_container_width=True)
            st.caption("✅ **Analyse :** Le Voting (Rouge) enveloppe les autres modèles, cumulant robustesse et précision.")
        else:
            st.warning("Radar introuvable")

    # COLONNE 2 : LA CALIBRATION
    with c2:
        st.markdown("##### 📐 Calibration (XGBoost)")
        img_calib = str(ASSETS_DIR / "calibré.png")
        if os.path.exists(img_calib):
            st.image(img_calib, use_container_width=True)
            st.caption("✅ **Correction :** Le 'Sharpening' (Vert) force XGBoost à trancher pour ne pas diluer le vote.")
        else:
            st.warning("Calibration introuvable")

    # COLONNE 3 : LA MATRICE
    with c3:
        st.markdown("##### 🤝 Diversité (Matrice)")
        img_matrice = str(ASSETS_DIR / "matrice.png")
        if os.path.exists(img_matrice):
            st.image(img_matrice, use_container_width=True)
            st.caption("✅ **Stratégie :** XGBoost (Ligne blanche) est indépendant des réseaux de neurones. C'est la sécurité.")
        else:
            st.warning("Matrice introuvable")

# ==========================================
# ONGLET 2 : SEMANTIQUE (TEXTE)
# ==========================================
with tabs[1]:
    st.subheader("Traitement du Langage Naturel (NLP)")
    
    col_txt1, col_txt2 = st.columns([1, 1], gap="large")
    
    with col_txt1:
        st.markdown("#### Pipeline Technique")
        st.code("""
        1. Clean : Stopwords, Lowercase
        2. Vector : TF-IDF (Word+Char)
        3. Model : LinearSVC
        """, language="python")
        
        st.info("⚡ **Vitesse :** < 10ms par produit. Idéal temps réel.")

    with col_txt2:
        st.markdown("#### Benchmark Texte")
        
        df_text = pd.DataFrame({
            "Modèle": ["LinearSVC", "Random Forest", "LogReg", "CamemBERT"],
            "F1-Score": ["83.0%", "72.0%", "69.5%", "81.0%"],
            "Vitesse": ["Fast", "Medium", "Fast", "Slow"]
        })
        
        # style noir sur vert pour lisibilite
        st.dataframe(
            df_text.style.highlight_max(
                axis=0, 
                subset=["F1-Score"], 
                props="color: black; background-color: #d4edda; font-weight: bold;"
            ),
            use_container_width=True,
            hide_index=True
        )

# Sidebar
with st.sidebar:
    st.markdown("### Modèles")
    st.divider()
    st.metric("Image (Voting)", "92%")
    st.metric("Texte (LinearSVC)", "83%")
    st.metric("Fusion", "~94%")
    st.divider()
    st.markdown("**Voting Weights**")
    st.markdown("DINOv3: 4/7 | EffNet: 2/7 | XGB: 1/7")