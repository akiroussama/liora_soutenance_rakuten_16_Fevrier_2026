"""
Page de comparaison des modèles de classification.

Cette page permet de comparer les 3 modèles texte ou les 3 modèles image
côte à côte avec des visualisations sophistiquées:
- Tableau comparatif
- Bar charts
- Radar charts
- Ranking visuel
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import APP_CONFIG, ASSETS_DIR
from utils.mock_classifier import (
    MultiModelClassifier,
    TEXT_MODELS,
    IMAGE_MODELS,
    get_available_text_models,
    get_available_image_models,
)
from utils.category_mapping import get_category_info
from utils.image_utils import load_image_from_upload, validate_image
from utils.preprocessing import preprocess_product_text
from utils.ui_utils import load_css

# =============================================================================
# Configuration de la page
# =============================================================================
st.set_page_config(
    page_title=f"Modèles - {APP_CONFIG['title']}",
    page_icon="🧠",
    layout=APP_CONFIG["layout"],
)

# Charger le CSS
load_css(ASSETS_DIR / "style.css")

# =============================================================================
# CSS personnalisé pour cette page
# =============================================================================
st.markdown("""
<style>
/* Header */
.comparison-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    padding: 2rem;
    border-radius: 15px;
    margin-bottom: 2rem;
}

.comparison-title {
    color: white !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    margin: 0 !important;
}

.comparison-subtitle {
    color: rgba(255,255,255,0.7) !important;
    font-size: 1rem !important;
}

/* Model cards */
.model-card {
    background: white;
    padding: 1.5rem;
    border-radius: 15px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    text-align: center;
    transition: all 0.3s ease;
    min-height: 280px;
    position: relative;
    overflow: hidden;
}

.model-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.15);
}

.model-card.winner {
    border: 3px solid #28A745;
    box-shadow: 0 8px 30px rgba(40, 167, 69, 0.3);
}

.model-card.winner::before {
    content: "🏆 MEILLEUR";
    position: absolute;
    top: 10px;
    right: -30px;
    background: #28A745;
    color: white;
    padding: 5px 40px;
    font-size: 0.7rem;
    font-weight: 700;
    transform: rotate(45deg);
}

.model-name {
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    margin-bottom: 0.5rem !important;
}

.model-desc {
    color: #888 !important;
    font-size: 0.75rem !important;
    margin-bottom: 1rem !important;
}

.model-emoji {
    font-size: 2.5rem;
    margin: 0.5rem 0;
}

.model-category {
    color: #333 !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
}

.model-confidence {
    font-size: 2rem;
    font-weight: 800;
    margin: 0.5rem 0;
}

/* Ranking table */
.ranking-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 1rem;
}

.ranking-table th {
    background: #f8f8f8;
    padding: 12px;
    text-align: left;
    font-weight: 600;
    border-bottom: 2px solid #eee;
}

.ranking-table td {
    padding: 12px;
    border-bottom: 1px solid #eee;
}

.ranking-table tr:hover {
    background: #fafafa;
}

.rank-badge {
    display: inline-block;
    width: 28px;
    height: 28px;
    line-height: 28px;
    text-align: center;
    border-radius: 50%;
    font-weight: 700;
    font-size: 0.85rem;
}

.rank-1 {
    background: linear-gradient(135deg, #FFD700, #FFA500);
    color: white;
}

.rank-2 {
    background: linear-gradient(135deg, #C0C0C0, #A0A0A0);
    color: white;
}

.rank-3 {
    background: linear-gradient(135deg, #CD7F32, #8B4513);
    color: white;
}

/* Metrics box */
.metrics-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 15px;
    color: white;
    text-align: center;
}

.metrics-value {
    font-size: 2rem !important;
    font-weight: 800 !important;
    color: white !important;
}

.metrics-label {
    font-size: 0.85rem !important;
    color: rgba(255,255,255,0.8) !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Agreement indicator */
.agreement-high {
    background: linear-gradient(135deg, #28A745, #20c997);
}

.agreement-medium {
    background: linear-gradient(135deg, #FF9800, #ffc107);
}

.agreement-low {
    background: linear-gradient(135deg, #DC3545, #c82333);
}

/* Input section */
.input-section {
    background: #f8f9fa;
    padding: 1.5rem;
    border-radius: 15px;
    margin-bottom: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Initialisation
# =============================================================================
if "multi_model_classifier" not in st.session_state:
    st.session_state.multi_model_classifier = MultiModelClassifier()

if "model_comparison_results" not in st.session_state:
    st.session_state.model_comparison_results = None

if "comparison_mode" not in st.session_state:
    st.session_state.comparison_mode = "text"

# =============================================================================
# Header
# =============================================================================
st.markdown("""
<div class="comparison-header">
    <h1 class="comparison-title">🔬 Comparaison des Modèles</h1>
    <p class="comparison-subtitle">Comparez les performances de nos 3 modèles sur la même entrée</p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# Sélection du mode (Texte ou Image)
# =============================================================================
st.markdown("### 🎯 Choisissez le type de modèles à comparer")

col_mode1, col_mode2 = st.columns(2)

with col_mode1:
    if st.button(
        "📝 Modèles Texte",
        use_container_width=True,
        type="primary" if st.session_state.comparison_mode == "text" else "secondary"
    ):
        st.session_state.comparison_mode = "text"
        st.session_state.model_comparison_results = None
        st.rerun()

with col_mode2:
    if st.button(
        "🖼️ Modèles Image",
        use_container_width=True,
        type="primary" if st.session_state.comparison_mode == "image" else "secondary"
    ):
        st.session_state.comparison_mode = "image"
        st.session_state.model_comparison_results = None
        st.rerun()

st.markdown("---")

# =============================================================================
# Afficher les modèles disponibles
# =============================================================================
if st.session_state.comparison_mode == "text":
    models = get_available_text_models()
    st.markdown("### 📝 Modèles Texte Disponibles")
else:
    models = get_available_image_models()
    st.markdown("### 🖼️ Modèles Image Disponibles")

# Afficher les cartes des modèles
model_cols = st.columns(3)
for col, (model_id, config) in zip(model_cols, models.items()):
    with col:
        st.markdown(f"""
        <div style="background: white; padding: 1rem; border-radius: 10px;
                    border-left: 4px solid {config.color}; margin-bottom: 1rem;">
            <h4 style="color: {config.color}; margin: 0;">{config.name}</h4>
            <p style="color: #888; font-size: 0.8rem; margin: 0.5rem 0 0 0;">{config.description}</p>
            <p style="color: #666; font-size: 0.75rem; margin: 0.3rem 0 0 0;">
                Confiance moyenne: <strong>{config.base_confidence*100:.0f}%</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# Zone d'entrée
# =============================================================================
st.markdown("### 📥 Entrée pour la comparaison")

if st.session_state.comparison_mode == "text":
    st.markdown('<div class="input-section">', unsafe_allow_html=True)

    col_input1, col_input2 = st.columns([2, 1])

    with col_input1:
        compare_designation = st.text_input(
            "Désignation du produit",
            placeholder="Ex: Console PlayStation 5 nouvelle génération",
            key="compare_text_designation"
        )
        compare_description = st.text_area(
            "Description (optionnel)",
            placeholder="Description détaillée du produit...",
            height=100,
            key="compare_text_description"
        )

    with col_input2:
        st.markdown("#### 💡 Exemples rapides")
        examples = [
            ("📚 Livre", "Harry Potter à l'école des sorciers, roman fantastique"),
            ("🎮 Console", "Console PlayStation 5 nouvelle génération Sony"),
            ("🏊 Piscine", "Piscine gonflable ronde pour jardin été"),
        ]
        for emoji_name, text in examples:
            if st.button(emoji_name, key=f"example_{emoji_name}", use_container_width=True):
                st.session_state.compare_text_designation = text
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # Bouton de comparaison
    if st.button("🚀 Comparer les 3 Modèles", use_container_width=True, type="primary", key="btn_compare_text"):
        if not compare_designation or not compare_designation.strip():
            st.error("❌ Veuillez saisir la désignation du produit.")
        else:
            with st.spinner("⏳ Exécution des 3 modèles en parallèle..."):
                full_text = preprocess_product_text(compare_designation, compare_description)
                results = st.session_state.multi_model_classifier.predict_all_text_models(full_text)
                metrics = st.session_state.multi_model_classifier.get_comparison_metrics(results)
                st.session_state.model_comparison_results = {
                    "results": results,
                    "metrics": metrics,
                    "input": compare_designation,
                    "mode": "text"
                }

else:  # Image mode
    st.markdown('<div class="input-section">', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Uploadez une image de produit",
        type=["jpg", "jpeg", "png", "webp"],
        key="compare_image_upload"
    )

    compare_image = None
    if uploaded_file:
        try:
            compare_image = load_image_from_upload(uploaded_file)
            is_valid, message = validate_image(compare_image)
            if is_valid:
                st.image(compare_image, caption="Image à analyser", width=300)
            else:
                st.error(message)
                compare_image = None
        except Exception as e:
            st.error(f"Erreur: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

    # Bouton de comparaison
    if st.button("🚀 Comparer les 3 Modèles", use_container_width=True, type="primary", key="btn_compare_image"):
        if compare_image is None:
            st.error("❌ Veuillez uploader une image.")
        else:
            with st.spinner("⏳ Exécution des 3 modèles en parallèle..."):
                results = st.session_state.multi_model_classifier.predict_all_image_models(compare_image)
                metrics = st.session_state.multi_model_classifier.get_comparison_metrics(results)
                st.session_state.model_comparison_results = {
                    "results": results,
                    "metrics": metrics,
                    "input": "Image uploadée",
                    "mode": "image"
                }


# =============================================================================
# Affichage des résultats
# =============================================================================
if st.session_state.model_comparison_results:
    data = st.session_state.model_comparison_results
    results = data["results"]
    metrics = data["metrics"]
    mode = data["mode"]

    models_config = TEXT_MODELS if mode == "text" else IMAGE_MODELS

    st.markdown("---")
    st.markdown("## 📊 Résultats de la Comparaison")

    # =========================================================================
    # Métriques globales
    # =========================================================================
    st.markdown("### 📈 Métriques Globales")

    agreement_class = (
        "agreement-high" if metrics["agreement_ratio"] >= 1.0 else
        "agreement-medium" if metrics["agreement_ratio"] >= 0.66 else
        "agreement-low"
    )

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

    with col_m1:
        st.markdown(f"""
        <div class="metrics-box {agreement_class}">
            <p class="metrics-value">{metrics['agreement_ratio']*100:.0f}%</p>
            <p class="metrics-label">Accord entre modèles</p>
        </div>
        """, unsafe_allow_html=True)

    with col_m2:
        st.markdown(f"""
        <div class="metrics-box" style="background: linear-gradient(135deg, #2196F3, #21CBF3);">
            <p class="metrics-value">{metrics['avg_confidence']*100:.1f}%</p>
            <p class="metrics-label">Confiance Moyenne</p>
        </div>
        """, unsafe_allow_html=True)

    with col_m3:
        st.markdown(f"""
        <div class="metrics-box" style="background: linear-gradient(135deg, #FF9800, #F57C00);">
            <p class="metrics-value">±{metrics['std_confidence']*100:.1f}%</p>
            <p class="metrics-label">Écart-type</p>
        </div>
        """, unsafe_allow_html=True)

    with col_m4:
        best_model_config = models_config[metrics["best_model"]]
        st.markdown(f"""
        <div class="metrics-box" style="background: linear-gradient(135deg, #28A745, #20c997);">
            <p class="metrics-value" style="font-size: 1.2rem !important;">{best_model_config.short_name}</p>
            <p class="metrics-label">Meilleur Modèle</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =========================================================================
    # Cartes des 3 modèles
    # =========================================================================
    st.markdown("### 🏆 Résultats par Modèle")

    # Trier par confiance
    sorted_results = sorted(results.items(), key=lambda x: x[1].confidence, reverse=True)

    model_cols = st.columns(3)

    for idx, (col, (model_id, result)) in enumerate(zip(model_cols, sorted_results)):
        config = models_config[model_id]
        cat_name, cat_full, cat_emoji = get_category_info(result.category)
        is_winner = idx == 0

        with col:
            winner_class = "winner" if is_winner else ""
            conf_color = config.color

            st.markdown(f"""
            <div class="model-card {winner_class}" style="border-top: 4px solid {config.color};">
                <p class="model-name" style="color: {config.color};">{config.name}</p>
                <p class="model-desc">{config.description}</p>
                <span class="model-emoji">{cat_emoji}</span>
                <p class="model-category">{cat_name}</p>
                <p class="model-confidence" style="color: {config.color};">{result.confidence*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =========================================================================
    # Tableau de classement
    # =========================================================================
    st.markdown("### 📋 Tableau Comparatif Détaillé")

    # Construire le DataFrame
    table_data = []
    for rank, (model_id, result) in enumerate(sorted_results, 1):
        config = models_config[model_id]
        cat_name, cat_full, cat_emoji = get_category_info(result.category)

        table_data.append({
            "Rang": rank,
            "Modèle": config.name,
            "Catégorie": f"{cat_emoji} {cat_name}",
            "Code": result.category,
            "Confiance": f"{result.confidence*100:.1f}%",
            "Confiance_num": result.confidence,
        })

    df = pd.DataFrame(table_data)

    # Afficher avec formatage
    st.dataframe(
        df[["Rang", "Modèle", "Catégorie", "Code", "Confiance"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rang": st.column_config.NumberColumn("🏅 Rang", width="small"),
            "Modèle": st.column_config.TextColumn("📦 Modèle", width="medium"),
            "Catégorie": st.column_config.TextColumn("🏷️ Catégorie", width="large"),
            "Code": st.column_config.TextColumn("🔢 Code", width="small"),
            "Confiance": st.column_config.TextColumn("📊 Confiance", width="small"),
        }
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # =========================================================================
    # Graphiques
    # =========================================================================
    st.markdown("### 📊 Visualisations")

    tab_bar, tab_radar, tab_top5 = st.tabs(["📊 Bar Chart", "🎯 Radar Chart", "📈 Top-5 Comparaison"])

    with tab_bar:
        # Bar chart comparatif
        bar_data = []
        colors = []
        for model_id, result in results.items():
            config = models_config[model_id]
            bar_data.append({
                "Modèle": config.name,
                "Confiance": result.confidence * 100
            })
            colors.append(config.color)

        df_bar = pd.DataFrame(bar_data)
        df_bar = df_bar.sort_values("Confiance", ascending=True)

        # Réorganiser les couleurs selon le tri
        color_map = {d["Modèle"]: c for d, c in zip(bar_data, colors)}
        sorted_colors = [color_map[m] for m in df_bar["Modèle"]]

        fig_bar = go.Figure(go.Bar(
            x=df_bar["Confiance"],
            y=df_bar["Modèle"],
            orientation='h',
            marker_color=sorted_colors,
            text=[f"{c:.1f}%" for c in df_bar["Confiance"]],
            textposition='outside',
            textfont=dict(size=14, color='#333', weight='bold')
        ))

        fig_bar.update_layout(
            title="Comparaison des Confiances",
            xaxis_title="Confiance (%)",
            yaxis_title="",
            height=300,
            xaxis=dict(range=[0, 110]),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#333'),
            title_font=dict(color='#333', size=16),
            margin=dict(l=0, r=50, t=40, b=40)
        )

        st.plotly_chart(fig_bar, use_container_width=True)

    with tab_radar:
        # Radar chart avec métriques simulées
        categories_radar = ['Confiance', 'Précision*', 'Rappel*', 'Vitesse*', 'Robustesse*']

        fig_radar = go.Figure()

        for model_id, result in results.items():
            config = models_config[model_id]

            # Simuler des métriques basées sur la confiance et les caractéristiques du modèle
            np.random.seed(config.seed_offset)
            base = result.confidence
            values = [
                base * 100,  # Confiance
                (base + np.random.uniform(-0.05, 0.08)) * 100,  # Précision
                (base + np.random.uniform(-0.08, 0.05)) * 100,  # Rappel
                70 + np.random.uniform(0, 25),  # Vitesse (simulée)
                (config.base_confidence + np.random.uniform(-0.05, 0.05)) * 100,  # Robustesse
            ]
            values = [min(max(v, 50), 98) for v in values]  # Clip entre 50 et 98

            fig_radar.add_trace(go.Scatterpolar(
                r=values + [values[0]],  # Fermer le polygone
                theta=categories_radar + [categories_radar[0]],
                fill='toself',
                name=config.name,
                line_color=config.color,
                fillcolor=f"rgba{tuple(list(int(config.color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.2])}"
            ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickfont=dict(color='#666')
                ),
                angularaxis=dict(tickfont=dict(color='#333', size=12))
            ),
            showlegend=True,
            title="Profil de Performance des Modèles",
            title_font=dict(color='#333', size=16),
            height=450,
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#333'),
        )

        st.plotly_chart(fig_radar, use_container_width=True)
        st.caption("*Métriques estimées à partir des caractéristiques du modèle")

    with tab_top5:
        # Comparaison des top-5 prédictions
        st.markdown("#### Distribution des probabilités Top-5")

        for model_id, result in sorted_results:
            config = models_config[model_id]

            with st.expander(f"{config.name} - Top 5 Prédictions", expanded=(model_id == sorted_results[0][0])):
                top5_data = []
                for cat_code, prob in result.top_k_predictions[:5]:
                    cat_name, _, cat_emoji = get_category_info(cat_code)
                    top5_data.append({
                        "Catégorie": f"{cat_emoji} {cat_name}",
                        "Probabilité": prob * 100
                    })

                df_top5 = pd.DataFrame(top5_data)

                fig_top5 = go.Figure(go.Bar(
                    x=df_top5["Probabilité"],
                    y=df_top5["Catégorie"],
                    orientation='h',
                    marker_color=[config.color if i == 0 else '#ddd' for i in range(5)],
                    text=[f"{p:.1f}%" for p in df_top5["Probabilité"]],
                    textposition='outside',
                ))

                fig_top5.update_layout(
                    height=200,
                    margin=dict(l=0, r=50, t=10, b=10),
                    xaxis=dict(range=[0, df_top5["Probabilité"].max() * 1.3], showgrid=True, gridcolor='#eee'),
                    yaxis=dict(autorange="reversed"),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )

                st.plotly_chart(fig_top5, use_container_width=True)

# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    st.markdown("### 🔬 Comparaison")
    st.markdown("---")

    st.markdown("#### 📊 Mode actuel")
    if st.session_state.comparison_mode == "text":
        st.info("**Modèles Texte**\n\n3 modèles NLP")
    else:
        st.info("**Modèles Image**\n\n3 modèles Vision")

    st.markdown("---")

    st.markdown("#### 📝 Modèles Texte")
    for model_id, config in TEXT_MODELS.items():
        st.markdown(f"- **{config.short_name}**")

    st.markdown("#### 🖼️ Modèles Image")
    for model_id, config in IMAGE_MODELS.items():
        st.markdown(f"- **{config.short_name}**")

    st.markdown("---")

    if st.button("🔄 Réinitialiser", use_container_width=True):
        st.session_state.model_comparison_results = None
        st.rerun()
