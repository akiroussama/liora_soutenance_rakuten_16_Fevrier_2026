"""
Page de Performance du Modèle - Métriques et Visualisations.

Affiche les performances du modèle de classification:
- Métriques globales (Accuracy, F1, Precision, Recall)
- Matrice de confusion interactive
- Performance par catégorie
- Courbes d'apprentissage
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import APP_CONFIG, ASSETS_DIR
from utils.category_mapping import CATEGORY_MAPPING, CATEGORY_CODES
from utils.ui_utils import load_css

# =============================================================================
# Configuration
# =============================================================================
st.set_page_config(
    page_title=f"Performance - {APP_CONFIG['title']}",
    page_icon="📈",
    layout=APP_CONFIG["layout"],
)

load_css(ASSETS_DIR / "style.css")

# =============================================================================
# CSS Custom pour cette page
# =============================================================================
st.markdown("""
<style>
.metric-card-perf {
    background: white;
    padding: 1.5rem;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    border-top: 4px solid #BF0000;
}

.metric-card-perf .value {
    font-size: 2.5rem;
    font-weight: 800;
    color: #BF0000;
    margin: 0.5rem 0;
}

.metric-card-perf .label {
    color: #666;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.metric-card-perf .delta {
    font-size: 0.85rem;
    color: #28A745;
    margin-top: 0.3rem;
}

.performance-header {
    background: linear-gradient(135deg, #BF0000 0%, #8B0000 100%);
    padding: 2rem;
    border-radius: 20px;
    margin-bottom: 2rem;
    box-shadow: 0 10px 40px rgba(191,0,0,0.2);
}

.performance-header h1 {
    color: white !important;
    margin: 0 !important;
}

.performance-header p {
    color: rgba(255,255,255,0.9) !important;
    margin-top: 0.5rem !important;
}

.insight-box {
    background: linear-gradient(135deg, #F8F9FA 0%, #E9ECEF 100%);
    padding: 1.2rem;
    border-radius: 12px;
    border-left: 4px solid #BF0000;
    margin: 1rem 0;
}

.insight-box p {
    color: #333 !important;
    margin: 0 !important;
}

.category-perf-card {
    background: white;
    padding: 1rem;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.legend-item {
    display: inline-flex;
    align-items: center;
    margin-right: 1.5rem;
    font-size: 0.85rem;
    color: #666;
}

.legend-color {
    width: 12px;
    height: 12px;
    border-radius: 3px;
    margin-right: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Données Mock de Performance (réalistes)
# =============================================================================
@st.cache_data
def generate_mock_metrics():
    """Génère des métriques de performance réalistes."""
    np.random.seed(42)

    # Métriques globales
    global_metrics = {
        "accuracy": 0.847,
        "f1_macro": 0.823,
        "f1_weighted": 0.851,
        "precision_macro": 0.835,
        "recall_macro": 0.812,
    }

    # Métriques par catégorie (simulées de manière réaliste)
    category_metrics = []
    for code in CATEGORY_CODES:
        name = CATEGORY_MAPPING[code][0]
        emoji = CATEGORY_MAPPING[code][2]

        # Simuler des performances variables mais réalistes
        base_f1 = np.random.uniform(0.70, 0.95)
        precision = base_f1 + np.random.uniform(-0.05, 0.08)
        recall = base_f1 + np.random.uniform(-0.08, 0.05)
        support = np.random.randint(800, 4500)

        category_metrics.append({
            "code": code,
            "name": name,
            "emoji": emoji,
            "f1": min(base_f1, 0.98),
            "precision": min(precision, 0.99),
            "recall": min(recall, 0.97),
            "support": support
        })

    return global_metrics, pd.DataFrame(category_metrics)

@st.cache_data
def generate_confusion_matrix():
    """Génère une matrice de confusion réaliste."""
    np.random.seed(42)
    n_classes = 27

    # Créer une matrice diagonale dominante
    cm = np.zeros((n_classes, n_classes))

    for i in range(n_classes):
        # Valeur diagonale forte
        cm[i, i] = np.random.randint(700, 3500)

        # Quelques confusions réalistes
        n_confusions = np.random.randint(2, 5)
        confusion_indices = np.random.choice(
            [j for j in range(n_classes) if j != i],
            size=n_confusions,
            replace=False
        )
        for j in confusion_indices:
            cm[i, j] = np.random.randint(10, 150)

    return cm.astype(int)

@st.cache_data
def generate_training_history():
    """Génère un historique d'entraînement simulé."""
    epochs = 50

    # Courbes réalistes avec convergence
    train_loss = 2.5 * np.exp(-0.08 * np.arange(epochs)) + 0.3 + np.random.normal(0, 0.02, epochs)
    val_loss = 2.5 * np.exp(-0.07 * np.arange(epochs)) + 0.4 + np.random.normal(0, 0.03, epochs)

    train_acc = 1 - 0.85 * np.exp(-0.1 * np.arange(epochs)) + np.random.normal(0, 0.01, epochs)
    val_acc = 1 - 0.87 * np.exp(-0.09 * np.arange(epochs)) + np.random.normal(0, 0.015, epochs)

    return pd.DataFrame({
        'epoch': np.arange(1, epochs + 1),
        'train_loss': np.clip(train_loss, 0.2, 3),
        'val_loss': np.clip(val_loss, 0.3, 3),
        'train_accuracy': np.clip(train_acc, 0.1, 0.95),
        'val_accuracy': np.clip(val_acc, 0.1, 0.92)
    })

# =============================================================================
# Chargement des données
# =============================================================================
global_metrics, category_df = generate_mock_metrics()
confusion_matrix = generate_confusion_matrix()
training_history = generate_training_history()

# =============================================================================
# HEADER
# =============================================================================
st.markdown("""
<div class="performance-header">
    <h1>📈 Performance du Modèle</h1>
    <p>Analyse détaillée des métriques de classification multimodale</p>
</div>
""", unsafe_allow_html=True)

# Mode indicator
st.info("ℹ️ **Données de démonstration** — Les métriques affichées sont simulées pour illustrer le dashboard. Les vraies métriques seront disponibles après l'entraînement du modèle.")

# =============================================================================
# MÉTRIQUES GLOBALES
# =============================================================================
st.markdown("## 🎯 Métriques Globales")

col1, col2, col3, col4, col5 = st.columns(5)

metrics_display = [
    ("Accuracy", global_metrics["accuracy"], "+2.3%"),
    ("F1 Macro", global_metrics["f1_macro"], "+1.8%"),
    ("F1 Weighted", global_metrics["f1_weighted"], "+2.1%"),
    ("Precision", global_metrics["precision_macro"], "+1.5%"),
    ("Recall", global_metrics["recall_macro"], "+2.7%"),
]

for col, (label, value, delta) in zip([col1, col2, col3, col4, col5], metrics_display):
    with col:
        st.markdown(f"""
        <div class="metric-card-perf">
            <p class="label">{label}</p>
            <p class="value">{value:.1%}</p>
            <p class="delta">▲ {delta} vs baseline</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Insight box
st.markdown("""
<div class="insight-box">
    <p>💡 <strong>Insight:</strong> L'approche multimodale (texte + image) améliore l'accuracy de <strong>+5.2%</strong> par rapport au modèle texte seul et <strong>+8.7%</strong> par rapport au modèle image seul.</p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# MATRICE DE CONFUSION
# =============================================================================
st.markdown("---")
st.markdown("## 🔥 Matrice de Confusion")

# Options d'affichage
col_opt1, col_opt2 = st.columns([1, 3])
with col_opt1:
    normalize = st.checkbox("Normaliser (%)", value=True)

# Préparer les labels
labels = [f"{CATEGORY_MAPPING[code][2]} {CATEGORY_MAPPING[code][0]}" for code in CATEGORY_CODES]
short_labels = [CATEGORY_MAPPING[code][0][:8] for code in CATEGORY_CODES]

# Normaliser si demandé
if normalize:
    cm_display = confusion_matrix.astype(float)
    cm_display = cm_display / cm_display.sum(axis=1, keepdims=True) * 100
    text_template = ".1f"
    colorbar_title = "Pourcentage (%)"
else:
    cm_display = confusion_matrix
    text_template = "d"
    colorbar_title = "Nombre"

# Créer la heatmap avec Plotly
fig_cm = go.Figure(data=go.Heatmap(
    z=cm_display,
    x=short_labels,
    y=short_labels,
    colorscale=[
        [0, '#FFFFFF'],
        [0.2, '#FFE5E5'],
        [0.4, '#FFB4B4'],
        [0.6, '#FF6B6B'],
        [0.8, '#BF0000'],
        [1, '#800000']
    ],
    text=np.round(cm_display, 1 if normalize else 0),
    texttemplate="%{text}",
    textfont={"size": 8},
    hovertemplate="Vrai: %{y}<br>Prédit: %{x}<br>Valeur: %{z:.1f}<extra></extra>",
    colorbar=dict(title=colorbar_title)
))

fig_cm.update_layout(
    height=700,
    xaxis=dict(title="Catégorie Prédite", tickangle=45, tickfont=dict(size=9)),
    yaxis=dict(title="Vraie Catégorie", tickfont=dict(size=9)),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
)

st.plotly_chart(fig_cm, use_container_width=True)

# Légende
st.markdown("""
<div style="text-align: center; margin-top: -1rem;">
    <span class="legend-item"><span class="legend-color" style="background: #FFFFFF; border: 1px solid #CCC;"></span> Aucune confusion</span>
    <span class="legend-item"><span class="legend-color" style="background: #FFB4B4;"></span> Confusion faible</span>
    <span class="legend-item"><span class="legend-color" style="background: #BF0000;"></span> Confusion élevée (diagonale)</span>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# PERFORMANCE PAR CATÉGORIE
# =============================================================================
st.markdown("---")
st.markdown("## 📊 Performance par Catégorie")

# Tri
sort_by = st.selectbox(
    "Trier par",
    ["F1-Score (décroissant)", "F1-Score (croissant)", "Support (décroissant)", "Nom"],
    index=0
)

if sort_by == "F1-Score (décroissant)":
    sorted_df = category_df.sort_values("f1", ascending=False)
elif sort_by == "F1-Score (croissant)":
    sorted_df = category_df.sort_values("f1", ascending=True)
elif sort_by == "Support (décroissant)":
    sorted_df = category_df.sort_values("support", ascending=False)
else:
    sorted_df = category_df.sort_values("name")

# Graphique en barres groupées
fig_cat = go.Figure()

fig_cat.add_trace(go.Bar(
    name='Precision',
    x=[f"{row['emoji']} {row['name']}" for _, row in sorted_df.iterrows()],
    y=sorted_df['precision'] * 100,
    marker_color='#FF6B6B',
    text=[f"{v:.1f}%" for v in sorted_df['precision'] * 100],
    textposition='outside',
    textfont=dict(size=9)
))

fig_cat.add_trace(go.Bar(
    name='Recall',
    x=[f"{row['emoji']} {row['name']}" for _, row in sorted_df.iterrows()],
    y=sorted_df['recall'] * 100,
    marker_color='#BF0000',
    text=[f"{v:.1f}%" for v in sorted_df['recall'] * 100],
    textposition='outside',
    textfont=dict(size=9)
))

fig_cat.add_trace(go.Bar(
    name='F1-Score',
    x=[f"{row['emoji']} {row['name']}" for _, row in sorted_df.iterrows()],
    y=sorted_df['f1'] * 100,
    marker_color='#4A0000',
    text=[f"{v:.1f}%" for v in sorted_df['f1'] * 100],
    textposition='outside',
    textfont=dict(size=9)
))

fig_cat.update_layout(
    barmode='group',
    height=500,
    xaxis=dict(tickangle=45, tickfont=dict(size=10)),
    yaxis=dict(title="Score (%)", range=[0, 110]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
)

st.plotly_chart(fig_cat, use_container_width=True)

# Top et Flop
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🏆 Top 5 Catégories")
    top5 = category_df.nlargest(5, 'f1')
    for _, row in top5.iterrows():
        st.markdown(f"""
        <div class="category-perf-card">
            <span>{row['emoji']} <strong>{row['name']}</strong></span>
            <span style="color: #28A745; font-weight: 700;">{row['f1']:.1%}</span>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("#### ⚠️ Catégories à Améliorer")
    bottom5 = category_df.nsmallest(5, 'f1')
    for _, row in bottom5.iterrows():
        st.markdown(f"""
        <div class="category-perf-card">
            <span>{row['emoji']} <strong>{row['name']}</strong></span>
            <span style="color: #DC3545; font-weight: 700;">{row['f1']:.1%}</span>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# COURBES D'APPRENTISSAGE
# =============================================================================
st.markdown("---")
st.markdown("## 📉 Courbes d'Apprentissage")

col1, col2 = st.columns(2)

with col1:
    # Loss
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(
        x=training_history['epoch'],
        y=training_history['train_loss'],
        name='Train Loss',
        line=dict(color='#BF0000', width=2),
        mode='lines'
    ))
    fig_loss.add_trace(go.Scatter(
        x=training_history['epoch'],
        y=training_history['val_loss'],
        name='Validation Loss',
        line=dict(color='#FF6B6B', width=2, dash='dash'),
        mode='lines'
    ))
    fig_loss.update_layout(
        title="Loss par Epoch",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        height=350,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig_loss, use_container_width=True)

with col2:
    # Accuracy
    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(
        x=training_history['epoch'],
        y=training_history['train_accuracy'] * 100,
        name='Train Accuracy',
        line=dict(color='#28A745', width=2),
        mode='lines'
    ))
    fig_acc.add_trace(go.Scatter(
        x=training_history['epoch'],
        y=training_history['val_accuracy'] * 100,
        name='Validation Accuracy',
        line=dict(color='#20C997', width=2, dash='dash'),
        mode='lines'
    ))
    fig_acc.update_layout(
        title="Accuracy par Epoch",
        xaxis_title="Epoch",
        yaxis_title="Accuracy (%)",
        height=350,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig_acc, use_container_width=True)

# =============================================================================
# COMPARAISON MODALITÉS
# =============================================================================
st.markdown("---")
st.markdown("## 🔀 Comparaison des Modalités")

modality_data = pd.DataFrame({
    'Modalité': ['Texte seul', 'Image seule', 'Multimodal'],
    'Accuracy': [79.5, 72.8, 84.7],
    'F1-Score': [76.2, 68.5, 82.3],
    'Temps inférence (ms)': [45, 120, 180]
})

col1, col2 = st.columns([2, 1])

with col1:
    fig_comp = go.Figure()

    fig_comp.add_trace(go.Bar(
        name='Accuracy',
        x=modality_data['Modalité'],
        y=modality_data['Accuracy'],
        marker_color=['#FF6B6B', '#FFB4B4', '#BF0000'],
        text=[f"{v}%" for v in modality_data['Accuracy']],
        textposition='outside'
    ))

    fig_comp.update_layout(
        title="Accuracy par Modalité",
        yaxis=dict(title="Accuracy (%)", range=[0, 100]),
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )

    st.plotly_chart(fig_comp, use_container_width=True)

with col2:
    st.markdown("#### 📋 Résumé")
    st.markdown("""
    | Modalité | Accuracy | F1 |
    |----------|----------|-----|
    | Texte seul | 79.5% | 76.2% |
    | Image seule | 72.8% | 68.5% |
    | **Multimodal** | **84.7%** | **82.3%** |
    """)

    st.markdown("""
    <div class="insight-box">
        <p>🎯 Le modèle <strong>multimodal</strong> surpasse les approches unimodales en combinant les forces du texte et de l'image.</p>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("### 📈 Performance")
    st.markdown("---")

    st.markdown("#### 🎯 Résumé")
    st.metric("Accuracy", f"{global_metrics['accuracy']:.1%}")
    st.metric("F1-Score", f"{global_metrics['f1_weighted']:.1%}")

    st.markdown("---")
    st.markdown("#### 📥 Export")

    # Export métriques
    metrics_export = category_df.to_csv(index=False)
    st.download_button(
        "📊 Métriques par catégorie",
        metrics_export,
        "rakuten_metrics.csv",
        "text/csv"
    )
