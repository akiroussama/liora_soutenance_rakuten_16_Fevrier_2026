"""
Page d'exploration des données Rakuten.

Cette page permet de visualiser:
- La distribution des 27 catégories de produits
- Les statistiques sur les textes (langues, longueurs)
- Des exemples de produits par catégorie
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import APP_CONFIG, THEME, ASSETS_DIR
from utils.category_mapping import CATEGORY_MAPPING, get_category_info
from utils.data_loader import (
    is_data_available,
    get_category_distribution,
    get_text_statistics,
    get_sample_products,
    get_dataset_summary,
    load_training_data
)
from utils.ui_utils import load_css

# =============================================================================
# Configuration de la page
# =============================================================================
st.set_page_config(
    page_title=f"Données - {APP_CONFIG['title']}",
    page_icon="📊",
    layout=APP_CONFIG["layout"],
)

# Charger le CSS
load_css(ASSETS_DIR / "style.css")

# =============================================================================
# En-tête
# =============================================================================
st.title("📊 Exploration des Données")

st.markdown("""
Explorez le dataset Rakuten France utilisé pour entraîner notre modèle de classification.
Découvrez la distribution des catégories, les statistiques textuelles et des exemples de produits.
""")

# Indicateur de source de données
if is_data_available():
    st.success("✅ **Données réelles chargées** - Statistiques basées sur le dataset complet")
else:
    st.info("ℹ️ **Mode démonstration** - Statistiques basées sur des données représentatives")

# =============================================================================
# Métriques clés
# =============================================================================
st.markdown("---")
st.markdown("### 📈 Vue d'ensemble du Dataset")

summary = get_dataset_summary()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Produits (train)",
        f"{summary['train_samples']:,}".replace(",", " "),
        help="Nombre de produits dans le jeu d'entraînement"
    )

with col2:
    st.metric(
        "Produits (test)",
        f"{summary['test_samples']:,}".replace(",", " ") if isinstance(summary['test_samples'], int) else summary['test_samples'],
        help="Nombre de produits dans le jeu de test"
    )

with col3:
    st.metric(
        "Catégories",
        summary['num_categories'],
        help="Nombre de catégories de produits"
    )

with col4:
    st.metric(
        "Features",
        len(summary['features']),
        help="Nombre de caractéristiques par produit"
    )

# =============================================================================
# Distribution des catégories
# =============================================================================
st.markdown("---")
st.markdown("### 🏷️ Distribution des Catégories")

# Charger la distribution
dist_df = get_category_distribution()

# Tabs pour différentes visualisations
tab_bar, tab_pie, tab_table = st.tabs(["📊 Barres", "🥧 Camembert", "📋 Tableau"])

with tab_bar:
    # Graphique en barres horizontales
    fig_bar = px.bar(
        dist_df,
        x='count',
        y='category_name',
        orientation='h',
        color='count',
        color_continuous_scale=['#FFE5E5', '#BF0000'],
        labels={'count': 'Nombre de produits', 'category_name': 'Catégorie'},
        title='Distribution des produits par catégorie',
        text='count'
    )

    fig_bar.update_layout(
        height=700,
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False,
        coloraxis_showscale=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#333333'),
        title_font=dict(color='#BF0000', size=18),
    )

    fig_bar.update_traces(
        textposition='outside',
        textfont=dict(color='#333333', size=10),
        marker_line_color='#BF0000',
        marker_line_width=0.5
    )

    st.plotly_chart(fig_bar, use_container_width=True)

with tab_pie:
    # Top 10 catégories pour le camembert
    top_10 = dist_df.head(10).copy()
    others = pd.DataFrame([{
        'category_name': 'Autres',
        'count': dist_df.iloc[10:]['count'].sum(),
        'percentage': dist_df.iloc[10:]['percentage'].sum()
    }])
    pie_data = pd.concat([top_10, others], ignore_index=True)

    fig_pie = px.pie(
        pie_data,
        values='count',
        names='category_name',
        title='Top 10 des catégories (+ autres)',
        color_discrete_sequence=px.colors.sequential.Reds_r
    )

    fig_pie.update_layout(
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#333333'),
        title_font=dict(color='#BF0000', size=18),
    )

    fig_pie.update_traces(
        textposition='inside',
        textinfo='percent+label',
        textfont=dict(color='white')
    )

    st.plotly_chart(fig_pie, use_container_width=True)

with tab_table:
    # Tableau avec toutes les catégories
    display_df = dist_df[['emoji', 'category_name', 'category_full', 'count', 'percentage']].copy()
    display_df.columns = ['', 'Catégorie', 'Description', 'Produits', '%']
    display_df['Produits'] = display_df['Produits'].apply(lambda x: f"{x:,}".replace(",", " "))
    display_df['%'] = display_df['%'].apply(lambda x: f"{x:.1f}%")

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=600
    )

# =============================================================================
# Statistiques textuelles
# =============================================================================
st.markdown("---")
st.markdown("### 📝 Analyse des Textes")

text_stats = get_text_statistics()

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Désignation (titre)")

    desg_stats = text_stats['designation']
    st.markdown(f"""
    | Métrique | Valeur |
    |----------|--------|
    | Longueur moyenne | **{desg_stats['mean_length']:.1f}** caractères |
    | Longueur médiane | **{desg_stats['median_length']:.0f}** caractères |
    | Minimum | **{desg_stats['min_length']}** caractères |
    | Maximum | **{desg_stats['max_length']}** caractères |
    """)

with col2:
    st.markdown("#### Description")

    desc_stats = text_stats['description']
    st.markdown(f"""
    | Métrique | Valeur |
    |----------|--------|
    | Longueur moyenne | **{desc_stats['mean_length']:.1f}** caractères |
    | Taux de remplissage | **{desc_stats['non_empty_pct']:.1f}%** |
    | Minimum | **{desc_stats['min_length']}** caractères |
    | Maximum | **{desc_stats['max_length']}** caractères |
    """)

# Distribution des langues
if text_stats.get('languages'):
    st.markdown("#### 🌍 Distribution des Langues")

    lang_data = pd.DataFrame([
        {"Langue": lang, "Produits": count}
        for lang, count in text_stats['languages'].items()
    ])

    lang_labels = {
        'fr': '🇫🇷 Français',
        'en': '🇬🇧 Anglais',
        'de': '🇩🇪 Allemand',
        'es': '🇪🇸 Espagnol',
        'it': '🇮🇹 Italien',
        'other': '🌐 Autres'
    }
    lang_data['Langue'] = lang_data['Langue'].map(lambda x: lang_labels.get(x, x))

    fig_lang = px.bar(
        lang_data,
        x='Langue',
        y='Produits',
        color='Produits',
        color_continuous_scale=['#FFE5E5', '#BF0000'],
        title='Répartition des langues détectées'
    )

    fig_lang.update_layout(
        height=400,
        showlegend=False,
        coloraxis_showscale=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#333333'),
        title_font=dict(color='#BF0000', size=16),
    )

    st.plotly_chart(fig_lang, use_container_width=True)

# =============================================================================
# Exemples de produits par catégorie
# =============================================================================
st.markdown("---")
st.markdown("### 🛍️ Exemples de Produits")

# Sélecteur de catégorie
categories_list = [(f"{info[2]} {info[0]} ({code})", code) for code, info in CATEGORY_MAPPING.items()]
selected_display, selected_code = st.selectbox(
    "Choisissez une catégorie",
    categories_list,
    format_func=lambda x: x[0]
)

# Afficher les exemples
X_train, Y_train = load_training_data()
samples = get_sample_products(X_train, Y_train, category_code=selected_code, n_samples=5)

if len(samples) > 0:
    cat_name, cat_full, cat_emoji = get_category_info(selected_code)
    st.markdown(f"#### {cat_emoji} {cat_name}")
    st.caption(cat_full)

    for idx, row in samples.iterrows():
        with st.expander(f"📦 {row['designation'][:80]}{'...' if len(str(row['designation'])) > 80 else ''}", expanded=False):
            st.markdown(f"**Désignation:** {row['designation']}")
            desc = row.get('description', '')
            if pd.notna(desc) and str(desc).strip():
                st.markdown(f"**Description:** {str(desc)[:500]}{'...' if len(str(desc)) > 500 else ''}")
            else:
                st.markdown("*Pas de description*")
else:
    st.warning("Aucun exemple disponible pour cette catégorie.")

# =============================================================================
# Analyse du déséquilibre
# =============================================================================
st.markdown("---")
st.markdown("### ⚖️ Analyse du Déséquilibre des Classes")

# Calculer les métriques de déséquilibre
max_count = dist_df['count'].max()
min_count = dist_df['count'].min()
imbalance_ratio = max_count / min_count

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Catégorie majoritaire",
        f"{dist_df.iloc[0]['category_name']}",
        f"{dist_df.iloc[0]['count']:,} produits".replace(",", " ")
    )

with col2:
    st.metric(
        "Catégorie minoritaire",
        f"{dist_df.iloc[-1]['category_name']}",
        f"{dist_df.iloc[-1]['count']:,} produits".replace(",", " ")
    )

with col3:
    st.metric(
        "Ratio de déséquilibre",
        f"{imbalance_ratio:.1f}x",
        help="Rapport entre la classe la plus fréquente et la moins fréquente"
    )

st.markdown("""
> **Note:** Le déséquilibre des classes est un défi important pour ce dataset.
> Des techniques comme le **SMOTE** ou le **class weighting** sont utilisées
> pour améliorer les performances sur les classes minoritaires.
""")

# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    st.markdown("### 📊 Exploration")
    st.markdown("---")

    st.markdown("#### 📁 Source des données")
    if is_data_available():
        st.success("Données réelles")
    else:
        st.info("Données démo")

    st.markdown("---")

    st.markdown("#### 🔗 Liens utiles")
    st.markdown("""
    - [Rakuten France](https://fr.shopping.rakuten.com/)
    - [Challenge Data](https://challengedata.ens.fr/)
    """)

    st.markdown("---")

    # Export des données
    st.markdown("#### 💾 Export")
    csv_data = dist_df.to_csv(index=False)
    st.download_button(
        label="📥 Télécharger distribution (CSV)",
        data=csv_data,
        file_name="rakuten_category_distribution.csv",
        mime="text/csv"
    )
