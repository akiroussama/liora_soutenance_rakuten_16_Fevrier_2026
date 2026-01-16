"""
Page de conclusions et perspectives.

Cette page présente les résultats clés, l'impact business,
les limites identifiées et les perspectives d'amélioration.
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
    page_title=f"Conclusions - {APP_CONFIG['title']}",
    page_icon="💡",
    layout=APP_CONFIG["layout"],
)

# Charger le CSS
load_css(ASSETS_DIR / "style.css")

# =============================================================================
# CSS personnalisé
# =============================================================================
st.markdown("""
<style>
/* Summary cards */
.summary-card {
    background: white;
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    margin-bottom: 1.5rem;
    border-left: 5px solid #BF0000;
}

.summary-card.success {
    border-left-color: #28A745;
    background: linear-gradient(135deg, #f8fff8 0%, white 100%);
}

.summary-card.warning {
    border-left-color: #FF9800;
    background: linear-gradient(135deg, #fffcf5 0%, white 100%);
}

.summary-card.info {
    border-left-color: #2196F3;
    background: linear-gradient(135deg, #f5f9ff 0%, white 100%);
}

.summary-title {
    color: #333 !important;
    font-size: 1.3rem !important;
    font-weight: 700 !important;
    margin-bottom: 1rem !important;
}

.summary-content {
    color: #555 !important;
    font-size: 1rem !important;
    line-height: 1.7;
}

/* Impact metrics */
.impact-box {
    background: linear-gradient(135deg, #BF0000 0%, #8B0000 100%);
    padding: 1.5rem;
    border-radius: 15px;
    color: white;
    text-align: center;
}

.impact-value {
    font-size: 2.5rem !important;
    font-weight: 800 !important;
    color: white !important;
}

.impact-label {
    font-size: 0.9rem !important;
    color: rgba(255,255,255,0.9) !important;
}

/* Timeline */
.timeline-item {
    background: white;
    padding: 1.2rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    border-left: 4px solid #BF0000;
    position: relative;
}

.timeline-item::before {
    content: "";
    position: absolute;
    left: -10px;
    top: 50%;
    transform: translateY(-50%);
    width: 16px;
    height: 16px;
    background: #BF0000;
    border-radius: 50%;
    border: 3px solid white;
}

.timeline-period {
    color: #BF0000 !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
}

.timeline-content {
    color: #555 !important;
    margin-top: 0.5rem;
}

/* Comparison table */
.comparison-table {
    width: 100%;
    border-collapse: collapse;
}

.comparison-table th {
    background: #BF0000;
    color: white;
    padding: 1rem;
    text-align: center;
}

.comparison-table td {
    padding: 1rem;
    text-align: center;
    border-bottom: 1px solid #eee;
}

.comparison-table tr:nth-child(even) {
    background: #f9f9f9;
}

/* Learning box */
.learning-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 15px;
    color: white;
    margin-bottom: 1rem;
}

.learning-title {
    font-weight: 700;
    font-size: 1.1rem;
    margin-bottom: 0.5rem;
}

.learning-content {
    font-size: 0.95rem;
    line-height: 1.6;
    opacity: 0.95;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Header
# =============================================================================
st.title("💡 Conclusions & Perspectives")

st.markdown("""
Cette page synthétise les résultats de notre projet de classification automatique
de produits Rakuten, l'impact business attendu, les limites identifiées et les
pistes d'amélioration pour aller plus loin.
""")

# =============================================================================
# Résultats clés
# =============================================================================
st.markdown("---")
st.markdown("## ✅ Résultats Clés")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="summary-card success">
        <h3 class="summary-title">✅ Objectif Atteint</h3>
        <p class="summary-content">
            Classification automatique de produits parmi <strong>27 catégories</strong>
            avec une <strong>accuracy de 85%</strong> sur le meilleur modèle (CamemBERT).
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="summary-card success">
        <h3 class="summary-title">🏆 Meilleur Modèle</h3>
        <p class="summary-content">
            <strong>CamemBERT</strong> (modèle Transformer français) surpasse
            les approches classiques TF-IDF grâce à sa compréhension contextuelle.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="summary-card success">
        <h3 class="summary-title">📊 Dataset Maîtrisé</h3>
        <p class="summary-content">
            <strong>84 916 produits</strong> analysés, déséquilibre de classes
            géré, multilinguisme traité (FR, EN, DE).
        </p>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# Impact Business
# =============================================================================
st.markdown("---")
st.markdown("## 💼 Impact Business")

st.markdown("""
<div class="summary-card info">
    <h3 class="summary-title">🎯 Problématique Résolue</h3>
    <p class="summary-content">
        Rakuten France traite des <strong>millions de produits</strong> chaque année.
        La classification manuelle est coûteuse, lente et source d'erreurs.
        Notre solution automatise ce processus critique.
    </p>
</div>
""", unsafe_allow_html=True)

# Tableau comparatif AVANT/APRÈS
st.markdown("### 📊 Comparaison AVANT / APRÈS")

col_before, col_after = st.columns(2)

with col_before:
    st.markdown("#### ❌ AVANT (Manuel)")
    st.markdown("""
    | Métrique | Valeur |
    |----------|--------|
    | Temps par produit | ~5 minutes |
    | Erreur humaine | ~10-15% |
    | Scalabilité | Limitée |
    | Coût | Élevé (ETP) |
    | Cohérence | Variable |
    """)

with col_after:
    st.markdown("#### ✅ APRÈS (IA)")
    st.markdown("""
    | Métrique | Valeur |
    |----------|--------|
    | Temps par produit | <1 seconde |
    | Erreur modèle | ~15% |
    | Scalabilité | Millions/jour |
    | Coût | Faible (infra) |
    | Cohérence | Constante |
    """)

st.markdown("<br>", unsafe_allow_html=True)

# Métriques d'impact
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="impact-box">
        <p class="impact-value">-90%</p>
        <p class="impact-label">Temps de traitement</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="impact-box" style="background: linear-gradient(135deg, #28A745 0%, #20c997 100%);">
        <p class="impact-value">24/7</p>
        <p class="impact-label">Disponibilité</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="impact-box" style="background: linear-gradient(135deg, #2196F3 0%, #21CBF3 100%);">
        <p class="impact-value">100K+</p>
        <p class="impact-label">Produits/jour possible</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="impact-box" style="background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%);">
        <p class="impact-value">ROI+</p>
        <p class="impact-label">Économies ETP</p>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# Limites identifiées
# =============================================================================
st.markdown("---")
st.markdown("## ⚠️ Limites Identifiées")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="summary-card warning">
        <h3 class="summary-title">📉 Classes Minoritaires</h3>
        <p class="summary-content">
            Les catégories avec peu d'exemples (< 1000 produits) ont un
            F1-score plus faible (~60%). Le déséquilibre 15:1 reste un défi.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="summary-card warning">
        <h3 class="summary-title">🖼️ Image Seule Insuffisante</h3>
        <p class="summary-content">
            Les modèles image atteignent ~72% d'accuracy max.
            Le texte contient l'essentiel de l'information discriminante
            pour ce dataset e-commerce.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="summary-card warning">
        <h3 class="summary-title">📝 Dépendance Qualité Texte</h3>
        <p class="summary-content">
            La performance dépend de la qualité des descriptions fournies
            par les vendeurs. Descriptions courtes ou absentes = confiance réduite.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="summary-card warning">
        <h3 class="summary-title">⏰ Drift Temporel</h3>
        <p class="summary-content">
            Le modèle a été entraîné sur des données 2018-2020.
            Les nouveaux produits et tendances pourraient nécessiter
            un réentraînement périodique.
        </p>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# Perspectives
# =============================================================================
st.markdown("---")
st.markdown("## 🚀 Perspectives & Améliorations")

st.markdown("### 📅 Roadmap Technique")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### Court terme (1-3 mois)")
    st.markdown("""
    <div class="timeline-item">
        <p class="timeline-period">Data Augmentation</p>
        <p class="timeline-content">
            Oversampling / SMOTE pour les classes minoritaires.
            Paraphrase des descriptions pour enrichir le dataset.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="timeline-item">
        <p class="timeline-period">Ensemble Learning</p>
        <p class="timeline-content">
            Voting des 6 modèles pour combiner leurs forces.
            Pondération par confiance.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="timeline-item">
        <p class="timeline-period">Seuil de Confiance</p>
        <p class="timeline-content">
            Revue humaine automatique si confiance < 70%.
            Amélioration progressive via feedback loop.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("#### Moyen terme (3-6 mois)")
    st.markdown("""
    <div class="timeline-item">
        <p class="timeline-period">Fine-tuning CamemBERT</p>
        <p class="timeline-content">
            Adaptation au domaine e-commerce avec vocabulaire spécifique
            (marques, caractéristiques produits).
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="timeline-item">
        <p class="timeline-period">Modèle CLIP</p>
        <p class="timeline-content">
            Vision-Language Model de OpenAI pour une vraie fusion
            multimodale texte-image.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="timeline-item">
        <p class="timeline-period">Active Learning</p>
        <p class="timeline-content">
            Sélection intelligente des cas difficiles pour annotation
            humaine et amélioration continue.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("#### Long terme (MLOps)")
    st.markdown("""
    <div class="timeline-item">
        <p class="timeline-period">Pipeline CI/CD</p>
        <p class="timeline-content">
            Automatisation avec MLflow/Kubeflow.
            Tests automatiques, versioning des modèles.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="timeline-item">
        <p class="timeline-period">Monitoring Drift</p>
        <p class="timeline-content">
            Détection automatique de la dégradation des performances.
            Alertes et réentraînement déclenché.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="timeline-item">
        <p class="timeline-period">A/B Testing</p>
        <p class="timeline-content">
            Comparaison en production de différentes versions.
            Rollout progressif des améliorations.
        </p>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# Ce que nous avons appris
# =============================================================================
st.markdown("---")
st.markdown("## 📚 Ce que Nous Avons Appris")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="learning-box">
        <p class="learning-title">🔬 Sur le plan technique</p>
        <p class="learning-content">
            L'importance du preprocessing dans les projets NLP.
            Les modèles Transformer (BERT) surpassent les approches classiques
            mais nécessitent plus de ressources. La qualité des données
            prime sur la complexité du modèle.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="learning-box" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
        <p class="learning-title">💼 Sur le plan métier</p>
        <p class="learning-content">
            Comprendre le problème business avant de coder.
            Les métriques techniques (accuracy) doivent être traduites
            en impact business (temps, coût, qualité). La communication
            avec les stakeholders est essentielle.
        </p>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# Conclusion finale
# =============================================================================
st.markdown("---")
st.markdown("## 🎯 Conclusion")

st.markdown("""
<div class="summary-card" style="border-left-color: #BF0000; background: linear-gradient(135deg, #fff5f5 0%, white 100%);">
    <h3 class="summary-title" style="color: #BF0000 !important;">Mission Accomplie</h3>
    <p class="summary-content" style="font-size: 1.1rem;">
        Ce projet démontre qu'il est possible d'automatiser la classification
        de produits e-commerce avec une <strong>précision de 85%</strong> en utilisant
        des techniques de <strong>Machine Learning / Deep Learning</strong>.
        <br><br>
        Notre solution est <strong>scalable</strong>, <strong>maintenable</strong>,
        et <strong>prête pour une mise en production</strong>.
        Les perspectives d'amélioration sont nombreuses et le potentiel
        de valeur business est significatif.
        <br><br>
        <em>Merci de votre attention. Nous sommes prêts pour vos questions.</em>
    </p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    st.markdown("### 💡 Conclusions")
    st.markdown("---")

    st.markdown("#### 📊 Résumé")
    st.success("✅ Accuracy: 85%")
    st.success("✅ 27 catégories")
    st.success("✅ 6 modèles comparés")

    st.markdown("---")

    st.markdown("#### 🏆 Points Forts")
    st.markdown("""
    - CamemBERT performant
    - Pipeline robuste
    - Interface professionnelle
    """)

    st.markdown("#### ⚠️ Points d'Attention")
    st.markdown("""
    - Classes minoritaires
    - Drift temporel
    - Dépendance au texte
    """)

    st.markdown("---")

    st.markdown("#### 🙏 Remerciements")
    st.markdown("""
    - DataScientest
    - Antoine (Mentor)
    - Équipe Rakuten
    """)
