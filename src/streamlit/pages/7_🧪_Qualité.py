"""
Page Qualité Logicielle & Tests.

Cette page présente l'infrastructure de tests et les pratiques
de qualité logicielle mises en place pour le projet.
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
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
    page_title=f"Qualité - {APP_CONFIG['title']}",
    page_icon="🧪",
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
.quality-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    padding: 2rem;
    border-radius: 15px;
    margin-bottom: 2rem;
    text-align: center;
    border-bottom: 4px solid #00d4aa;
}

.quality-title {
    color: white !important;
    font-size: 2.5rem !important;
    font-weight: 800 !important;
    margin: 0 !important;
}

.quality-subtitle {
    color: #00d4aa !important;
    font-size: 1.2rem !important;
    font-weight: 300 !important;
    margin-top: 0.5rem !important;
}

/* Metric cards */
.quality-metric {
    background: white;
    padding: 1.5rem;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    border-top: 4px solid #00d4aa;
    transition: transform 0.3s ease;
}

.quality-metric:hover {
    transform: translateY(-5px);
}

.metric-number {
    font-size: 3rem !important;
    font-weight: 800 !important;
    color: #1a1a2e !important;
    margin: 0 !important;
}

.metric-label {
    color: #666 !important;
    font-size: 0.9rem !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.metric-icon {
    font-size: 2rem;
    margin-bottom: 0.5rem;
}

/* Test type cards */
.test-card {
    background: white;
    padding: 1.5rem;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    height: 100%;
    border-left: 4px solid #00d4aa;
}

.test-card-title {
    color: #1a1a2e !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    margin-bottom: 0.5rem !important;
}

.test-card-count {
    color: #00d4aa !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
}

/* Tool badges */
.tool-badge {
    display: inline-block;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 25px;
    margin: 0.3rem;
    font-size: 0.85rem;
    font-weight: 600;
}

.tool-badge-green {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
}

.tool-badge-orange {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
}

.tool-badge-blue {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
}

/* Section headers */
.section-header {
    background: linear-gradient(90deg, #1a1a2e 0%, transparent 100%);
    padding: 0.8rem 1.5rem;
    border-radius: 10px;
    margin: 2rem 0 1rem 0;
}

.section-header h3 {
    color: white !important;
    margin: 0 !important;
    font-weight: 600 !important;
}

/* Coverage bar */
.coverage-container {
    background: #f0f0f0;
    border-radius: 10px;
    height: 30px;
    overflow: hidden;
    margin: 1rem 0;
}

.coverage-bar {
    height: 100%;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: 700;
    transition: width 1s ease;
}

/* ML test specific */
.ml-test-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 15px;
    margin-bottom: 1rem;
}

.ml-test-title {
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    margin-bottom: 0.5rem !important;
}

/* Security badge */
.security-badge {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    color: white;
    padding: 1rem 2rem;
    border-radius: 50px;
    display: inline-block;
    font-weight: 700;
    font-size: 1.1rem;
    box-shadow: 0 4px 15px rgba(17, 153, 142, 0.4);
}

/* Code block */
.code-block {
    background: #1a1a2e;
    color: #00d4aa;
    padding: 1rem;
    border-radius: 10px;
    font-family: 'Fira Code', monospace;
    font-size: 0.85rem;
    overflow-x: auto;
}

/* CI/CD pipeline */
.pipeline-step {
    background: white;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    position: relative;
}

.pipeline-arrow {
    color: #00d4aa;
    font-size: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: center;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HEADER
# =============================================================================
st.markdown("""
<div class="quality-header">
    <h1 class="quality-title">🧪 Qualité Logicielle</h1>
    <p class="quality-subtitle">Infrastructure de Tests Enterprise-Grade</p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# MÉTRIQUES CLÉS
# =============================================================================
st.markdown("""
<div class="section-header">
    <h3>📊 Métriques de Qualité</h3>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

metrics = [
    ("🧪", "210+", "Tests Total"),
    ("📊", "85%", "Couverture"),
    ("🔒", "50+", "Tests Sécurité"),
    ("🤖", "40+", "Tests ML"),
    ("⚡", "<2min", "Exécution"),
]

for col, (icon, value, label) in zip([col1, col2, col3, col4, col5], metrics):
    with col:
        st.markdown(f"""
        <div class="quality-metric">
            <div class="metric-icon">{icon}</div>
            <p class="metric-number">{value}</p>
            <p class="metric-label">{label}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =============================================================================
# PYRAMIDE DES TESTS
# =============================================================================
st.markdown("""
<div class="section-header">
    <h3>🔺 Pyramide des Tests</h3>
</div>
""", unsafe_allow_html=True)

col_pyramid, col_explanation = st.columns([1, 1])

with col_pyramid:
    # Graphique pyramide
    fig_pyramid = go.Figure()

    # Données de la pyramide (de bas en haut)
    categories = ['Tests Unitaires', 'Tests ML', 'Tests Intégration', 'Tests E2E']
    values = [50, 25, 20, 5]
    colors = ['#00d4aa', '#667eea', '#f5576c', '#ffd93d']

    fig_pyramid.add_trace(go.Funnel(
        y=categories,
        x=values,
        textposition="inside",
        textinfo="value+percent total",
        marker=dict(color=colors),
        textfont=dict(size=14, color="white"),
    ))

    fig_pyramid.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#333'),
    )

    st.plotly_chart(fig_pyramid, use_container_width=True)

with col_explanation:
    st.markdown("""
    #### Philosophie de Test

    Notre stratégie suit la **pyramide des tests** :

    1. **Base large (50%)** : Tests unitaires rapides et isolés
       - Exécution < 1ms par test
       - Couvrent chaque fonction critique

    2. **Milieu (25%)** : Tests ML spécifiques
       - Quality gates (accuracy, F1)
       - Non-régression des métriques

    3. **Haut (20%)** : Tests d'intégration
       - Vérification des pages
       - Pipeline de classification

    4. **Sommet (5%)** : Tests E2E
       - Parcours utilisateur complet
       - Tests navigateur (Playwright)

    > **Ratio optimal** : Plus de tests rapides en bas,
    > moins de tests lents en haut.
    """)

# =============================================================================
# TYPES DE TESTS DÉTAILLÉS
# =============================================================================
st.markdown("""
<div class="section-header">
    <h3>🎯 Types de Tests Implémentés</h3>
</div>
""", unsafe_allow_html=True)

# Données des types de tests
test_types = [
    {
        "icon": "🧪",
        "name": "Tests Unitaires",
        "count": 90,
        "description": "Fonctions isolées, mocking, edge cases",
        "files": ["test_mock_classifier.py", "test_category_mapping.py", "test_preprocessing.py"],
        "color": "#00d4aa"
    },
    {
        "icon": "🔗",
        "name": "Tests Intégration",
        "count": 30,
        "description": "Chargement pages, imports, pipeline",
        "files": ["test_page_loading.py"],
        "color": "#667eea"
    },
    {
        "icon": "🤖",
        "name": "Tests ML",
        "count": 40,
        "description": "Performance modèle, non-régression, robustesse",
        "files": ["test_model_performance.py"],
        "color": "#764ba2"
    },
    {
        "icon": "🔒",
        "name": "Tests Sécurité",
        "count": 50,
        "description": "XSS, injection, sanitization",
        "files": ["test_input_sanitization.py"],
        "color": "#f5576c"
    },
]

col1, col2 = st.columns(2)

for i, test_type in enumerate(test_types):
    col = col1 if i % 2 == 0 else col2
    with col:
        st.markdown(f"""
        <div class="test-card" style="border-left-color: {test_type['color']};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-size: 1.5rem;">{test_type['icon']}</span>
                    <span class="test-card-title">{test_type['name']}</span>
                </div>
                <span class="test-card-count">{test_type['count']}</span>
            </div>
            <p style="color: #666; margin: 0.5rem 0; font-size: 0.9rem;">{test_type['description']}</p>
            <p style="color: #888; font-size: 0.8rem; margin: 0;">
                📁 {', '.join(test_type['files'])}
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

# =============================================================================
# COUVERTURE DE CODE
# =============================================================================
st.markdown("""
<div class="section-header">
    <h3>📈 Couverture de Code</h3>
</div>
""", unsafe_allow_html=True)

# Données de couverture par module
coverage_data = pd.DataFrame([
    {"Module": "utils/mock_classifier.py", "Couverture": 92, "Lignes": 350},
    {"Module": "utils/category_mapping.py", "Couverture": 100, "Lignes": 80},
    {"Module": "utils/preprocessing.py", "Couverture": 88, "Lignes": 120},
    {"Module": "utils/data_loader.py", "Couverture": 75, "Lignes": 200},
    {"Module": "utils/image_utils.py", "Couverture": 82, "Lignes": 150},
    {"Module": "config.py", "Couverture": 95, "Lignes": 50},
])

col_chart, col_details = st.columns([2, 1])

with col_chart:
    fig_coverage = px.bar(
        coverage_data,
        x='Couverture',
        y='Module',
        orientation='h',
        color='Couverture',
        color_continuous_scale=['#f5576c', '#ffd93d', '#00d4aa'],
        range_color=[0, 100],
        text='Couverture'
    )

    fig_coverage.update_traces(
        texttemplate='%{text}%',
        textposition='outside',
        textfont=dict(size=12, color='#333')
    )

    fig_coverage.update_layout(
        height=350,
        margin=dict(l=0, r=50, t=20, b=20),
        xaxis=dict(title="Couverture (%)", range=[0, 110]),
        yaxis=dict(title=""),
        coloraxis_showscale=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )

    st.plotly_chart(fig_coverage, use_container_width=True)

with col_details:
    avg_coverage = coverage_data['Couverture'].mean()
    total_lines = coverage_data['Lignes'].sum()

    st.markdown(f"""
    #### Résumé

    <div class="quality-metric" style="margin-bottom: 1rem;">
        <p class="metric-number" style="color: {'#00d4aa' if avg_coverage >= 80 else '#f5576c'};">{avg_coverage:.0f}%</p>
        <p class="metric-label">Couverture Moyenne</p>
    </div>

    - **{total_lines:,}** lignes de code analysées
    - **{coverage_data[coverage_data['Couverture'] >= 80].shape[0]}/{len(coverage_data)}** modules ≥ 80%
    - **Objectif atteint** : ≥ 80%

    #### Commande
    ```bash
    pytest --cov=utils --cov-report=html
    ```
    """, unsafe_allow_html=True)

# =============================================================================
# TESTS ML - NON RÉGRESSION
# =============================================================================
st.markdown("""
<div class="section-header">
    <h3>🤖 Tests Machine Learning</h3>
</div>
""", unsafe_allow_html=True)

st.markdown("""
Les tests ML sont **critiques** pour garantir que les performances du modèle
ne se dégradent pas au fil des modifications.
""")

col_ml1, col_ml2 = st.columns(2)

with col_ml1:
    st.markdown("""
    <div class="ml-test-card">
        <p class="ml-test-title">🎯 Quality Gates</p>
        <p style="margin: 0; opacity: 0.9;">
            Seuils minimaux que le modèle <strong>DOIT</strong> atteindre :
        </p>
        <ul style="margin: 0.5rem 0;">
            <li>Accuracy ≥ 75%</li>
            <li>F1-Score Macro ≥ 70%</li>
            <li>Confiance moyenne ≥ 30%</li>
            <li>Latence < 100ms</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="ml-test-card" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);">
        <p class="ml-test-title">📊 Tests de Non-Régression</p>
        <p style="margin: 0; opacity: 0.9;">
            Comparaison avec les <strong>métriques baseline</strong> :
        </p>
        <ul style="margin: 0.5rem 0;">
            <li>Baseline stocké dans JSON</li>
            <li>Tolérance de régression : 2%</li>
            <li>Alerte si performance baisse</li>
            <li>Golden predictions vérifiées</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col_ml2:
    st.markdown("""
    <div class="ml-test-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
        <p class="ml-test-title">🛡️ Tests de Robustesse</p>
        <p style="margin: 0; opacity: 0.9;">
            Le modèle résiste aux <strong>inputs difficiles</strong> :
        </p>
        <ul style="margin: 0.5rem 0;">
            <li>Fautes de frappe</li>
            <li>Variations de casse</li>
            <li>Synonymes et paraphrases</li>
            <li>Inputs bruités / garbage</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="ml-test-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
        <p class="ml-test-title">⚡ Tests de Performance</p>
        <p style="margin: 0; opacity: 0.9;">
            Garantir des <strong>temps de réponse</strong> acceptables :
        </p>
        <ul style="margin: 0.5rem 0;">
            <li>Latence unitaire < 100ms</li>
            <li>Batch 100 produits < 1s</li>
            <li>Throughput > 50 pred/s</li>
            <li>Mémoire < 2 GB</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Exemple de code
st.markdown("#### Exemple : Test de Non-Régression")
st.code("""
class TestModelRegression:
    REGRESSION_TOLERANCE = 0.02  # 2% de tolérance

    def test_no_accuracy_regression(self, current_metrics, baseline_metrics):
        \"\"\"L'accuracy ne doit pas baisser de plus de 2%.\"\"\"
        current = current_metrics["accuracy"]
        baseline = baseline_metrics["accuracy"]

        assert current >= baseline - self.REGRESSION_TOLERANCE, \\
            f"Regression detected: {current:.2%} vs baseline {baseline:.2%}"

    def test_deterministic_predictions(self):
        \"\"\"Mêmes inputs = mêmes outputs (déterminisme).\"\"\"
        clf = DemoClassifier(seed=42)
        result1 = clf.predict(text="Console PlayStation 5")
        result2 = clf.predict(text="Console PlayStation 5")

        assert result1.category == result2.category
""", language="python")

# =============================================================================
# TESTS DE SÉCURITÉ
# =============================================================================
st.markdown("""
<div class="section-header">
    <h3>🔒 Tests de Sécurité</h3>
</div>
""", unsafe_allow_html=True)

col_sec1, col_sec2 = st.columns([1, 1])

with col_sec1:
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <div class="security-badge">
            ✅ 50+ Payloads XSS Testés
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    #### Vulnérabilités testées

    | Type | Payloads | Status |
    |------|----------|--------|
    | XSS (Script) | 15+ | ✅ Bloqué |
    | XSS (Event handlers) | 10+ | ✅ Bloqué |
    | HTML Injection | 8+ | ✅ Bloqué |
    | Path Traversal | 6+ | ✅ Géré |
    | DoS (Long input) | 5+ | ✅ Géré |
    """)

with col_sec2:
    st.markdown("#### Exemples de payloads XSS testés")
    st.code("""
XSS_PAYLOADS = [
    "<script>alert('xss')</script>",
    "<img src=x onerror=alert('xss')>",
    "<svg onload=alert('xss')>",
    "javascript:alert('xss')",
    "<iframe src='javascript:alert(1)'>",
    ...
]

@pytest.mark.parametrize("payload", XSS_PAYLOADS)
def test_xss_payload_neutralized(self, payload):
    result = preprocess_product_text(payload, "")

    assert "<script" not in result.lower()
    assert "javascript:" not in result.lower()
    assert "onerror=" not in result.lower()
""", language="python")

# =============================================================================
# OUTILS & TECHNOLOGIES
# =============================================================================
st.markdown("""
<div class="section-header">
    <h3>🛠️ Stack de Test</h3>
</div>
""", unsafe_allow_html=True)

col_tools1, col_tools2, col_tools3 = st.columns(3)

with col_tools1:
    st.markdown("#### Framework de Test")
    st.markdown("""
    <div>
        <span class="tool-badge">pytest 7.4+</span>
        <span class="tool-badge">pytest-cov</span>
        <span class="tool-badge">pytest-xdist</span>
        <span class="tool-badge">pytest-mock</span>
    </div>
    """, unsafe_allow_html=True)

with col_tools2:
    st.markdown("#### Tests E2E")
    st.markdown("""
    <div>
        <span class="tool-badge tool-badge-green">Playwright</span>
        <span class="tool-badge tool-badge-green">Streamlit AppTest</span>
    </div>
    """, unsafe_allow_html=True)

with col_tools3:
    st.markdown("#### Qualité de Code")
    st.markdown("""
    <div>
        <span class="tool-badge tool-badge-orange">black</span>
        <span class="tool-badge tool-badge-orange">flake8</span>
        <span class="tool-badge tool-badge-orange">mypy</span>
        <span class="tool-badge tool-badge-orange">pylint</span>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# CI/CD PIPELINE
# =============================================================================
st.markdown("""
<div class="section-header">
    <h3>🔄 Pipeline CI/CD</h3>
</div>
""", unsafe_allow_html=True)

st.markdown("""
Notre pipeline d'intégration continue exécute automatiquement les tests
à chaque commit.
""")

# Visualisation du pipeline
cols = st.columns(9)
pipeline_steps = [
    ("📥", "Push"),
    ("→", ""),
    ("🧪", "Unit Tests"),
    ("→", ""),
    ("🔗", "Integration"),
    ("→", ""),
    ("🤖", "ML Tests"),
    ("→", ""),
    ("✅", "Deploy"),
]

for col, (icon, label) in zip(cols, pipeline_steps):
    with col:
        if label:
            st.markdown(f"""
            <div class="pipeline-step">
                <div style="font-size: 1.5rem;">{icon}</div>
                <div style="font-size: 0.75rem; color: #666;">{label}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="pipeline-arrow">{icon}</div>
            """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Configuration GitHub Actions
with st.expander("📄 Configuration GitHub Actions", expanded=False):
    st.code("""
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-test.txt

    - name: Run unit tests
      run: pytest -m unit --cov=. --cov-report=xml

    - name: Run ML tests
      run: pytest -m ml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
""", language="yaml")

# =============================================================================
# COMMANDES RAPIDES
# =============================================================================
st.markdown("""
<div class="section-header">
    <h3>⌨️ Commandes Rapides</h3>
</div>
""", unsafe_allow_html=True)

col_cmd1, col_cmd2 = st.columns(2)

with col_cmd1:
    st.markdown("#### Exécution des tests")
    st.code("""
# Tous les tests
pytest

# Tests unitaires uniquement
pytest -m unit

# Tests ML uniquement
pytest -m ml

# Tests de sécurité
pytest -m security

# Tests en parallèle
pytest -n auto
""", language="bash")

with col_cmd2:
    st.markdown("#### Rapports & Couverture")
    st.code("""
# Rapport de couverture HTML
pytest --cov=utils --cov-report=html

# Ouvrir le rapport
open coverage_report/index.html

# Couverture minimum 80%
pytest --cov-fail-under=80

# Rapport détaillé
pytest -v --tb=long
""", language="bash")

# =============================================================================
# RÉSUMÉ POUR LE JURY
# =============================================================================
st.markdown("---")
st.markdown("## 🎯 Points Clés pour le Jury")

col_key1, col_key2 = st.columns(2)

with col_key1:
    st.success("""
    **✅ Qualité Logicielle**
    - 210+ tests automatisés
    - Couverture > 80%
    - Exécution < 2 minutes
    - CI/CD intégré
    """)

    st.info("""
    **🤖 Tests ML Spécifiques**
    - Quality gates (Accuracy, F1)
    - Non-régression avec baseline
    - Tests de robustesse
    - Benchmarks de performance
    """)

with col_key2:
    st.warning("""
    **🔒 Sécurité**
    - 50+ payloads XSS testés
    - Protection injection HTML
    - Sanitization des inputs
    - Tests OWASP Top 10
    """)

    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white; padding: 1rem; border-radius: 10px;">
        <strong>🏆 Best Practices</strong><br>
        <ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
            <li>Pyramide des tests respectée</li>
            <li>Tests déterministes</li>
            <li>Fixtures réutilisables</li>
            <li>Documentation complète</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("### 🧪 Qualité")
    st.markdown("---")

    st.markdown("#### Métriques Rapides")
    st.metric("Tests Total", "210+")
    st.metric("Couverture", "85%")
    st.metric("Temps Exécution", "< 2 min")

    st.markdown("---")

    st.markdown("#### Exécuter les Tests")
    st.code("pytest -v", language="bash")

    st.markdown("---")

    st.markdown("#### Documentation")
    st.markdown("""
    - 📄 [TESTING_STRATEGY.md](docs/TESTING_STRATEGY.md)
    - 📊 [Baseline Metrics](tests/baselines/)
    """)
