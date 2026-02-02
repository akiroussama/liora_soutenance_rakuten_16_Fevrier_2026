# STEP 1: Analyse des Modèles Baseline

## Table des Matières

1. [Introduction](#1-introduction)
2. [Objectifs](#2-objectifs)
3. [Méthodologie](#3-méthodologie)
4. [Modèles Baseline](#4-modèles-baseline)
5. [Métriques d'Évaluation](#5-métriques-dévaluation)
6. [Analyse des Résultats](#6-analyse-des-résultats)
7. [Confusion Matrix](#7-confusion-matrix)
8. [Classes Problématiques](#8-classes-problématiques)
9. [Conclusions](#9-conclusions)
10. [Prochaines Étapes](#10-prochaines-étapes)

---

## 1. Introduction

### 1.1 Contexte du Projet

Le projet Rakuten vise à classifier automatiquement des produits e-commerce en **27 catégories** à partir de leurs images. Ce document détaille l'analyse des modèles baseline réalisée dans le cadre de l'**Étape 3, Step 1** du projet.

### 1.2 Pipeline Global

```
Images (500x500) → EfficientNet-B0 → Features (1280 dims) → Classifieur → Catégorie
```

Les features ont été extraites lors de l'étape de preprocessing. Ce step se concentre sur la partie classification.

### 1.3 Données Utilisées

| Métrique | Valeur |
|----------|--------|
| Images d'entraînement | 84,916 |
| Images de test | 13,812 |
| Nombre de classes | 27 |
| Dimension des features | 1,280 |
| Extracteur | EfficientNet-B0 |
| Ratio de déséquilibre | 13.4x |

---

## 2. Objectifs

### 2.1 Objectifs Principaux

1. **Établir une ligne de base** avec plusieurs familles de modèles
2. **Comparer les performances** de différents algorithmes
3. **Identifier les classes problématiques** (mal classifiées)
4. **Analyser les confusions** entre classes similaires
5. **Comprendre l'impact du déséquilibre** des classes

### 2.2 Questions de Recherche

- Quel type de modèle fonctionne le mieux sur ces features CNN?
- Y a-t-il de l'overfitting? Comment le détecter?
- Quelles classes sont systématiquement confondues?
- Le déséquilibre des classes impacte-t-il les performances?

---

## 3. Méthodologie

### 3.1 Split des Données

```python
# Split stratifié 80/20
X_train, X_val, y_train, y_val = train_test_split(
    X_full, y_full,
    test_size=0.2,
    stratify=y_full,  # Maintient la distribution
    random_state=42   # Reproductibilité
)
```

**Pourquoi stratifié?**
- Assure que chaque classe a la même proportion dans train et val
- Critique pour les classes minoritaires
- Évite les biais d'évaluation

### 3.2 Normalisation

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit sur train uniquement
X_val_scaled = scaler.transform(X_val)          # Transform sur val
```

**Pourquoi StandardScaler?**
- Les features EfficientNet ont des échelles variables
- Les modèles linéaires (LogReg, SVC, MLP) y sont très sensibles
- Formule: `x_scaled = (x - mean) / std`

### 3.3 Gestion du Déséquilibre

```python
# Option 1: class_weight='balanced'
model = LogisticRegression(class_weight='balanced')

# Option 2: Sample weights (XGBoost)
sample_weights = [class_weights[y] for y in y_train]
model.fit(X_train, y_train, sample_weight=sample_weights)
```

**Calcul des poids:**
```
weight[class_i] = n_samples / (n_classes * n_samples_class_i)
```

Les classes minoritaires reçoivent un poids plus élevé.

---

## 4. Modèles Baseline

### 4.1 Tableau Récapitulatif

| Modèle | Famille | Normalisation | Forces | Faiblesses |
|--------|---------|---------------|--------|------------|
| Logistic Regression | Linéaire | ✓ Requise | Interprétable, rapide | Linéaire uniquement |
| Random Forest | Ensemble | ✗ Non requise | Robuste, feature importance | Peut overfitter |
| XGBoost | Boosting | ✗ Non requise | SOTA tabulaire | Complexe à tuner |
| LightGBM | Boosting | ✗ Non requise | Très rapide | Moins stable |
| MLP | Neural Net | ✓ Requise | Non-linéarités | Besoin de tuning |
| LinearSVC | SVM | ✓ Requise | Bon haute dim | Linéaire |

### 4.2 Logistic Regression

**Pourquoi l'inclure?**
- Baseline obligatoire en classification
- Très interprétable (coefficients = importance)
- Rapide à entraîner
- Établit le plancher de performance

**Configuration:**
```python
LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    solver='lbfgs',
    multi_class='multinomial'
)
```

### 4.3 Random Forest

**Pourquoi l'inclure?**
- Robuste au bruit et outliers
- Feature importance native
- Pas besoin de normalisation
- Bon compromis biais-variance

**Configuration:**
```python
RandomForestClassifier(
    n_estimators=200,      # Nombre d'arbres
    max_depth=30,          # Profondeur max
    min_samples_split=5,   # Min samples pour split
    class_weight='balanced'
)
```

**Attention:** RF tend à overfitter sur features haute dimension (1280).

### 4.4 XGBoost

**Pourquoi l'inclure?**
- State-of-the-art sur données tabulaires
- Régularisation L1/L2 intégrée
- Gère les valeurs manquantes
- Très performant en compétition

**Configuration:**
```python
XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,         # Fraction d'échantillons par arbre
    colsample_bytree=0.8   # Fraction de features par arbre
)
```

### 4.5 LightGBM

**Pourquoi l'inclure?**
- 10x plus rapide que XGBoost
- Très bon sur gros datasets (>10k samples)
- Histogram-based splitting
- Moins de mémoire

**Configuration:**
```python
LGBMClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    num_leaves=31,         # Max feuilles par arbre
    class_weight='balanced'
)
```

### 4.6 MLP (Neural Network)

**Pourquoi l'inclure?**
- Capture les non-linéarités complexes
- Architecture similaire au Deep Learning
- Early stopping pour régularisation
- Bon sur features denses (CNN)

**Configuration:**
```python
MLPClassifier(
    hidden_layer_sizes=(512, 256, 128),  # 3 couches
    activation='relu',
    solver='adam',
    alpha=0.001,           # Régularisation L2
    early_stopping=True,
    validation_fraction=0.1
)
```

**Architecture:** 1280 → 512 → 256 → 128 → 27

### 4.7 LinearSVC

**Pourquoi l'inclure?**
- SVM sans kernel (linéaire)
- Très bon sur haute dimension
- Plus rapide que SVC avec kernel RBF
- Interprétable (coefficients)

**Configuration:**
```python
LinearSVC(
    C=1.0,                 # Régularisation
    class_weight='balanced',
    max_iter=2000,
    loss='squared_hinge'
)
```

---

## 5. Métriques d'Évaluation

### 5.1 Métriques Globales

| Métrique | Formule | Interprétation |
|----------|---------|----------------|
| **Accuracy** | TP+TN / Total | Taux de prédictions correctes |
| **Balanced Accuracy** | Moyenne des recalls | Ignore le déséquilibre |
| **F1 Weighted** | Moyenne pondérée par support | Tient compte du déséquilibre |
| **F1 Macro** | Moyenne simple des F1 | Chaque classe compte pareil |
| **Cohen's Kappa** | (Acc - Acc_random) / (1 - Acc_random) | Accord au-delà du hasard |

### 5.2 Métriques par Classe

```
Precision = TP / (TP + FP)
→ "Quand je prédis cette classe, ai-je raison?"

Recall = TP / (TP + FN)
→ "Ai-je trouvé tous les exemples de cette classe?"

F1 = 2 * (Precision * Recall) / (Precision + Recall)
→ Moyenne harmonique (pénalise les déséquilibres P/R)
```

### 5.3 Interprétation des Métriques

| Situation | Precision | Recall | Diagnostic |
|-----------|-----------|--------|------------|
| Haute / Haute | ✓ | ✓ | Classe bien classifiée |
| Haute / Basse | ✓ | ✗ | Classe "prudente" (rate beaucoup) |
| Basse / Haute | ✗ | ✓ | Classe "permissive" (accepte trop) |
| Basse / Basse | ✗ | ✗ | Classe très problématique |

### 5.4 Détection de l'Overfitting

```
Overfitting Gap = Train Accuracy - Val Accuracy

Gap < 5%   → Normal
Gap 5-15%  → Overfitting léger
Gap > 15%  → Overfitting sévère ⚠️
```

**Solutions si overfitting:**
- Augmenter la régularisation (alpha, C)
- Réduire la complexité du modèle (max_depth, hidden_layers)
- Augmenter les données (data augmentation)
- Early stopping

---

## 6. Analyse des Résultats

### 6.1 Comparaison des Modèles

*(Section à compléter après exécution)*

| Modèle | Val Acc | Val F1 | Overfitting | Temps |
|--------|---------|--------|-------------|-------|
| MLP | 60.36% | 60.15% | 29.6% ⚠️ | 161s |
| Random Forest | 56.35% | 54.38% | 43.1% ⚠️ | 109s |
| Logistic Regression | 54.66% | 55.45% | 14.2% | 406s |

### 6.2 Observations Clés

1. **MLP est le meilleur modèle** avec 60.36% accuracy
2. **Random Forest overfit sévèrement** (99.47% train vs 56.35% val)
3. **Logistic Regression** est le plus stable (faible overfitting)
4. **Les modèles boosting** (XGBoost, LightGBM) à tester

### 6.3 Analyse de l'Overfitting

Le gap train-val important indique que:
- Les features EfficientNet contiennent du bruit
- Les modèles complexes (RF, MLP) mémorisent plutôt que généraliser
- Une régularisation plus forte est nécessaire

---

## 7. Confusion Matrix

### 7.1 Comment Lire une Confusion Matrix

```
          Classe Prédite
            A    B    C
Vraie   A [ 80   15   5  ]  → Support classe A = 100
Classe  B [ 10   70   20 ]  → Support classe B = 100
        C [ 5    25   70 ]  → Support classe C = 100

Diagonale = Prédictions correctes
Hors diagonale = Erreurs
```

### 7.2 Normalisation par Ligne (Recall)

Normaliser par ligne montre le **recall** de chaque classe:
```
Ligne A: [0.80, 0.15, 0.05] → 80% des A sont bien classés
```

### 7.3 Interprétation des Confusions

Quand deux classes sont souvent confondues:
1. **Similitude visuelle** (ex: deux types de livres)
2. **Ambiguïté sémantique** (ex: jeux vidéo vs jeux PC)
3. **Features insuffisantes** (ex: couleur similaire)

---

## 8. Classes Problématiques

### 8.1 Identification

*(Section à compléter après analyse)*

Classes avec le F1 le plus bas:
1. Classe X - F1 = 0.XX
2. Classe Y - F1 = 0.XX
3. Classe Z - F1 = 0.XX

### 8.2 Paires Confondues

Top 5 des confusions les plus fréquentes:

| Classe A | Classe B | Confusions | Explication probable |
|----------|----------|------------|---------------------|
| Livres | Livres anciens | XX | Visuellement similaires |
| Jeux vidéo | Jeux PC | XX | Même catégorie sémantique |
| ... | ... | ... | ... |

### 8.3 Impact du Déséquilibre

Corrélation entre support et F1:
- Si positive: le modèle favorise les classes fréquentes
- Si nulle: le `class_weight='balanced'` fonctionne bien
- Si négative: les classes minoritaires sont mieux classées (rare)

---

## 9. Conclusions

### 9.1 Résultats Principaux

1. **Performance globale acceptable** (~60% sur 27 classes)
2. **MLP meilleur modèle** baseline
3. **Overfitting à surveiller** sur Random Forest
4. **Certaines classes systématiquement confondues**

### 9.2 Limites des Baselines

- Features figées (pas de fine-tuning)
- Pas d'optimisation d'hyperparamètres
- Pas d'ensemble de modèles
- Pas de features textuelles (multimodal)

### 9.3 Recommandations

1. **Hyperparameter tuning** sur MLP et XGBoost
2. **Cross-validation** pour robustesse
3. **Fine-tuning EfficientNet** pour améliorer les features
4. **Fusion multimodale** avec le texte

---

## 10. Prochaines Étapes

### Step 2: Optimisation (à venir)

- [ ] GridSearch sur MLP
- [ ] GridSearch sur XGBoost/LightGBM
- [ ] Cross-validation 5-fold
- [ ] SMOTE pour déséquilibre
- [ ] Métriques avancées (AUC-ROC, PR curves)

### Step 3: Deep Learning (à venir)

- [ ] Fine-tuning EfficientNet
- [ ] Ensemble de modèles
- [ ] Interprétabilité (SHAP, LIME)
- [ ] Analyse des erreurs

---

## Annexes

### A. Structure des Fichiers

```
modeling/
├── __init__.py              # Exports du module
├── config.py                # Configuration centralisée
├── baseline_models.py       # Entraînement des modèles
├── baseline_analysis.py     # Analyse des résultats
└── run_step1_baselines.py   # Script principal

outputs/
├── models/                  # Modèles sauvegardés (.joblib)
├── modeling/                # Rapports JSON
└── figures/                 # Visualisations PNG
```

### B. Commandes d'Exécution

```bash
# Exécuter le Step 1 complet
cd implementation
python -m modeling.run_step1_baselines

# Ou en Python
from modeling.run_step1_baselines import run_step1_complete
results = run_step1_complete()
```

### C. Dépendances

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
xgboost>=1.5.0  # Optionnel
lightgbm>=3.3.0  # Optionnel
joblib>=1.1.0
```

---

*Document généré le {date} - Équipe Rakuten DataScientest*
