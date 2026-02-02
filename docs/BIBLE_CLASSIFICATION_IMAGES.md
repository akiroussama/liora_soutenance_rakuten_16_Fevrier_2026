# BIBLE - Classification d'Images
## Projet Rakuten - Guide Complet

---

## Table des Matières

1. [Introduction](#1-introduction)
2. [Contexte et Données](#2-contexte-et-données)
3. [Théorie de la Classification](#3-théorie-de-la-classification)
4. [Modèles Implémentés](#4-modèles-implémentés)
5. [Résultats et Comparaison](#5-résultats-et-comparaison)
6. [Analyse Détaillée](#6-analyse-détaillée)
7. [Optimisation et Amélioration](#7-optimisation-et-amélioration)
8. [Guide de Décision](#8-guide-de-décision)
9. [Code et Implémentation](#9-code-et-implémentation)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Introduction

### 1.1 Objectif

Ce document détaille le processus complet de classification des images produits Rakuten, depuis les features extraites par EfficientNet-B0 jusqu'aux prédictions finales.

### 1.2 Pipeline Global

```
Features (1280-dim) → Normalisation → Modèle → Prédictions (27 classes)
      ↓                    ↓              ↓
 train_features.npy   StandardScaler   MLP/RF/LR
```

### 1.3 Résultats Clés

| Métrique | Valeur |
|----------|--------|
| **Meilleur Modèle** | MLP (Multi-Layer Perceptron) |
| **Validation Accuracy** | 60.36% |
| **Validation F1 (weighted)** | 0.6016 |
| **Nombre de Classes** | 27 |
| **Temps d'entraînement** | 161 secondes |

---

## 2. Contexte et Données

### 2.1 Données d'Entrée

Les features ont été extraites par EfficientNet-B0 lors de l'étape de preprocessing :

```python
# Dimensions des données
X_train_full.shape = (84916, 1280)  # 84,916 images, 1280 features
X_test.shape = (13812, 1280)        # 13,812 images test
y_train.shape = (84916,)            # Labels encodés 0-26
```

### 2.2 Split des Données

```python
# Split stratifié 80/20
X_train: 67,932 échantillons (80%)
X_val:   16,984 échantillons (20%)
X_test:  13,812 échantillons (prédictions finales)
```

### 2.3 Distribution des Classes

Le dataset présente un **déséquilibre significatif** :

| Classe | Code | Count | % |
|--------|------|-------|---|
| Majoritaire | 2583 | 10,217 | 12.0% |
| Médiane | 1560 | 5,076 | 6.0% |
| Minoritaire | 1940 | 804 | 0.9% |

**Ratio max/min** : 12.7x

### 2.4 Class Weights

Pour gérer le déséquilibre, des poids de classe ont été calculés :

```python
from sklearn.utils.class_weight import compute_class_weight

# Formule : w_c = n_samples / (n_classes * n_samples_c)
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)

# Résultat :
# Classe minoritaire (1940) : weight ≈ 3.92
# Classe majoritaire (2583) : weight ≈ 0.31
```

---

## 3. Théorie de la Classification

### 3.1 Types de Classificateurs

#### 3.1.1 Modèles Linéaires

**Logistic Regression**
- Hypothèse : frontière de décision linéaire
- Avantages : rapide, interprétable, baseline solide
- Inconvénients : limité pour patterns non-linéaires

```
P(y=k|x) = softmax(Wx + b)
```

#### 3.1.2 Modèles à Base d'Arbres

**Random Forest**
- Ensemble de Decision Trees
- Avantages : robuste, gère bien les features haute dimension
- Inconvénients : peut overfitter, lent sur très gros datasets

```
Prédiction = vote_majoritaire(tree_1, tree_2, ..., tree_n)
```

#### 3.1.3 Réseaux de Neurones

**Multi-Layer Perceptron (MLP)**
- Couches fully-connected avec non-linéarités
- Avantages : capture patterns complexes, flexible
- Inconvénients : nécessite tuning, risque d'overfitting

```
h_1 = ReLU(W_1 * x + b_1)
h_2 = ReLU(W_2 * h_1 + b_2)
y = softmax(W_3 * h_2 + b_3)
```

### 3.2 Métriques d'Évaluation

#### 3.2.1 Accuracy

```
Accuracy = Nombre de prédictions correctes / Total
```

**Limitation** : biaisée pour classes déséquilibrées.

#### 3.2.2 F1-Score

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

**Types** :
- **Macro** : moyenne simple des F1 par classe
- **Weighted** : moyenne pondérée par le support (recommandé pour déséquilibre)
- **Micro** : F1 global (équivalent à accuracy en multiclass)

#### 3.2.3 Matrice de Confusion

Visualise les erreurs de classification :
- Diagonale : prédictions correctes
- Hors diagonale : confusions entre classes

### 3.3 Gestion du Déséquilibre

#### 3.3.1 Class Weights

Pénalise plus fortement les erreurs sur les classes minoritaires :

```python
model = LogisticRegression(class_weight='balanced')
```

#### 3.3.2 Oversampling / Undersampling

- **SMOTE** : génère des exemples synthétiques
- **Random Undersampling** : réduit la classe majoritaire

#### 3.3.3 Focal Loss

Downweight les exemples faciles, focus sur les difficiles :

```
FL(p) = -α(1-p)^γ * log(p)
```

---

## 4. Modèles Implémentés

### 4.1 Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42,
    solver='lbfgs'  # Bon pour multiclass
)
```

**Hyperparamètres** :
- `C` : inverse de la régularisation (default=1.0)
- `solver` : algorithme d'optimisation
- `max_iter` : itérations maximum

### 4.2 Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,      # Nombre d'arbres
    max_depth=30,          # Profondeur max
    min_samples_split=5,   # Min samples pour split
    min_samples_leaf=2,    # Min samples par feuille
    class_weight='balanced',
    random_state=42,
    n_jobs=-1              # Parallélisation
)
```

**Hyperparamètres clés** :
- `n_estimators` : plus = meilleur mais plus lent
- `max_depth` : contrôle l'overfitting
- `min_samples_split/leaf` : régularisation

### 4.3 MLP (Neural Network)

```python
from sklearn.neural_network import MLPClassifier

model = MLPClassifier(
    hidden_layer_sizes=(512, 256, 128),  # Architecture
    activation='relu',
    solver='adam',
    alpha=0.001,                          # L2 regularization
    batch_size=256,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=100,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=42
)
```

**Architecture choisie** :
```
Input (1280) → Dense(512) → ReLU → Dense(256) → ReLU → Dense(128) → ReLU → Output(27)
```

**Hyperparamètres critiques** :
- `hidden_layer_sizes` : profondeur et largeur du réseau
- `alpha` : régularisation L2
- `early_stopping` : évite l'overfitting

### 4.4 Modèles Non Implémentés (Recommandés)

#### XGBoost

```python
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

#### LightGBM

```python
import lightgbm as lgb

model = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    class_weight='balanced',
    random_state=42
)
```

---

## 5. Résultats et Comparaison

### 5.1 Tableau Comparatif

| Modèle | Train Acc | Val Acc | Train F1 | Val F1 | Time (s) | Overfit |
|--------|-----------|---------|----------|--------|----------|---------|
| **MLP** | 0.8999 | **0.6036** | 0.8996 | **0.6016** | 161 | 0.2963 |
| Logistic Regression | 0.6891 | 0.5466 | 0.6899 | 0.5545 | 406 | 0.1425 |
| Random Forest | 0.9947 | 0.5635 | 0.9947 | 0.5438 | 109 | 0.4312 |

### 5.2 Analyse

1. **MLP** : Meilleure généralisation malgré un certain overfitting
2. **Logistic Regression** : Baseline solide, peu d'overfitting
3. **Random Forest** : Fort overfitting (train ~100% vs val ~56%)

### 5.3 Performance par Classe (MLP)

| Classe | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| 1160 | 0.8710 | 0.9052 | **0.8878** | 791 |
| 2583 | 0.8439 | 0.8232 | **0.8334** | 2042 |
| 1920 | 0.7818 | 0.8072 | **0.7943** | 861 |
| ... | ... | ... | ... | ... |
| 1281 | 0.2382 | 0.3285 | **0.2761** | 414 |

**Observations** :
- Classes bien classifiées : 1160, 2583, 1920 (F1 > 0.79)
- Classes difficiles : 1281, 1280, 2220 (F1 < 0.40)
- Corrélation avec la taille de la classe et la similarité visuelle

---

## 6. Analyse Détaillée

### 6.1 Sources d'Erreurs

#### 6.1.1 Confusions Fréquentes

Basé sur la matrice de confusion, les confusions principales :

1. **1280 ↔ 1281** : Produits visuellement similaires
2. **2060 ↔ 2582** : Catégories proches
3. **10 ↔ 40** : Livres/Magazines similaires

#### 6.1.2 Classes Difficiles

| Classe | F1 | Problème Probable |
|--------|-----|-------------------|
| 1281 | 0.28 | Confusion avec 1280 |
| 1280 | 0.37 | Grande variabilité intra-classe |
| 2220 | 0.39 | Sous-représentée |

### 6.2 Analyse de l'Overfitting

```
Overfitting Gap = Train Accuracy - Val Accuracy

MLP:     0.8999 - 0.6036 = 0.2963 (modéré)
RF:      0.9947 - 0.5635 = 0.4312 (sévère)
LogReg:  0.6891 - 0.5466 = 0.1425 (faible)
```

**Solutions** :
- Augmenter la régularisation (`alpha` pour MLP)
- Réduire la complexité du modèle
- Data augmentation
- Dropout (pour réseaux plus profonds)

### 6.3 Importance des Features (Random Forest)

```python
# Top 10 features les plus importantes
importances = rf_model.feature_importances_
top_10 = np.argsort(importances)[-10:]
```

---

## 7. Optimisation et Amélioration

### 7.1 Hyperparameter Tuning

#### Grid Search

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'hidden_layer_sizes': [(256, 128), (512, 256), (512, 256, 128)],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.001, 0.01]
}

grid_search = GridSearchCV(
    MLPClassifier(max_iter=100, early_stopping=True),
    param_grid,
    cv=3,
    scoring='f1_weighted',
    n_jobs=-1
)
```

#### Random Search (recommandé pour grand espace)

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

param_dist = {
    'hidden_layer_sizes': [(256,), (512,), (256, 128), (512, 256)],
    'alpha': uniform(0.0001, 0.01),
    'learning_rate_init': uniform(0.0001, 0.01)
}

random_search = RandomizedSearchCV(
    MLPClassifier(max_iter=100),
    param_dist,
    n_iter=50,
    cv=3,
    scoring='f1_weighted'
)
```

### 7.2 Techniques d'Amélioration

#### 7.2.1 Ensemble Methods

```python
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier([
    ('lr', LogisticRegression(class_weight='balanced')),
    ('mlp', MLPClassifier(hidden_layer_sizes=(512, 256))),
    ('rf', RandomForestClassifier(n_estimators=100))
], voting='soft')
```

#### 7.2.2 Stacking

```python
from sklearn.ensemble import StackingClassifier

stacking = StackingClassifier(
    estimators=[
        ('lr', LogisticRegression()),
        ('rf', RandomForestClassifier())
    ],
    final_estimator=MLPClassifier(hidden_layer_sizes=(64,)),
    cv=5
)
```

### 7.3 Deep Learning (PyTorch)

Pour aller plus loin :

```python
import torch
import torch.nn as nn

class ImageClassifier(nn.Module):
    def __init__(self, input_dim=1280, num_classes=27):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.layers(x)
```

### 7.4 Améliorations Potentielles

| Technique | Gain Estimé | Effort |
|-----------|-------------|--------|
| XGBoost/LightGBM | +2-5% | Faible |
| Hyperparameter Tuning | +1-3% | Moyen |
| Ensemble | +2-4% | Moyen |
| Fine-tuning CNN | +5-10% | Élevé |
| Multimodal (Image+Texte) | +10-15% | Élevé |

---

## 8. Guide de Décision

### 8.1 Choix du Modèle

```
Besoin de rapidité ?
    ├── Oui → Logistic Regression
    └── Non → Besoin d'interprétabilité ?
                ├── Oui → Random Forest
                └── Non → MLP ou XGBoost
```

### 8.2 Quand Utiliser Chaque Modèle

| Contexte | Modèle Recommandé |
|----------|-------------------|
| Baseline rapide | Logistic Regression |
| Interprétabilité | Random Forest |
| Performance maximale | MLP / XGBoost |
| Production (latence) | Logistic Regression |
| Grand dataset | LightGBM |

### 8.3 Trade-offs

| Aspect | LR | RF | MLP | XGB |
|--------|----|----|-----|-----|
| Vitesse entraînement | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| Vitesse inférence | ⭐⭐⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐ |
| Performance | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| Interprétabilité | ⭐⭐⭐ | ⭐⭐⭐ | ⭐ | ⭐⭐ |
| Robustesse | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ |

---

## 9. Code et Implémentation

### 9.1 Script Principal

```bash
# Exécution
cd implementation/
python run_classification.py
```

### 9.2 Notebook Interactif

```bash
# Jupyter Notebook
jupyter notebook notebooks/3.0-classification-images.ipynb
```

### 9.3 Chargement du Modèle Sauvegardé

```python
import joblib
import numpy as np

# Charger le modèle et le scaler
model = joblib.load('implementation/models/best_model_mlp.joblib')
scaler = joblib.load('implementation/models/scaler.joblib')

# Prédiction
features = np.load('path/to/features.npy')
features_scaled = scaler.transform(features)
predictions = model.predict(features_scaled)
```

### 9.4 API de Prédiction (Exemple)

```python
def predict_class(image_features: np.ndarray) -> dict:
    """
    Prédit la classe d'un produit à partir de ses features.

    Args:
        image_features: array (1280,) - features EfficientNet

    Returns:
        dict avec class_idx, class_code, confidence
    """
    features_scaled = scaler.transform(image_features.reshape(1, -1))

    # Prédiction
    class_idx = model.predict(features_scaled)[0]

    # Probabilités (si disponible)
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(features_scaled)[0]
        confidence = proba[class_idx]
    else:
        confidence = None

    return {
        'class_idx': int(class_idx),
        'class_code': idx_to_class[class_idx],
        'confidence': confidence
    }
```

---

## 10. Troubleshooting

### 10.1 Problèmes Courants

#### Erreur : ConvergenceWarning (Logistic Regression)

```
Solution : Augmenter max_iter
model = LogisticRegression(max_iter=2000)
```

#### Erreur : MemoryError (Random Forest)

```
Solution : Réduire n_estimators ou max_depth
model = RandomForestClassifier(n_estimators=100, max_depth=20)
```

#### Erreur : MLP ne converge pas

```
Solutions :
1. Normaliser les données (StandardScaler)
2. Réduire learning_rate_init
3. Augmenter max_iter
4. Activer early_stopping
```

### 10.2 Performance Faible

| Symptôme | Cause Probable | Solution |
|----------|----------------|----------|
| Val Acc << Train Acc | Overfitting | Régularisation, réduire complexité |
| Val Acc ≈ Train Acc mais faible | Underfitting | Modèle plus complexe, plus de features |
| Classes minoritaires mal prédites | Déséquilibre | class_weight, SMOTE |

### 10.3 Temps d'Exécution

| Modèle | Temps Typique | Optimisation |
|--------|---------------|--------------|
| LogReg | 5-10 min | solver='sag' pour grands datasets |
| RF | 2-5 min | n_jobs=-1, réduire n_estimators |
| MLP | 3-5 min | GPU (PyTorch), batch_size plus grand |
| XGBoost | 1-3 min | tree_method='gpu_hist' |

---

## Annexes

### A. Fichiers Générés

```
implementation/
├── models/
│   ├── best_model_mlp.joblib      # Meilleur modèle
│   ├── model_logistic_regression.joblib
│   ├── model_random_forest.joblib
│   ├── model_mlp.joblib
│   └── scaler.joblib              # StandardScaler
├── outputs/
│   ├── classification_results.json # Résultats détaillés
│   └── test_predictions.csv       # Prédictions test
└── docs/
    └── BIBLE_CLASSIFICATION_IMAGES.md
```

### B. Dépendances

```
numpy>=1.21
pandas>=1.3
scikit-learn>=1.0
joblib>=1.0
matplotlib>=3.4
seaborn>=0.11

# Optionnel
xgboost>=1.5
lightgbm>=3.3
```

### C. Références

1. Scikit-learn Documentation: https://scikit-learn.org/
2. XGBoost Documentation: https://xgboost.readthedocs.io/
3. "Hands-On Machine Learning" - Aurélien Géron
4. "Pattern Recognition and Machine Learning" - Christopher Bishop

---

**Document généré le** : 2025-12-12
**Version** : 1.0
**Auteur** : Projet Rakuten - Classification de Produits
