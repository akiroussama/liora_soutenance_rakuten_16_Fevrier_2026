# RAPPORT DE SOUTENANCE - Projet Rakuten
## Classification Multimodale de Produits E-commerce

**Formation** : Machine Learning Engineer - DataScientest (Oct 2025)

**Équipe** :
- Johan FRACHON
- Liviu ANDRONIC
- Hery M. RALAIMANANTSOA
- Oussama AKIR

**Mentor** : Antoine

**Date de soutenance** : Semaine du 16 février 2026

---

## Table des matières

1. [Introduction](#1-introduction)
2. [Exploration des données](#2-exploration-des-données)
3. [Preprocessing & Feature Engineering](#3-preprocessing--feature-engineering)
4. [Modélisation Texte](#4-modélisation-texte)
5. [Modélisation Image](#5-modélisation-image)
6. [Résultats & Interprétabilité](#6-résultats--interprétabilité)
7. [Application Streamlit](#7-application-streamlit)
8. [Conclusion & Perspectives](#8-conclusion--perspectives)
9. [Annexes](#9-annexes)

---

## 1. Introduction
**[2 pages]**

### 1.1 Contexte métier
<!-- Challenge Rakuten France, classification automatique de produits e-commerce -->

### 1.2 Objectifs du projet
<!-- Classification en 27 catégories, approche multimodale texte+image -->

### 1.3 Présentation de l'équipe
<!-- Répartition des tâches : Johan (Image), Liviu (NLP), Hery (Modèles), Oussama (Intégration/App) -->

### 1.4 Organisation du projet
<!-- Méthodologie, outils (Git, Drive), planning -->

---

## 2. Exploration des données
**[4 pages]**

### 2.1 Description du dataset
<!--
- 84 916 images train
- 13 812 images test
- 27 catégories
- Taille moyenne : ~500×500 px
- Format : JPEG
-->

### 2.2 Analyse des variables
<!-- Colonnes : designation, description, productid, imageid, prdtypecode -->

### 2.3 Distribution des classes
<!--
- Déséquilibre important : ratio max/min = 13.4×
- Visualisation : barplot distribution
- Tableau des 27 catégories avec effectifs
-->

### 2.4 Analyse exploratoire des images
<!-- Exemples visuels, variabilité intra-classe, qualité des images -->

### 2.5 Analyse exploratoire du texte
<!-- Longueur des descriptions, langues, caractères spéciaux -->

### 2.6 Conclusions de l'exploration
<!-- Défis identifiés, stratégie adoptée -->

---

## 3. Preprocessing & Feature Engineering
**[5 pages]**

### 3.1 Preprocessing Texte

#### 3.1.1 Nettoyage
<!-- Lowercase, suppression accents, gestion NaN -->

#### 3.1.2 Feature Engineering
<!--
- TF-IDF Word (1-2 grams, max 120K features)
- TF-IDF Char (3-5 grams, max 160K features)
- FeatureUnion combinant les deux
- sublinear_tf=True
-->

### 3.2 Preprocessing Image

#### 3.2.1 Pipeline de transformation
<!--
Image brute (500×500×3)
    → Resize (224×224×3)
    → ToTensor [0,1]
    → Normalize (ImageNet mean/std)
    → EfficientNet-B0
    → Features (1×1280)
-->

#### 3.2.2 Transfer Learning
<!--
- Choix EfficientNet-B0 : 5.3M params, 77.1% ImageNet
- Feature extraction (modèle gelé)
- Compression : 750K → 1280 (facteur 586×)
-->

#### 3.2.3 Data Augmentation
<!--
- Oversampling des classes minoritaires
- Augmentation : 15K images/classe → 405K total
- Techniques : rotation, flip, color jitter
-->

### 3.3 Gestion du déséquilibre
<!-- class_weight, oversampling, stratégie adoptée -->

---

## 4. Modélisation Texte
**[4 pages]**

### 4.1 Approche baseline
<!-- TF-IDF simple + classifieurs basiques -->

### 4.2 Modèle final : TF-IDF + LinearSVC

#### 4.2.1 Architecture
```python
# Pipeline final
FeatureUnion([
    ("word", TfidfVectorizer(ngram_range=(1,2), max_features=120000)),
    ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), max_features=160000))
])
+ LinearSVC(C=0.5)
```

#### 4.2.2 Optimisation des hyperparamètres
<!-- Grid search, choix de C=0.5 -->

#### 4.2.3 Résultats
<!-- Accuracy, F1-score, temps d'entraînement -->

### 4.3 Alternatives explorées
<!-- Autres classifieurs testés, comparaison -->

---

## 5. Modélisation Image
**[6 pages]**

### 5.1 Stratégie globale
<!-- Transfer learning, feature extraction, comparaison architectures -->

### 5.2 Modèle M1 : DINOv3 (ViT-Large)
<!--
- Architecture Vision Transformer
- Score : 79.1%
- Avantages/inconvénients
-->

### 5.3 Modèle M2 : XGBoost sur features ResNet
<!--
- Extraction features ResNet50 (2048 dims)
- XGBoost classifier
- Score : 80.1% (Champion)
-->

### 5.4 Modèle M3 : EfficientNet-B0
<!--
- Fine-tuning léger
- Score : ~75%
- Rôle dans le voting
-->

### 5.5 Modèle M4 : ResNet Overfit (cas d'étude)
<!--
- Score train : 91%
- Score test : beaucoup plus bas
- Analyse de l'overfitting, leçons apprises
-->

### 5.6 Voting System (M5) - Modèle Final
<!--
- Principe de l'ensemble voting
- Pondération : DINO + XGBoost + EfficientNet
- Score final : ~79% (robustesse architecturale)
- Justification des poids
-->

### 5.7 Comparatif des modèles image
| Modèle | Accuracy | F1-Score | Params | Temps inference |
|--------|----------|----------|--------|-----------------|
| M1 DINOv3 | 79.43% | - | - | - |
| M2 XGBoost | **85.32%** | - | - | - |
| M3 EfficientNet | 66.63% | - | - | - |
| M4 ResNet (overfit) | 91%* | - | - | - |
| **M5 Voting** | **79.28%** | - | - | - |

---

## 6. Résultats & Interprétabilité
**[4 pages]**

### 6.1 Métriques globales
<!-- Accuracy, F1-score weighted, precision, recall -->

### 6.2 Matrice de confusion
<!-- Visualisation, analyse des erreurs fréquentes -->

### 6.3 Analyse par catégorie
<!-- Classes difficiles, confusions récurrentes -->

### 6.4 Interprétabilité

#### 6.4.1 SHAP (texte)
<!-- Feature importance, exemples -->

#### 6.4.2 Grad-CAM (image)
<!-- Visualisation des zones d'attention -->

### 6.5 Analyse des erreurs
<!-- Exemples de mauvaises prédictions, causes identifiées -->

---

## 7. Application Streamlit
**[3 pages]**

### 7.1 Architecture de l'application
```
src/streamlit/
├── app.py              # Page d'accueil
├── config.py           # Configuration paths et modèles
├── pages/
│   ├── 1_Données       # Stats du dataset
│   ├── 2_Preprocessing # Pipeline NLP et image
│   ├── 3_Modèles       # Comparaison des modèles
│   ├── 4_Démo          # Classification interactive
│   ├── 5_Performance   # Métriques
│   ├── 6_Conclusions   # Résultats
│   ├── 7_Qualité       # Tests
│   └── 8_Explicabilité # SHAP, Grad-CAM
├── utils/              # Code métier
└── tests/              # Tests pytest
```

### 7.2 Fonctionnalités
<!--
- Classification texte seul
- Classification image seule
- Classification multimodale (fusion)
- Comparaison de modèles
- Visualisations interactives
-->

### 7.3 Choix techniques
<!--
- Streamlit pour le prototypage rapide
- Plotly pour les visualisations
- Séparation config/code
- Gestion des modèles lourds (Drive)
-->

### 7.4 Screenshots
<!-- Captures d'écran des principales pages -->

---

## 8. Conclusion & Perspectives
**[2 pages]**

### 8.1 Bilan du projet
<!--
- Objectifs atteints : classification ~85% (fusion texte+image)
- Démarche scientifique : exploration → modélisation → production
- Travail d'équipe
-->

### 8.2 Difficultés rencontrées
<!--
- Déséquilibre des classes
- Taille des modèles (>3GB)
- Synchronisation des modèles pour le voting
- Overfitting
-->

### 8.3 Limites
<!--
- Fine-tuning end-to-end non exploré
- Pas de modèle BERT/CamemBERT pour le texte
- Fusion multimodale simple (late fusion)
-->

### 8.4 Perspectives
<!--
- Fine-tuning complet des modèles image
- Transformers pour le texte (CamemBERT)
- Early fusion multimodale
- Déploiement cloud (AWS/GCP)
- API REST pour intégration
-->

---

## 9. Annexes
**[3-5 pages]**

### Annexe A : Configuration technique
```python
# config.py - Extrait
MODEL_CONFIG = {
    "use_mock": False,
    "fusion_weights": (0.6, 0.4),  # image, texte
    "top_k": 5,
    "confidence_threshold": 0.1
}
```

### Annexe B : Mapping des catégories
<!-- 27 catégories avec codes et noms -->

### Annexe C : Requirements
```
# requirements.txt
streamlit
torch
torchvision
scikit-learn
xgboost
pandas
numpy
plotly
pillow
joblib
```

### Annexe D : Instructions de reproduction
```bash
# Installation
pip install -r requirements.txt

# Télécharger les modèles depuis Drive
# Placer dans /models :
# - M1_IMAGE_DeepLearning_DINOv3.pth
# - M2_IMAGE_Classic_XGBoost.json
# - M2_IMAGE_XGBoost_Encoder.pkl
# - M3_IMAGE_Classic_EfficientNetB0.pth
# - text_classifier.joblib

# Lancer l'application
streamlit run src/streamlit/app.py
```

### Annexe E : Liens utiles
- **Repository GitHub** : [lien]
- **Drive modèles** : [lien]
- **Données Rakuten** : [lien challenge]

---

## Informations sur ce document

| Élément | Valeur |
|---------|--------|
| **Pages recommandées** | 28-33 pages |
| **Format** | PDF |
| **Police** | 11-12pt |
| **Interligne** | 1.15-1.5 |
| **Deadline** | 11 février 2026 |

---

*Document généré le 30 janvier 2026*
*Structure validée par le comité d'experts IA*
