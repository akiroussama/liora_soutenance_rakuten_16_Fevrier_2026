# SPÉCIFICATION TECHNIQUE - APPLICATION STREAMLIT RAKUTEN
## Objectif : Félicitations du Jury - Score 20/20

---

## TABLE DES MATIÈRES

1. [Analyse des Critères d'Évaluation](#1-analyse-des-critères-dévaluation)
2. [Architecture de l'Application](#2-architecture-de-lapplication)
3. [Spécification Détaillée par Page](#3-spécification-détaillée-par-page)
4. [Checklist Qualité](#4-checklist-qualité)
5. [Script de Démonstration](#5-script-de-démonstration)
6. [Anticipation des Questions du Jury](#6-anticipation-des-questions-du-jury)

---

## 1. ANALYSE DES CRITÈRES D'ÉVALUATION

### 1.1 Critères Explicites du Mentor

| Critère | Poids | Notre Réponse |
|---------|-------|---------------|
| **Application esthétique** | ÉLEVÉ | Design Rakuten professionnel, CSS custom |
| **Plusieurs onglets** | OBLIGATOIRE | 6+ pages structurées |
| **Code propre** | ÉLEVÉ | Commentaires, architecture modulaire |
| **Sans ré-entraîner le modèle** | OBLIGATOIRE | Modèles pré-chargés (.joblib) |
| **Fonctionne sans bugs** | CRITIQUE | Tests exhaustifs, fallbacks |

### 1.2 Critères Implicites (Ce qui fait la différence)

| Critère Implicite | Impact | Notre Approche |
|-------------------|--------|----------------|
| **Narration business** | TRÈS ÉLEVÉ | Chaque page liée à un enjeu métier |
| **Rigueur scientifique** | ÉLEVÉ | Métriques, matrices de confusion, intervalles |
| **Originalité** | MOYEN | Comparaison multi-modèles, visualisations interactives |
| **Professionnalisme** | ÉLEVÉ | UX fluide, temps de chargement courts |
| **Maîtrise technique** | ÉLEVÉ | Réponses précises aux questions |

### 1.3 Grille d'Évaluation Anticipée

```
EXCELLENT (18-20) : Application fluide, storytelling clair,
                    rigueur scientifique, zéro bug, questions maîtrisées

TRÈS BIEN (15-17) : Application fonctionnelle, contenu complet,
                    quelques hésitations sur les questions

BIEN (12-14)      : Application basique, manque de polish,
                    bugs mineurs, lacunes techniques

INSUFFISANT (<12) : Bugs bloquants, manque de contenu,
                    incompréhension du projet
```

---

## 2. ARCHITECTURE DE L'APPLICATION

### 2.1 Structure des Pages (6 onglets)

```
📁 src/streamlit/
├── app.py                           # Page d'accueil (IMPACT VISUEL)
├── pages/
│   ├── 1_📊_Données.py              # Exploration & DataViz
│   ├── 2_⚙️_Preprocessing.py        # Pipeline de traitement
│   ├── 3_🧠_Modèles.py              # Résultats & Comparaisons
│   ├── 4_🔍_Démo.py                 # Classification interactive
│   ├── 5_📈_Performance.py          # Métriques détaillées
│   └── 6_💡_Conclusions.py          # Business insights & Perspectives
```

### 2.2 Flow Narratif de la Présentation

```
┌─────────────────────────────────────────────────────────────────┐
│  ACCUEIL (1 min)                                                │
│  "Rakuten : 85K produits, 27 catégories, enjeu business"        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  DONNÉES (3 min)                                                │
│  "Dataset multimodal, déséquilibre des classes, multilinguisme" │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PREPROCESSING (3 min)                                          │
│  "Nettoyage texte, détection langue, extraction features CNN"   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  MODÈLES (5 min)                                                │
│  "3 modèles texte, 3 modèles image, comparaison rigoureuse"     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  DÉMO LIVE (4 min)                                              │
│  "Classification en temps réel, comparaison des modèles"        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PERFORMANCE (2 min)                                            │
│  "Matrices de confusion, F1 par classe, analyse des erreurs"    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  CONCLUSIONS (2 min)                                            │
│  "Impact business, limites, perspectives MLOps"                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. SPÉCIFICATION DÉTAILLÉE PAR PAGE

### 3.1 PAGE ACCUEIL (app.py) - "L'Effet WOW"

**Objectif** : Captiver l'attention en 30 secondes

**Éléments obligatoires** :

| Élément | Spécification | Justification |
|---------|---------------|---------------|
| **Header Rakuten** | Logo officiel + barre rouge | Crédibilité corporate |
| **Titre impactant** | "Classification Automatique de Produits" | Clarté immédiate |
| **4 Métriques clés** | 84,916 produits / 27 catégories / 6 modèles / 85%+ accuracy | Chiffres impressionnants |
| **Pipeline visuel** | Schéma Texte → Modèle / Image → Modèle | Compréhension instantanée |
| **Grille des 27 catégories** | Emojis + noms courts | Aperçu du problème |
| **CTA principal** | "Tester la Classification" | Call-to-action clair |

**CSS obligatoire** :
```css
/* Palette Rakuten */
--rakuten-red: #BF0000;
--rakuten-dark: #8B0000;
--background: #FAFAFA;
--text-primary: #333333;
--text-secondary: #666666;

/* Aucun dark mode - thème clair uniquement */
```

**Temps de chargement** : < 2 secondes

---

### 3.2 PAGE DONNÉES (1_📊_Données.py)

**Objectif** : Démontrer la maîtrise du dataset

**Sections obligatoires** :

#### 3.2.1 Vue d'ensemble
```python
col1, col2, col3, col4 = st.columns(4)
# Métriques : Train size, Test size, Features, Classes
```

#### 3.2.2 Distribution des catégories
- **Bar chart horizontal** : 27 barres, triées par fréquence
- **Camembert** : Top 10 + "Autres"
- **Tableau** : Code, Nom, Count, Pourcentage

#### 3.2.3 Analyse du déséquilibre
```
Ratio max/min : ~15x
Classe majoritaire : Livres (X produits)
Classe minoritaire : [catégorie] (Y produits)
```

#### 3.2.4 Analyse textuelle
- Longueur moyenne designation vs description
- Distribution des langues (FR dominant)
- Wordcloud par catégorie (optionnel)

#### 3.2.5 Exemples de produits
- Sélecteur de catégorie
- 3-5 exemples avec designation + description

**Graphiques Plotly obligatoires** :
- `px.bar()` pour distribution
- `px.pie()` pour proportions
- `px.histogram()` pour longueurs de texte

---

### 3.3 PAGE PREPROCESSING (2_⚙️_Preprocessing.py)

**Objectif** : Justifier chaque choix technique

**Sections obligatoires** :

#### 3.3.1 Pipeline Texte (schéma interactif)
```
┌─────────┐    ┌──────────┐    ┌───────────┐    ┌─────────┐
│ Texte   │ -> │ Nettoyage│ -> │ Détection │ -> │ Traduct.│
│ brut    │    │ HTML/spec│    │ langue    │    │ → FR    │
└─────────┘    └──────────┘    └───────────┘    └─────────┘
                                                      │
                                                      ▼
                               ┌───────────┐    ┌─────────┐
                               │ TF-IDF    │ <- │ Tokeniz.│
                               │ Vectors   │    │ Lemma   │
                               └───────────┘    └─────────┘
```

#### 3.3.2 Pipeline Image
```
┌─────────┐    ┌──────────┐    ┌───────────┐    ┌─────────┐
│ Image   │ -> │ Resize   │ -> │ Normalize │ -> │ResNet50 │
│ brute   │    │ 224x224  │    │ ImageNet  │    │ Features│
└─────────┘    └──────────┘    └───────────┘    └─────────┘
                                                      │
                                                      ▼
                                              ┌───────────┐
                                              │ 2048-dim  │
                                              │ vector    │
                                              └───────────┘
```

#### 3.3.3 Démo interactive du preprocessing
```python
# Input utilisateur
raw_text = st.text_area("Texte brut")
# Output
st.json({
    "original": raw_text,
    "cleaned": cleaned_text,
    "language": detected_lang,
    "translated": translated_text,
    "tokens": tokens,
    "tfidf_shape": (1, vocab_size)
})
```

#### 3.3.4 Justifications techniques
| Choix | Justification | Alternative considérée |
|-------|---------------|------------------------|
| TF-IDF | Interprétable, rapide | Word2Vec (essayé) |
| ResNet50 | Pré-entraîné ImageNet, bon compromis | VGG16 (plus lourd) |
| Traduction FR | Dataset majoritairement FR | Embeddings multilingues |

---

### 3.4 PAGE MODÈLES (3_🧠_Modèles.py)

**Objectif** : Comparer rigoureusement les approches

**Sections obligatoires** :

#### 3.4.1 Tableau récapitulatif
```
| Modèle              | Type  | Accuracy | F1 Macro | Temps Train |
|---------------------|-------|----------|----------|-------------|
| TF-IDF + SVM        | Texte | 78%      | 75%      | 2 min       |
| TF-IDF + RF         | Texte | 75%      | 72%      | 5 min       |
| CamemBERT           | Texte | 85%      | 82%      | 45 min      |
| ResNet50 + SVM      | Image | 72%      | 68%      | 10 min      |
| ResNet50 + RF       | Image | 70%      | 66%      | 15 min      |
| VGG16 + SVM         | Image | 68%      | 64%      | 12 min      |
```

#### 3.4.2 Graphique comparatif
- Bar chart groupé : Accuracy vs F1 par modèle
- Radar chart : Précision, Rappel, F1, Vitesse, Robustesse

#### 3.4.3 Analyse du meilleur modèle
```
Meilleur Texte : CamemBERT (85% accuracy)
- Points forts : Compréhension contextuelle
- Points faibles : Temps d'inférence

Meilleur Image : ResNet50 + SVM (72% accuracy)
- Points forts : Vitesse, features riches
- Points faibles : Limité sans texte
```

#### 3.4.4 Pourquoi pas de multimodal ?
> "Après expérimentation, la fusion tardive (late fusion) n'a pas
> amélioré significativement les performances par rapport au meilleur
> modèle texte seul. Le texte contient l'essentiel de l'information
> discriminante pour ce dataset."

---

### 3.5 PAGE DÉMO (4_🔍_Démo.py) - CRITIQUE

**Objectif** : Prouver que ça fonctionne EN LIVE

**Onglets obligatoires** :

#### 3.5.1 Classification Simple
- Input : Texte OU Image
- Sélection du modèle
- Résultat : Top-5 avec confiances
- Graphique des probabilités

#### 3.5.2 Comparaison des 3 Modèles
- Un input → 3 résultats côte à côte
- Tableau comparatif
- Graphique radar
- Badge "MEILLEUR"

#### 3.5.3 Galerie d'Exemples
- 9 exemples pré-définis (1 par grande catégorie)
- Un clic = résultat instantané
- Montre la diversité du modèle

#### 3.5.4 Historique de Session
- Sidebar avec les dernières classifications
- Statistiques de session

**Points critiques pour la démo** :
```
✅ TEMPS DE RÉPONSE < 1 seconde
✅ AUCUN SPINNER qui dure
✅ RÉSULTATS COHÉRENTS (mots-clés → bonnes catégories)
✅ FALLBACK si erreur (message user-friendly)
```

---

### 3.6 PAGE PERFORMANCE (5_📈_Performance.py)

**Objectif** : Rigueur scientifique

**Sections obligatoires** :

#### 3.6.1 Métriques globales
```
┌────────────┬────────────┬────────────┬────────────┐
│ Accuracy   │ F1 Macro   │ Precision  │ Recall     │
│   84.7%    │   82.3%    │   83.1%    │   81.5%    │
└────────────┴────────────┴────────────┴────────────┘
```

#### 3.6.2 Matrice de confusion
- Heatmap interactive (Plotly)
- Option : valeurs normalisées
- Zoom sur les confusions fréquentes

#### 3.6.3 Performance par catégorie
- Bar chart : F1 par classe
- Tri : du meilleur au pire
- Identification des classes difficiles

#### 3.6.4 Analyse des erreurs
```
Top 3 confusions :
1. "Livres" ↔ "Magazines" (12% d'erreurs)
2. "Jeux vidéo" ↔ "Accessoires gaming" (8%)
3. "Vêtements" ↔ "Accessoires mode" (7%)

Explication : Similarité sémantique des descriptions
```

#### 3.6.5 Courbes d'apprentissage
- Loss train vs validation
- Accuracy au fil des epochs
- Détection d'overfitting

---

### 3.7 PAGE CONCLUSIONS (6_💡_Conclusions.py)

**Objectif** : Vision business + ouverture

**Sections obligatoires** :

#### 3.7.1 Résumé des résultats
```
✅ Objectif atteint : Classification automatique à 85%
✅ Meilleur modèle : CamemBERT (texte)
✅ Dataset maîtrisé : 27 classes, 85K produits
```

#### 3.7.2 Impact business
```
AVANT                          APRÈS
─────────────────────────────────────────────────
Classification manuelle        Classification auto
~5 min/produit                 <1 sec/produit
Erreur humaine ~10%            Erreur modèle ~15%
Non scalable                   Millions/jour possible
```

#### 3.7.3 Limites identifiées
```
⚠️ Classes minoritaires moins bien classées
⚠️ Images seules insuffisantes (texte prépondérant)
⚠️ Dépendance à la qualité des descriptions vendeurs
```

#### 3.7.4 Perspectives / Ouverture
```
COURT TERME (1-3 mois)
├── Data augmentation pour classes minoritaires
├── Ensemble learning (voting des 6 modèles)
└── Seuil de confiance pour revue humaine

MOYEN TERME (3-6 mois)
├── Fine-tuning CamemBERT sur le domaine e-commerce
├── Modèle multimodal CLIP
└── Active learning pour amélioration continue

LONG TERME (MLOps)
├── Pipeline CI/CD avec MLflow
├── Monitoring de drift
├── A/B testing en production
```

#### 3.7.5 Ce que nous avons appris
> Point personnel de chaque membre de l'équipe

---

## 4. CHECKLIST QUALITÉ

### 4.1 Avant la Soutenance (J-3)

```
□ Tous les onglets fonctionnent sans erreur
□ Temps de chargement < 3 sec par page
□ Tous les graphiques s'affichent correctement
□ Les modèles sont pré-chargés (pas de training)
□ Le CSS est cohérent sur toutes les pages
□ Les textes sont relus (pas de fautes)
□ Le flow narratif est fluide
□ Les transitions entre pages sont testées
```

### 4.2 Check Technique

```
□ python -m py_compile *.py (0 erreur)
□ streamlit run app.py (démarre sans warning)
□ Test sur Chrome, Firefox, Edge
□ Test avec différentes résolutions d'écran
□ Données de démo fonctionnelles
□ Fallback si modèle absent
□ Messages d'erreur user-friendly
```

### 4.3 Check Présentation

```
□ Timing répété : 20 min pile
□ Chaque membre parle
□ Transitions préparées ("Passons maintenant à...")
□ Questions anticipées (voir section 6)
□ Plan B si bug (screenshot, vidéo backup)
□ Connexion internet stable
□ Micro/Caméra testés
```

### 4.4 Le Jour J

```
□ Redémarrer l'application 10 min avant
□ Fermer toutes les autres applications
□ Mode "Ne pas déranger" activé
□ URL Streamlit partagé dans le chat
□ Second écran avec notes si besoin
□ Verre d'eau à portée
```

---

## 5. SCRIPT DE DÉMONSTRATION (4 min)

### 5.1 Intro (30 sec)
> "Nous allons maintenant voir l'application en action. Notre objectif :
> montrer qu'un opérateur Rakuten peut classifier un produit en quelques
> secondes avec une confiance élevée."

### 5.2 Démo 1 : Classification simple (1 min)
```
1. Aller sur "Démo"
2. Saisir : "Harry Potter à l'école des sorciers, roman fantastique"
3. Cliquer "Classifier"
4. Montrer : Top-5, confiance, graphique
5. Commenter : "87% de confiance sur Livres, cohérent"
```

### 5.3 Démo 2 : Comparaison des modèles (1 min 30)
```
1. Aller sur "Comparaison Modèles"
2. Saisir : "Coque iPhone motif floral"
3. Cliquer "Comparer les 3 modèles"
4. Montrer : Les 3 résultats côte à côte
5. Commenter : "CamemBERT est le plus confiant,
                les 3 modèles convergent vers la même catégorie"
6. Montrer le radar chart
```

### 5.4 Démo 3 : Galerie d'exemples (1 min)
```
1. Cliquer sur 3 exemples variés (Livre, Console, Piscine)
2. Montrer la rapidité (<1 sec)
3. Montrer l'historique qui se remplit
4. Commenter : "Le modèle généralise bien
                sur des catégories très différentes"
```

### 5.5 Conclusion démo (30 sec)
> "Comme vous pouvez le voir, notre solution est rapide, fiable,
> et prête pour une mise en production. Passons maintenant aux
> métriques de performance détaillées."

---

## 6. ANTICIPATION DES QUESTIONS DU JURY

### 6.1 Questions Techniques Probables

| Question | Réponse clé |
|----------|-------------|
| "Pourquoi TF-IDF plutôt que Word2Vec ?" | "Interprétabilité, performance équivalente sur ce dataset, moins de ressources" |
| "Comment gérez-vous le déséquilibre ?" | "Class weights dans SVM, oversampling exploré mais peu d'amélioration" |
| "Pourquoi ResNet50 ?" | "Compromis features/taille, pré-entraîné ImageNet, 2048 dimensions" |
| "Temps d'inférence en production ?" | "~50ms pour TF-IDF+SVM, ~200ms pour CamemBERT" |
| "Pourquoi pas de multimodal ?" | "Expérimenté, gain marginal, complexité accrue" |

### 6.2 Questions Business Probables

| Question | Réponse clé |
|----------|-------------|
| "ROI estimé ?" | "~90% réduction temps classification, ~X ETP économisés" |
| "Et si le modèle se trompe ?" | "Seuil de confiance + revue humaine sous 70%" |
| "Scalabilité ?" | "Pipeline batch possible, ~100K produits/jour sur 1 GPU" |
| "Mise en production ?" | "API REST avec FastAPI, conteneur Docker, monitoring MLflow" |

### 6.3 Questions Pièges

| Question | Réponse (honnête) |
|----------|-------------------|
| "Quelle est la vraie accuracy sur données récentes ?" | "Nous n'avons pas de données post-2020, drift possible" |
| "Avez-vous testé CLIP ?" | "Non par manque de temps, c'est une perspective" |
| "Le modèle est-il biaisé ?" | "Potentiellement sur classes minoritaires, à surveiller" |

---

## 7. CRITÈRES DE FÉLICITATION DU JURY

Pour obtenir les félicitations, votre application doit démontrer :

### 7.1 Excellence Technique
- [x] Code propre, modulaire, commenté
- [x] Architecture professionnelle
- [x] Gestion des erreurs
- [x] Performance optimisée

### 7.2 Rigueur Scientifique
- [x] Métriques appropriées (pas juste accuracy)
- [x] Analyse des erreurs
- [x] Comparaison de modèles
- [x] Reproductibilité

### 7.3 Vision Business
- [x] Lien constant avec la problématique Rakuten
- [x] Chiffres d'impact
- [x] Limites identifiées
- [x] Perspectives réalistes

### 7.4 Qualité de Présentation
- [x] Storytelling clair
- [x] Timing respecté
- [x] Réponses précises
- [x] Travail d'équipe visible

### 7.5 Le "Plus" qui fait la différence
- [x] Une fonctionnalité originale (comparaison multi-modèles)
- [x] Design professionnel (niveau startup)
- [x] Démo fluide sans accroc
- [x] Autocritique constructive

---

## 8. ACTIONS PRIORITAIRES

### Immédiat (Aujourd'hui)
1. [ ] Vérifier que toutes les pages existent et fonctionnent
2. [ ] Tester le flow complet de la démo
3. [ ] Corriger tout bug bloquant

### Court terme (J-2)
4. [ ] Remplir les pages avec les vraies données/métriques
5. [ ] Répéter la présentation (timing)
6. [ ] Préparer les réponses aux questions

### Veille de la soutenance (J-1)
7. [ ] Test final complet
8. [ ] Backup (video, screenshots)
9. [ ] Repos et préparation mentale

---

**Document préparé le** : $(date)
**Équipe** : RAKUTEN - Formation BMLE Oct 2025
**Objectif** : 🏆 FÉLICITATIONS DU JURY
