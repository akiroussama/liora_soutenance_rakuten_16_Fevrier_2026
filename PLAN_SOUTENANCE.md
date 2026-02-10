# Plan de Soutenance - Classification Multimodale Rakuten
## RNCP Machine Learning Engineer - DataScientest x Mines Paris-PSL
### Promotion Octobre 2025 - Soutenance Février 2026

---

## Equipe & Rôles

| Membre | Travail réalisé | Rôle Soutenance |
|--------|----------------|-----------------|
| **Oussama Akir** | Image + Présentation | MC : Ouverture, Démo Live, Conclusion |
| **Johan Frachon** | Image (tout le pipeline, Voting) | Pipeline Image, Voting System |
| **Michael (Hery M.)** | Texte (preprocessing, données) | Contexte, Données, Preprocessing Texte |
| **Liviu Andronic** | NLP + Expert technique (DS Senior) | Modélisation NLP, Fusion, Explicabilité |

---

## Format : 20 min Présentation + 10 min Q&A
## Règle : 5 minutes CHRONO par personne

---

## PLAN MINUTE PAR MINUTE (20 min)

---

### OUSSAMA — 0:00 à 5:00 (5 min)

| Chrono | Slide | Contenu | Durée |
|--------|-------|---------|-------|
| 0:00 | Cover | Présenter l'équipe et le projet, mentor Antoine | 40s |
| 0:40 | Sommaire | Plan, fil rouge, métriques finales annoncées | 20s |
| 1:00 | — | **DEMO LIVE STREAMLIT** | **3 min** |
| 1:00 | Streamlit Accueil | Ouvrir l'app, montrer les métriques globales | 30s |
| 1:30 | Streamlit Données | Distribution des 27 classes, déséquilibre visible | 20s |
| 1:50 | Streamlit Texte | Saisir "iPhone 15 Pro Max 256Go" → Prédiction texte (83%) | 30s |
| 2:20 | Streamlit Image | Upload photo produit → Voting visible (3 modèles, 92%) | 40s |
| 2:40 | Streamlit Fusion | Texte + Image → Résultat ~94% | 30s |
| 3:10 | Streamlit Slider | Varier le ratio image/texte → montrer l'impact live | 30s |
| 3:40 | Streamlit | Montrer un cas où texte et image se corrigent | 20s |
| 4:00 | Conclusion | Résultats : 92% image, 83% texte, ~94% fusion | 30s |
| 4:30 | Perspectives | OCR (+3-5%), CamemBERT, fine-tuning DINOv3, Cloud | 30s |
| 5:00 | — | **Transition → Michael** | — |

**Script Oussama :**
> "Bonjour, nous sommes l'équipe Rakuten. Notre mission : classifier automatiquement 85 000 produits e-commerce en 27 catégories en combinant texte et image. Je suis Oussama, j'ai travaillé sur la partie image. Johan est notre architecte image, il a construit le Voting System. Michael a préparé les données texte. Et Liviu, data scientist senior, a développé les modèles NLP et la fusion. Avant la théorie, je vous montre le résultat final en action."

> *[DEMO 3 min — montrer les 3 cas : facile, image décisive, texte décisif]*

> "Vous venez de voir ~94% d'accuracy en live. Michael va maintenant vous expliquer les données et le preprocessing qui rendent ça possible."

---

### MICHAEL — 5:00 à 10:00 (5 min)

| Chrono | Slide | Contenu | Durée |
|--------|-------|---------|-------|
| 5:00 | Challenge Rakuten | Contexte Rakuten France, 84 916 produits, 27 catégories, 5 langues, 35% descriptions manquantes, métrique F1 pondéré | 2 min |
| 7:00 | Déséquilibre | Ratio 13:1 (Piscines 12% → Consoles 0.9%), solutions : class_weight, F1 pondéré, stratified split | 1 min |
| 8:00 | Preprocessing Texte | Nettoyage HTML → Concaténation designation+description → TF-IDF 280K features (word 1-2 + char 3-5). Pourquoi TF-IDF > Word2Vec. Pas de lemmatisation (marques) | 2 min |
| 10:00 | — | **Transition → Liviu** | — |

**Script Michael :**
> "Le challenge Rakuten, c'est 85 000 produits avec des descriptions souvent sales — du HTML, 5 langues, et 35% de descriptions manquantes. Le déséquilibre est majeur : 13 fois plus de piscines que de consoles. On a compensé avec class_weight='balanced' et le F1-Score pondéré. Pour le texte, j'ai construit un pipeline qui nettoie et vectorise en 280 000 dimensions via TF-IDF, avec des n-grams de mots et de caractères pour capter les marques comme iPhone ou PlayStation. On ne lemmatise pas pour préserver ces marques. Liviu va maintenant vous montrer les modèles NLP."

---

### LIVIU — 10:00 à 15:00 (5 min)

| Chrono | Slide | Contenu | Durée |
|--------|-------|---------|-------|
| 10:00 | Modélisation Texte | Benchmark : LogReg (0.79), NB (0.76), **LinearSVC (0.83)**, RF (0.68). C=0.5 optimisé par GridSearch, class_weight='balanced'. Pourquoi SVM > CamemBERT en haute dimension | 2 min |
| 12:00 | Fusion Multimodale | Late Fusion : Image 92% × 0.6 + Texte 83% × 0.4 = **~94%**. Complémentarité des modalités. Exemple : DVD bleu corrigé par le texte | 1 min 30 |
| 13:30 | Explicabilité | SHAP (global texte), LIME (local texte), Grad-CAM (heatmap image). Conformité AI Act. On sait POURQUOI le modèle décide | 1 min 30 |
| 15:00 | — | **Transition → Johan** | — |

**Script Liviu :**
> "Avec les 280K features de Michael, j'ai benchmarké 4 classifieurs. Le champion est LinearSVC à 83% de F1-Score. Pourquoi un SVM linéaire bat CamemBERT ? En 280 000 dimensions, les données sont linéairement séparables — c'est le terrain idéal du SVM. C=0.5 optimisé par GridSearch, en 10 millisecondes par produit.

> Pour la fusion, on combine Image et Texte en 60/40. Quand l'image confond un DVD bleu avec une piscine, le texte 'DVD Le Grand Bleu' corrige. Résultat : ~94%.

> Enfin, l'explicabilité : SHAP montre quels mots comptent, Grad-CAM montre où le modèle regarde dans l'image. On respecte les exigences de transparence du AI Act européen. Johan va maintenant détailler la partie image."

---

### JOHAN — 15:00 à 20:00 (5 min)

| Chrono | Slide | Contenu | Durée |
|--------|-------|---------|-------|
| 15:00 | Preprocessing Image | Resize 224×224, normalisation ImageNet, extraction DINOv3 (1024 features) + EfficientNet-B0 (1280 features). Transfer learning, self-supervised ViT | 1 min |
| 16:00 | Modélisation Image | Benchmark : RF (0.71), XGBoost GPU (0.72), XGBoost Heavy **(0.765, 6h)**, **DINOv3+MLP (0.914, 58s)**. Le saut de 76.5% à 91.4% | 1 min 30 |
| 17:30 | Voting System | Architecture 3 modèles : DINOv3+MLP (4/7), EfficientNet (2/7), XGBoost calibré (1/7). Complémentarité architecturale (ViT vs CNN vs ML). Sharpening p³/Σp³. Résultat : **92%** | 2 min 30 |
| 20:00 | — | **FIN — Merci, questions ?** | — |

**Script Johan :**
> "Côté image, on resize à 224×224 avec normalisation ImageNet, puis on extrait des features via DINOv3, un Vision Transformer self-supervised de Meta, et EfficientNet-B0. Le transfer learning est la clé.

> J'ai benchmarké tous les classifieurs : Random Forest à 71%, XGBoost plafonne à 76.5% après 6 heures de GPU. Puis DINOv3 avec un MLP en tête explose tout : 91.4% en 58 secondes. C'est le moment eureka du projet.

> Mais je ne me suis pas arrêté là. Le Voting System combine 3 classifieurs complémentaires. DINOv3 est un Transformer — il voit le contexte global, poids 4/7. EfficientNet est un CNN — il capte les textures locales, poids 2/7. XGBoost travaille sur les features tabulaires, indépendant des réseaux de neurones, poids 1/7. Sa particularité : le 'sharpening', qui élève ses probabilités au cube pour qu'il tranche net au lieu de diluer le vote. Ces 3 modèles font des erreurs différentes. Ensemble : 92%. Merci pour votre attention, nous sommes prêts pour vos questions."

---

## RECAPITULATIF

| Ordre | Membre | Minutes | Slides couvertes |
|-------|--------|---------|-----------------|
| 1er | **Oussama** | 0:00-5:00 | Cover, Sommaire, **DEMO LIVE (3 min)**, Conclusion, Perspectives |
| 2ème | **Michael** | 5:00-10:00 | Challenge Rakuten, Déséquilibre, Preprocessing Texte |
| 3ème | **Liviu** | 10:00-15:00 | Modélisation Texte/NLP, Fusion Multimodale, Explicabilité |
| 4ème | **Johan** | 15:00-20:00 | Preprocessing Image, Modélisation Image, **Voting System** |

> **Logique de l'ordre :** Oussama ouvre et montre le résultat (démo). Michael explique les données. Liviu montre les modèles NLP et la fusion. Johan finit en apothéose avec le Voting System — c'est l'innovation majeure du projet, elle reste en dernier pour marquer le jury.

---

## TRANSITIONS (phrase exacte à dire)

| Qui | Phrase |
|-----|--------|
| Oussama → Michael | "Michael va maintenant vous expliquer les données et le preprocessing texte." |
| Michael → Liviu | "Les données sont prêtes. Liviu va vous montrer ce que les modèles NLP en font." |
| Liviu → Johan | "83% en texte, ~94% en fusion. Johan va conclure avec la partie image et le Voting System." |
| Johan (fin) | "Merci pour votre attention. Nous sommes prêts pour vos questions." |

---

## STRATEGIE Q&A (10 min)

| Sujet de la question | Répond en premier | Backup |
|---------------------|-------------------|--------|
| Données, déséquilibre, preprocessing texte | **Michael** | Liviu |
| Modèles texte, LinearSVC, NLP, TF-IDF | **Liviu** | Michael |
| Image, DINOv3, EfficientNet, CNN, Voting | **Johan** | Oussama |
| Fusion, architecture, explicabilité, AI Act | **Liviu** | Johan |
| Démo, Streamlit, déploiement, MLOps | **Oussama** | Liviu |
| Perspectives, améliorations futures | **Oussama** | Liviu |

**Règle d'or :** Chacun répond aux questions sur SA partie. Si on ne sait pas : "Excellente question. [Prénom], tu peux compléter ?"

---

## PRODUITS DE DEMO (à préparer avant)

| # | Texte à saisir | Image à uploader | Résultat attendu | Intérêt pédagogique |
|---|---------------|-----------------|-------------------|---------------------|
| 1 | "Harry Potter tome 1 édition poche" | Couverture du livre | Livres (99%) | Cas facile : texte + image convergent |
| 2 | "Accessoire bleu pour extérieur" | Photo bouée piscine | Piscine (92%) | Image décisive (texte trop vague) |
| 3 | "DVD Le Grand Bleu Luc Besson" | Jaquette bleue | DVD/Film (87%) | Texte décisif (image trompeuse = bleu) |

---

## CHECKLIST

### 1 semaine avant :
- [ ] Chacun connaît ses slides par coeur (pas de lecture)
- [ ] Préparer les 3 images de démo en local
- [ ] Relire les 60 questions Q&A

### 3 jours avant :
- [ ] **Répétition 1** sans chrono — fluidité des transitions
- [ ] Identifier les points faibles, reformuler

### La veille :
- [ ] **Répétition 2** avec chrono strict (5 min/personne)
- [ ] Tester Streamlit sur le poste de soutenance
- [ ] Vérifier chargement des modèles (1er lancement = 30-60 sec)
- [ ] Backup présentation en PDF

### Le jour J :
- [ ] Arriver 15 min en avance
- [ ] Lancer Streamlit et charger les modèles AVANT la soutenance
- [ ] Préparer les images dans un dossier sur le bureau
