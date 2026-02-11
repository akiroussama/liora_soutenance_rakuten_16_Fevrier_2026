# Plan de Soutenance - Classification Multimodale Rakuten
## RNCP Machine Learning Engineer - DataScientest x Mines Paris-PSL
### Promotion Octobre 2025 - Soutenance Février 2026

---

## Equipe & Roles

| Membre | Travail realise | Role Soutenance |
|--------|----------------|-----------------|
| **Oussama Akir** | Image + Presentation + Deploiement | MC : Ouverture, Demo Live (3 min) |
| **Michael (Hery M.)** | Texte (preprocessing, donnees) | Contexte, Donnees, Preprocessing Texte |
| **Johan Frachon** | Image (tout le pipeline, Voting) | Pipeline Image, Voting System |
| **Liviu Andronic** | NLP + Expert technique (DS Senior) | Modelisation NLP, Fusion, Explicabilite, Conclusion |

---

## Format : 20 min Presentation + 10 min Q&A
## Regle : 5 minutes CHRONO par personne

---

## POURQUOI LA DEMO EN PREMIER ?

> **Pour convaincre vos camarades — voici les 5 arguments :**

1. **Courbe d'attention du jury.** L'attention est maximale dans les 3-5 premieres minutes, puis chute. Une demo live est la chose la plus engageante qu'on puisse montrer. La mettre a la minute 15 = la montrer a des cerveaux fatigues.

2. **Effet d'ancrage (Anchoring Bias).** Quand le jury VOIT 94% fonctionner en live, chaque explication technique qui suit est percue comme "la preuve que ca marche". Sans la demo d'abord, les slides techniques sont abstraites — le jury se demande "ok mais est-ce que ca marche vraiment ?".

3. **Gestion du risque technique.** Si la demo plante a la minute 17 (reseau, modele qui charge pas), on a perdu le moment fort ET gache les 2 dernieres minutes. En premier : si ca marche, c'est l'euphorie. Si ca plante, il reste 15 min de technique solide pour compenser.

4. **Format TED Talk / Steve Jobs.** On montre le resultat ("voici ce qu'on a construit"), puis on explique comment. Le jury passe de "impressionne" a "curieux" — ils VEULENT savoir comment on a fait. C'est bien plus engageant que : slides, slides, slides, demo, merci.

5. **Differenciation.** 99% des groupes font : intro → technique → technique → demo a la fin → merci. En ouvrant par la demo, on casse le pattern. Le jury a vu 10 soutenances ce jour-la — la notre commence differemment, ils s'en souviendront.

---

## PLAN MINUTE PAR MINUTE (20 min)

**Logique narrative :** Demo (le WOW) → Donnees (la matiere premiere) → Image (le gros morceau, 92%) → Texte + Fusion + Conclusion (on assemble tout, 94%, on ferme la boucle)

---

### 1. OUSSAMA — 0:00 a 5:00 (5 min) — "Le Hook"

| Chrono | Slide | Contenu | Duree |
|--------|-------|---------|-------|
| 0:00 | Cover | Presenter l'equipe et le projet, mentor Antoine | 40s |
| 0:40 | Sommaire | Plan de la presentation, fil rouge, teasing des metriques | 20s |
| 1:00 | — | **DEMO LIVE STREAMLIT** | **3 min 30** |
| 1:00 | Streamlit Accueil | Ouvrir l'app, montrer les metriques globales | 30s |
| 1:30 | Streamlit Donnees | Distribution des 27 classes, desequilibre visible | 20s |
| 1:50 | Streamlit Texte | Saisir "iPhone 15 Pro Max 256Go" → Prediction texte | 30s |
| 2:20 | Streamlit Image | Upload photo produit → Voting visible (3 modeles) | 40s |
| 3:00 | Streamlit Fusion | Texte + Image → Resultat ~94% | 30s |
| 3:30 | Streamlit | Cas ou texte et image se corrigent mutuellement | 30s |
| 4:00 | Streamlit Slider | Varier le ratio image/texte → montrer l'impact live | 30s |
| 4:30 | Transition | "Vous avez vu le resultat. Michael va vous montrer la matiere premiere." | 30s |
| 5:00 | — | **Transition → Michael** | — |

**Script Oussama :**
> "Bonjour, nous sommes l'equipe Rakuten. Notre mission : classifier automatiquement 85 000 produits e-commerce en 27 categories en combinant texte et image. Je suis Oussama, j'ai travaille sur la partie image et le deploiement. Johan est notre architecte image, il a construit le Voting System. Michael a prepare les donnees et le preprocessing texte. Et Liviu, data scientist senior, a developpe les modeles NLP et la fusion. Avant la theorie, je vous montre le resultat final en action."

> *[DEMO 3 min 30 — montrer les 3 cas : facile, image decisive, texte decisif]*

> "Vous venez de voir ~94% d'accuracy en live. Comment on y arrive ? Michael va vous expliquer les donnees brutes et le preprocessing qui rendent tout ca possible."

**Objectif :** Le jury est impressionne. Il a VU le systeme fonctionner. Maintenant il veut comprendre.

---

### 2. MICHAEL — 5:00 a 10:00 (5 min) — "La Realite du Terrain"

| Chrono | Slide | Contenu | Duree |
|--------|-------|---------|-------|
| 5:00 | Challenge Rakuten | Contexte Rakuten France, 84 916 produits, 27 categories, 5 langues, 35% descriptions manquantes, metrique F1 pondere | 2 min |
| 7:00 | Desequilibre | Ratio 13:1 (Piscines 12% → Consoles 0.9%), solutions : class_weight, F1 pondere, stratified split | 1 min |
| 8:00 | Preprocessing Texte | Nettoyage HTML → Concatenation designation+description → TF-IDF 280K features (word 1-2 + char 3-5). Pourquoi TF-IDF > Word2Vec. Pas de lemmatisation (marques) | 2 min |
| 10:00 | — | **Transition → Johan** | — |

**Script Michael :**
> "Le challenge Rakuten, c'est 85 000 produits avec des descriptions souvent sales — du HTML, 5 langues, et 35% de descriptions manquantes. Le desequilibre est majeur : 13 fois plus de piscines que de consoles. On a compense avec class_weight='balanced' et le F1-Score pondere. Pour le texte, j'ai construit un pipeline qui nettoie et vectorise en 280 000 dimensions via TF-IDF, avec des n-grams de mots et de caracteres pour capter les marques comme iPhone ou PlayStation. On ne lemmatise pas pour preserver ces marques. Les donnees sont pretes. Johan va maintenant s'attaquer au plus gros morceau : la partie image."

**Objectif :** Le jury comprend la difficulte du probleme (donnees sales, desequilibre, multilingue).

---

### 3. JOHAN — 10:00 a 15:00 (5 min) — "L'Artillerie Lourde"

| Chrono | Slide | Contenu | Duree |
|--------|-------|---------|-------|
| 10:00 | Preprocessing Image | Resize 224x224, normalisation ImageNet, extraction DINOv3 (1024 features) + EfficientNet-B0 (1280 features). Transfer learning, self-supervised ViT | 1 min |
| 11:00 | Modelisation Image | Benchmark : RF (0.71), XGBoost GPU (0.72), XGBoost Heavy **(0.765, 6h)**, **DINOv3+MLP (0.914, 58s)**. Le saut de 76.5% a 91.4% | 1 min 30 |
| 12:30 | Voting System | Architecture 3 modeles : DINOv3+MLP (4/7), EfficientNet (2/7), XGBoost calibre (1/7). Complementarite architecturale (ViT vs CNN vs ML). Sharpening p^3/sum(p^3). Resultat : **92%** | 2 min 30 |
| 15:00 | — | **Transition → Liviu** | — |

**Script Johan :**
> "Cote image, on resize a 224x224 avec normalisation ImageNet, puis on extrait des features via DINOv3, un Vision Transformer self-supervised de Meta, et EfficientNet-B0. Le transfer learning est la cle.

> J'ai benchmarke tous les classifieurs : Random Forest a 71%, XGBoost plafonne a 76.5% apres 6 heures de GPU. Puis DINOv3 avec un MLP en tete explose tout : 91.4% en 58 secondes. C'est le moment eureka du projet.

> Mais je ne me suis pas arrete la. Le Voting System combine 3 classifieurs complementaires. DINOv3 est un Transformer — il voit le contexte global, poids 4/7. EfficientNet est un CNN — il capte les textures locales, poids 2/7. XGBoost travaille sur les features tabulaires, independant des reseaux de neurones, poids 1/7. Sa particularite : le 'sharpening', qui eleve ses probabilites au cube pour qu'il tranche net au lieu de diluer le vote. Ces 3 modeles font des erreurs differentes. Ensemble : 92%. Liviu va maintenant vous montrer comment on combine ca avec le texte pour atteindre 94%."

**Objectif :** Le jury a vu l'architecture image ET le score de 92%. Il est pret pour la fusion.

---

### 4. LIVIU — 15:00 a 20:00 (5 min) — "Le Strategiste"

| Chrono | Slide | Contenu | Duree |
|--------|-------|---------|-------|
| 15:00 | Modelisation Texte | Benchmark : LogReg (0.79), NB (0.76), **LinearSVC (0.83)**, RF (0.68). C=0.5 optimise par GridSearch, class_weight='balanced'. Pourquoi SVM > CamemBERT en haute dimension | 1 min 30 |
| 16:30 | Fusion Multimodale | Late Fusion : Image 92% x 0.6 + Texte 83% x 0.4 = **~94%**. Complementarite des modalites. Exemple : DVD bleu corrige par le texte. **Maintenant le jury connait les deux composants !** | 1 min 30 |
| 18:00 | Explicabilite | SHAP (global texte), LIME (local texte), Grad-CAM (heatmap image). Conformite AI Act. On sait POURQUOI le modele decide | 1 min |
| 19:00 | Conclusion & Perspectives | Bilan : 94% en fusion, systeme explicable, deploye sur Streamlit. Perspectives : OCR (+3-5%), CamemBERT, fine-tuning DINOv3, monitoring en production | 1 min |
| 20:00 | — | **FIN — Merci, questions ?** | — |

**Script Liviu :**
> "Avec les 280K features de Michael, j'ai benchmarke 4 classifieurs. Le champion est LinearSVC a 83% de F1-Score. Pourquoi un SVM lineaire bat CamemBERT ? En 280 000 dimensions, les donnees sont lineairement separables — c'est le terrain ideal du SVM. C=0.5 optimise par GridSearch, en 10 millisecondes par produit.

> Maintenant vous connaissez les deux briques : l'Image de Johan a 92%, et mon Texte a 83%. La fusion combine les deux en late fusion : 60% image, 40% texte. Quand l'image confond un DVD bleu avec une piscine, le texte 'DVD Le Grand Bleu' corrige. Resultat : ~94%.

> Pour l'explicabilite : SHAP montre quels mots comptent, Grad-CAM montre ou le modele regarde dans l'image. On respecte les exigences de transparence du AI Act europeen.

> En conclusion : nous avons un systeme multimodal a 94%, explicable, deploye en production sur Streamlit. Les perspectives : OCR pour extraire le texte des images (+3-5% potentiel), fine-tuning de DINOv3, et un pipeline de monitoring pour detecter le drift en production. Merci pour votre attention, nous sommes prets pour vos questions."

**Objectif :** La fusion est expliquee APRES les deux composants (logique !). La presentation se ferme sur une ouverture business. Le jury retient : 94%, explicable, deploye, avec une roadmap.

---

## RECAPITULATIF

| Ordre | Membre | Minutes | Slides couvertes |
|-------|--------|---------|-----------------|
| 1er | **Oussama** | 0:00-5:00 | Cover, Sommaire, **DEMO LIVE (3 min 30)** |
| 2eme | **Michael** | 5:00-10:00 | Challenge Rakuten, Desequilibre, Preprocessing Texte |
| 3eme | **Johan** | 10:00-15:00 | Preprocessing Image, Modelisation Image, **Voting System (92%)** |
| 4eme | **Liviu** | 15:00-20:00 | Modelisation Texte, **Fusion (~94%)**, Explicabilite, **Conclusion** |

> **Logique narrative :** Demo (WOW) → Donnees (la matiere premiere) → Image/Voting (le gros morceau, 92%) → Texte + Fusion + Conclusion (on assemble tout, 94%, on ferme la boucle). La Fusion arrive APRES que les deux composants aient ete expliques — le jury peut suivre la logique.

---

## TRANSITIONS (phrase exacte a dire)

| Qui | Phrase |
|-----|--------|
| Oussama → Michael | "Vous avez vu le resultat. Michael va vous montrer la matiere premiere : les donnees et le preprocessing." |
| Michael → Johan | "Les donnees sont pretes. Johan va s'attaquer au plus gros morceau : la partie image et le Voting System." |
| Johan → Liviu | "92% en image grace au Voting. Liviu va completer le puzzle avec le texte et la fusion pour atteindre 94%." |
| Liviu (fin) | "Merci pour votre attention. Nous sommes prets pour vos questions." |

---

## STRATEGIE Q&A (10 min)

| Sujet de la question | Repond en premier | Backup |
|---------------------|-------------------|--------|
| Donnees, desequilibre, preprocessing texte | **Michael** | Liviu |
| Modeles texte, LinearSVC, NLP, TF-IDF | **Liviu** | Michael |
| Image, DINOv3, EfficientNet, CNN, Voting | **Johan** | Oussama |
| Fusion, architecture, explicabilite, AI Act | **Liviu** | Johan |
| Demo, Streamlit, deploiement, MLOps | **Oussama** | Liviu |
| Perspectives, ameliorations futures | **Oussama** | Liviu |

**Regle d'or :** Chacun repond aux questions sur SA partie. Si on ne sait pas : "Excellente question. [Prenom], tu peux completer ?"

---

## PRODUITS DE DEMO (a preparer avant)

| # | Texte a saisir | Image a uploader | Resultat attendu | Interet pedagogique |
|---|---------------|-----------------|-------------------|---------------------|
| 1 | "Harry Potter tome 1 edition poche" | Couverture du livre | Livres (99%) | Cas facile : texte + image convergent |
| 2 | "Accessoire bleu pour exterieur" | Photo bouee piscine | Piscine (92%) | Image decisive (texte trop vague) |
| 3 | "DVD Le Grand Bleu Luc Besson" | Jaquette bleue | DVD/Film (87%) | Texte decisif (image trompeuse = bleu) |

---

## CHECKLIST

### 1 semaine avant :
- [ ] Chacun connait ses slides par coeur (pas de lecture)
- [ ] Preparer les 3 images de demo en local
- [ ] Relire les 60 questions Q&A

### 3 jours avant :
- [ ] **Repetition 1** sans chrono — fluidite des transitions
- [ ] Identifier les points faibles, reformuler

### La veille :
- [ ] **Repetition 2** avec chrono strict (5 min/personne)
- [ ] Tester Streamlit sur le poste de soutenance
- [ ] Verifier chargement des modeles (1er lancement = 30-60 sec)
- [ ] Backup presentation en PDF

### Le jour J :
- [ ] Arriver 15 min en avance
- [ ] Lancer Streamlit et charger les modeles AVANT la soutenance
- [ ] Preparer les images dans un dossier sur le bureau
