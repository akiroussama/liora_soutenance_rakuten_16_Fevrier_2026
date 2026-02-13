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

2. **Effet d'ancrage (Anchoring Bias).** Quand le jury VOIT F1~0.85 fonctionner en live, chaque explication technique qui suit est percue comme "la preuve que ca marche". Sans la demo d'abord, les slides techniques sont abstraites — le jury se demande "ok mais est-ce que ca marche vraiment ?".

3. **Gestion du risque technique.** Si la demo plante a la minute 17 (reseau, modele qui charge pas), on a perdu le moment fort ET gache les 2 dernieres minutes. En premier : si ca marche, c'est l'euphorie. Si ca plante, il reste 15 min de technique solide pour compenser.

4. **Format TED Talk / Steve Jobs.** On montre le resultat ("voici ce qu'on a construit"), puis on explique comment. Le jury passe de "impressionne" a "curieux" — ils VEULENT savoir comment on a fait. C'est bien plus engageant que : slides, slides, slides, demo, merci.

5. **Differenciation.** 99% des groupes font : intro → technique → technique → demo a la fin → merci. En ouvrant par la demo, on casse le pattern. Le jury a vu 10 soutenances ce jour-la — la notre commence differemment, ils s'en souviendront.

---

## PLAN MINUTE PAR MINUTE (20 min)

**Logique narrative :** Demo (le WOW) → Donnees (la matiere premiere) → Image (diversite architecturale, ~79%) → Texte (champion 83%) + Fusion + Conclusion (on assemble tout, F1~0.85, on ferme la boucle)

---

### 1. OUSSAMA — 0:00 a 5:00 (5 min) — "Le Hook"

| Chrono | Ecran Streamlit | Ce qu'on voit | Ce que tu dis | Duree |
|--------|-----------------|---------------|---------------|-------|
| 0:00 | *Slides* Cover | Logo Rakuten, equipe, mentor | Intro equipe + mission | 40s |
| 0:40 | *Slides* Sommaire | Plan 4 parties, fil rouge | Teasing: "F1~0.85, en live" | 20s |
| 1:00 | **Accueil** (app.py) | 4 metriques (84 916 / 27 / Texte+Image / F1~0.85), pipeline Texte→Image→Fusion, grille 27 categories avec emojis | "Voici notre application deployee. 84 916 produits, 27 categories, F1~0.85 en fusion." | 25s |
| 1:25 | **Page Demo** > Onglet Texte | Zone de saisie, bouton "Analyser" | Taper "Harry Potter edition poche" → Prediction Livres (99%) + top 5 avec barres de confiance + mots-cles surlignés | 35s |
| 2:00 | **Page Demo** > Onglet Image | Upload image, bouton "Lancer le Voting" | Upload photo bouee piscine → Le Conseil des Sages : DINOv3 (poids 4), EffNet (poids 2), XGBoost (poids 1) avec barres visuelles → Prediction Piscine | 45s |
| 2:45 | **Page Demo** > Onglet Fusion | Slider 60/40, zone texte + image | "DVD Le Grand Bleu" + jaquette bleue → Le texte corrige l'image → Prediction DVD correcte. Puis bouger le slider → montrer l'impact live des poids | 45s |
| 3:30 | **Page Demo** > Architecture | Expander "Architecture du systeme" : schema explainability_drive.png + model_accuracy_comparison.png | "Le Voting combine 3 architectures pour 79% en image. Le texte a 83% est notre meilleure modalite. En fusion : F1~0.85." | 30s |
| 4:00 | **Page Performance** > Tab Benchmark | Graphique benchmark_cpu_gpu.png : CPU vs GPU, x24 acceleration DINOv3 | "En production GPU, le systeme classe un produit en 170ms. Soit 100 000 produits par jour sur un seul serveur." | 25s |
| 4:25 | Transition | — | "Vous avez vu le resultat final. Michael va maintenant vous plonger dans la matiere premiere : les donnees brutes et les defis du preprocessing." | 35s |
| 5:00 | — | — | **→ Michael prend la parole** | — |

---

#### SCRIPT DETAILLE OUSSAMA (mot a mot, 5 min)

**[0:00 — COVER — debout, regard au jury]**

> "Bonjour. Nous sommes l'equipe Rakuten. Notre mission : classifier automatiquement 85 000 produits e-commerce en 27 categories, en combinant texte et image.
>
> Je suis Oussama, j'ai travaille sur la partie image et le deploiement de l'application. A ma droite, Johan, notre architecte image — c'est lui qui a construit le Voting System a 3 modeles. Michael a prepare les donnees et tout le preprocessing texte. Et Liviu, data scientist senior, a developpe les modeles NLP et la strategie de fusion.
>
> Notre mentor est Antoine, qui nous a guides tout au long du projet."

**[0:40 — SOMMAIRE — pointer l'ecran]**

> "Notre presentation suit une logique simple : je commence par vous montrer le resultat en live, puis Michael explique les donnees, Johan detaille la partie image, et Liviu conclut avec le texte, la fusion, et l'explicabilite. Avant toute theorie... laissez-moi vous montrer ce qu'on a construit."

**[1:00 — DEMO LIVE — ouvrir Streamlit, page Accueil]**

> *[Ecran : page Accueil avec les 4 metriques en haut, le pipeline en 3 colonnes, et la grille des 27 categories]*
>
> "Voici notre application deployee sur Hugging Face. En haut : les chiffres cles — 84 916 produits, 27 categories, deux modalites texte et image, et F1~0.85 en fusion. En dessous, notre pipeline : le texte passe par un nettoyage et TF-IDF, l'image par DINOv3, et la fusion combine les deux. Et voila nos 27 categories — des livres aux piscines, des jeux video au bricolage."
>
> *[Cliquer sur "Classifier un produit" → Page Demo]*

**[1:25 — DEMO TEXTE — Onglet "Analyse Texte"]**

> *[Ecran : zone de saisie a gauche, resultats a droite]*
>
> "Premier test : le texte seul. Je tape 'Harry Potter edition poche'..."
>
> *[Taper le texte, cliquer "Analyser le Texte"]*
>
> "Le modele identifie les mots-cles — 'harry', 'potter', 'edition', 'poche' — et predit avec 99% de confiance : Livres. Le top 5 confirme : aucune hesitation. Cas facile."

**[2:00 — DEMO IMAGE — Onglet "Analyse Image"]**

> *[Cliquer sur l'onglet Image, uploader la photo de bouee]*
>
> "Deuxieme test : l'image seule. J'uploade une photo de bouee de piscine, sans aucun texte."
>
> *[Cliquer "Lancer le Voting Image" — apparition du Conseil des Sages : DINOv3 poids 4, EffNet poids 2, XGBoost poids 1 avec barres de progression]*
>
> "Regardez l'architecture du vote : notre 'Conseil des Sages'. DINOv3, le patron, a poids 4 — c'est un Vision Transformer qui voit le contexte global. EfficientNet, l'expert, poids 2 — un CNN qui capte les details. XGBoost, le statisticien, poids 1 — il corrige. Ensemble, ils votent : Piscine. Cas ou l'image suffit."

**[2:45 — DEMO FUSION — Onglet "FUSION Multimodale"]**

> *[Cliquer sur l'onglet Fusion. Montrer le slider a 60% image / 40% texte]*
>
> "Troisieme test : la fusion. C'est le cas le plus interessant. Je tape 'DVD Le Grand Bleu Luc Besson' et j'uploade une jaquette bleue."
>
> *[Entrer le texte et l'image, cliquer "Calculer la Fusion"]*
>
> "L'image seule hesiterait — c'est bleu, ca pourrait etre une piscine. Mais le texte dit 'DVD Luc Besson'. La fusion corrige : DVD. C'est exactement la complementarite texte-image."
>
> *[Bouger le slider vers 90% image]*
>
> "Si je pousse l'image a 90%... le texte perd son influence. C'est pour ca qu'on a calibre a 60/40 — le meilleur equilibre."

**[3:30 — ARCHITECTURE VISUELLE — Expander sur la page Demo]**

> *[Ouvrir l'expander "Architecture du systeme de classification" → affiche le schema explainability + le graphique accuracy]*
>
> "Sous le capot : voici comment les 3 modeles du Voting analysent chaque image differemment. Le Voting combine ces 3 visions pour 79% en image. Le texte a 83% est notre meilleure modalite. En fusion : F1~0.85."

**[4:00 — BENCHMARK RAPIDE — Page Performance, onglet Benchmark]**

> *[Naviguer vers Page Performance > onglet Benchmark CPU/GPU — benchmark_cpu_gpu.png visible]*
>
> "Un mot sur la production. DINOv3 prend 2 secondes en CPU, mais 82 millisecondes en GPU — acceleration de x24. Le Voting complet : moins de 200ms par produit. Soit plus de 100 000 produits par jour sur un seul serveur."

**[4:25 — TRANSITION]**

> "Vous venez de voir le systeme en action : texte, image, fusion. F1~0.85, en temps reel. La question maintenant : comment on y arrive ? Michael va vous plonger dans la matiere premiere — les donnees brutes, le desequilibre des classes, et le preprocessing qui rend tout ca possible."
>
> *[Regard vers Michael, geste de la main]*

---

#### ECRANS STREAMLIT — RESUME VISUEL (ce que le jury voit)

| Moment | Page | Elements visuels marquants |
|--------|------|---------------------------|
| 1:00 | Accueil | 4 metriques en gros, pipeline 3 colonnes, 27 emojis |
| 1:25 | Demo > Texte | Mots-cles surlignés `harry` `potter`, barre confiance 99%, top 5 DataFrame |
| 2:00 | Demo > Image | Photo bouee, 3 colonnes Conseil des Sages (DINOv3/EffNet/XGBoost) avec poids et barres |
| 2:45 | Demo > Fusion | Slider interactif 60/40, barres laterales, resultat fusionne |
| 3:30 | Demo > Architecture | Schema explicabilite (explainability_drive.png), comparaison accuracy (model_accuracy_comparison.png) |
| 4:00 | Performance > Benchmark | 3 graphiques : CPU (gris), GPU (rouge), Acceleration (vert x24) |

#### PRODUITS DE DEMO (prepares a l'avance)

| # | Texte | Image | Resultat | Pourquoi ce cas |
|---|-------|-------|----------|-----------------|
| 1 | "Harry Potter edition poche" | *(pas d'image)* | Livres 99% | Cas facile — le texte suffit |
| 2 | *(pas de texte)* | Photo bouee piscine | Piscine ~79% | Image decisive — montre le Voting |
| 3 | "DVD Le Grand Bleu Luc Besson" | Jaquette bleue | DVD 87% | Texte corrige l'image — fusion |

**Objectif minute 5 :** Le jury est impressionne. Il a VU le systeme fonctionner sur 3 cas concrets. Il a vu l'architecture, les poids, le benchmark. Maintenant il VEUT comprendre comment ca marche.

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
| 11:00 | Modelisation Image | Benchmark : RF (0.71), XGBoost GPU (0.72), **XGBoost features optimisees (85.32%)**, DINOv3+MLP (79.43%, 58s). XGBoost = champion individuel | 1 min 30 |
| 12:30 | Voting System | Architecture 3 modeles : DINOv3+MLP (4/7), EfficientNet (2/7), XGBoost calibre (1/7). Complementarite architecturale (ViT vs CNN vs ML). Sharpening p^3/sum(p^3). Resultat : **~79%** (robustesse) | 2 min 30 |
| 15:00 | — | **Transition → Liviu** | — |

**Script Johan :**
> "Cote image, on resize a 224x224 avec normalisation ImageNet, puis on extrait des features via DINOv3, un Vision Transformer self-supervised de Meta, et EfficientNet-B0. Le transfer learning est la cle.

> J'ai benchmarke tous les classifieurs : Random Forest a 71%, DINOv3 avec un MLP en tete atteint 79.43% en 58 secondes. Mais la surprise : XGBoost sur les features ResNet optimisees monte a 85.32% — le ML classique n'est pas mort.

> Le Voting System combine 3 classifieurs complementaires. DINOv3 est un Transformer — il voit le contexte global, poids 4/7. EfficientNet est un CNN — il capte les textures locales, poids 2/7. XGBoost travaille sur les features tabulaires, independant des reseaux de neurones, poids 1/7. Sa particularite : le 'sharpening', qui eleve ses probabilites au cube pour qu'il tranche net au lieu de diluer le vote. Ces 3 modeles font des erreurs differentes. Ensemble : ~79%. Le Voting apporte la robustesse architecturale. Liviu va maintenant vous montrer comment on combine ca avec le texte pour atteindre F1~0.85."

**Objectif :** Le jury a vu l'architecture image ET la diversite. Il est pret pour la fusion.

---

### 4. LIVIU — 15:00 a 20:00 (5 min) — "Le Strategiste"

| Chrono | Slide | Contenu | Duree |
|--------|-------|---------|-------|
| 15:00 | Modelisation Texte | Benchmark : LogReg (0.79), NB (0.76), **LinearSVC (0.83)**, RF (0.68). C=0.5 optimise par GridSearch, class_weight='balanced'. Pourquoi SVM plutot que CamemBERT (non teste, justification litterature) | 1 min 30 |
| 16:30 | Fusion Multimodale | Late Fusion : Image ~79% x 0.6 + Texte 83% x 0.4 = **F1~0.85**. Complementarite des modalites. Exemple : DVD bleu corrige par le texte. **Maintenant le jury connait les deux composants !** | 1 min 30 |
| 18:00 | Explicabilite | SHAP (global texte), Grad-CAM (heatmap image). Conformite AI Act. On sait POURQUOI le modele decide | 1 min |
| 19:00 | Conclusion & Perspectives | Bilan : F1~0.85 en fusion, systeme explicable, deploye sur Streamlit. Perspectives : OCR (+3-5%), CamemBERT, fine-tuning DINOv3, monitoring en production | 1 min |
| 20:00 | — | **FIN — Merci, questions ?** | — |

**Script Liviu :**
> "Avec les 280K features de Michael, j'ai benchmarke 4 classifieurs. Le champion est LinearSVC a 83% de F1-Score. Pourquoi un SVM lineaire plutot que CamemBERT ? En 280 000 dimensions, les donnees sont lineairement separables — c'est le terrain ideal du SVM. C=0.5 optimise par GridSearch, en 10 millisecondes par produit.

> Maintenant vous connaissez les deux briques : l'Image de Johan a ~79%, et mon Texte a 83%. La fusion combine les deux en late fusion : 60% image, 40% texte. Quand l'image confond un DVD bleu avec une piscine, le texte 'DVD Le Grand Bleu' corrige. Resultat : F1~0.85.

> Pour l'explicabilite : SHAP montre quels mots comptent, Grad-CAM montre ou le modele regarde dans l'image. On respecte les exigences de transparence du AI Act europeen.

> En conclusion : nous avons un systeme multimodal a F1~0.85, explicable, deploye en production sur Streamlit. Les perspectives : OCR pour extraire le texte des images (+3-5% potentiel), fine-tuning de DINOv3, et un pipeline de monitoring pour detecter le drift en production. Merci pour votre attention, nous sommes prets pour vos questions."

**Objectif :** La fusion est expliquee APRES les deux composants (logique !). La presentation se ferme sur une ouverture business. Le jury retient : F1~0.85, explicable, deploye, avec une roadmap.

---

## RECAPITULATIF

| Ordre | Membre | Minutes | Slides couvertes |
|-------|--------|---------|-----------------|
| 1er | **Oussama** | 0:00-5:00 | Cover, Sommaire, **DEMO LIVE (3 min 30)** |
| 2eme | **Michael** | 5:00-10:00 | Challenge Rakuten, Desequilibre, Preprocessing Texte |
| 3eme | **Johan** | 10:00-15:00 | Preprocessing Image, Modelisation Image, **Voting System (~79%)** |
| 4eme | **Liviu** | 15:00-20:00 | Modelisation Texte, **Fusion (F1~0.85)**, Explicabilite, **Conclusion** |

> **Logique narrative :** Demo (WOW) → Donnees (la matiere premiere) → Image/Voting (diversite, ~79%) → Texte (champion 83%) + Fusion + Conclusion (on assemble tout, F1~0.85, on ferme la boucle). La Fusion arrive APRES que les deux composants aient ete expliques — le jury peut suivre la logique.

---

## TRANSITIONS (phrase exacte a dire)

| Qui | Phrase |
|-----|--------|
| Oussama → Michael | "Vous avez vu le resultat. Michael va vous montrer la matiere premiere : les donnees et le preprocessing." |
| Michael → Johan | "Les donnees sont pretes. Johan va s'attaquer au plus gros morceau : la partie image et le Voting System." |
| Johan → Liviu | "~79% en image grace au Voting, 83% en texte. Liviu va completer le puzzle avec la fusion pour atteindre F1~0.85." |
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
| 2 | "Accessoire bleu pour exterieur" | Photo bouee piscine | Piscine (~79%) | Image decisive (texte trop vague) |
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
