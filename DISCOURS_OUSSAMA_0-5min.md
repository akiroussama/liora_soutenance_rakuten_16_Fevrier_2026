# DISCOURS OUSSAMA — 0:00 a 7:00 (version WOW)

> Fichier a lire tel quel pendant la soutenance.
> Les **[ACTIONS]** indiquent ce que tu fais a l'ecran.
> Les textes en gras sont les phrases a prononcer.
> Duree cible : **7 minutes**. Rythme soutenu, pas de temps mort.

---

## 0:00 — LE HOOK (30 secondes)

**[ECRAN : Slide de couverture]**
**[DEBOUT, regard direct au jury]**

**"Imaginez : vous etes responsable catalogue chez Rakuten France. Chaque jour, des milliers de produits arrivent. Un vendeur poste une jaquette bleue avec le titre 'Le Grand Bleu'. C'est un DVD ? Un accessoire piscine ? Un poster ?**

**Un humain met 5 minutes par produit, avec 10 a 15% d'erreur. Nous, on le fait en moins d'une seconde, sur 100 000 produits par jour.**

**Bonjour, nous sommes l'equipe Rakuten."**

> *Pourquoi ce hook :* Le jury est happe par un probleme concret. Pas de "bonjour, je m'appelle...". On pose le probleme AVANT de se presenter.

---

## 0:30 — EQUIPE (30 secondes)

**[ECRAN : toujours slide de couverture]**

**"Je suis Oussama — j'ai pris en charge la partie image et le deploiement de l'application que vous allez voir dans 30 secondes.**

**Johan, a ma droite, a construit le Voting System — un systeme de vote entre 3 architectures d'IA.**

**Michael a nettoye et prepare les donnees texte — le carburant de nos modeles.**

**Liviu, data scientist senior, a developpe le NLP et la strategie de fusion.**

**Notre mentor est Antoine. Notre mission : 85 000 produits, 27 categories, texte plus image."**

---

## 1:00 — LIVE DEMO : ACCUEIL (20 secondes)

**[ACTION : Basculer sur le navigateur — http://localhost:8502 deja ouvert]**

> Le jury voit : 4 metriques en gros (84 916 / 27 / Texte+Image / F1~0.85), le pipeline, les 27 categories.

**"Voici notre application, deployee en production sur Hugging Face Spaces. Les chiffres : 84 916 produits d'entrainement, 27 categories, F1-score 0.85 en fusion multimodale.**

**Maintenant... la preuve en direct."**

**[ACTION : Cliquer "Classifier un produit" → Page Demo]**

---

## 1:20 — DEMO TEXTE (30 secondes)

**[ECRAN : Page Demo > Onglet "Analyse Texte"]**

**"Test numero 1 : le texte seul."**

**[ACTION : Taper "Harry Potter edition poche"]**
**[ACTION : Cliquer "Analyser le Texte"]**

> Le jury voit : mots-cles surlignés (harry, potter, edition, poche), barre 99%, top 5.

**"Le modele capte les mots-cles — harry, potter, poche — et predit : Livres, 99% de confiance. Zero hesitation. Quand le texte est clair, 10 millisecondes suffisent."**

---

## 1:50 — DEMO IMAGE + CONSEIL DES SAGES (50 secondes)

**[ACTION : Cliquer onglet "Analyse Image"]**
**[ACTION : Upload la photo de bouee piscine]**

**"Test numero 2 : l'image seule, sans aucun texte. Une bouee de piscine."**

**[ACTION : Cliquer "Lancer le Voting Image"]**

> Le jury voit : Le "Conseil des Sages" — 3 colonnes DINOv3 (poids 4, 57%), EfficientNet (poids 2, 28%), XGBoost (poids 1, 14%). Prediction Piscine.

**"Et la, c'est la ou ca devient interessant. On ne fait pas tourner UN modele, mais TROIS.**

**DINOv3, le patron — un Vision Transformer de Meta. Il voit l'image dans sa globalite. Poids 4 sur 7.**

**EfficientNet, l'expert des details — un CNN qui capte les textures. Poids 2.**

**XGBoost, le statisticien — du machine learning classique sur des features extraites. Poids 1. Son role : corriger quand les deux autres hesitent.**

**Trois architectures, trois facons de 'voir' une image. Resultat : Piscine.**

**C'est notre Conseil des Sages."**

---

## 2:40 — DEMO FUSION : LE MOMENT CLE (55 secondes)

**[ACTION : Cliquer onglet "FUSION Multimodale"]**

> Le jury voit : slider 60% image / 40% texte, barres visuelles.

**"Test numero 3 — et c'est celui-ci qui justifie tout le projet."**

**[ACTION : Taper "DVD Le Grand Bleu Luc Besson"]**
**[ACTION : Upload la jaquette bleue]**
**[ACTION : Cliquer "Calculer la Fusion"]**

**"Je tape 'DVD Le Grand Bleu Luc Besson' et j'uploade cette jaquette. Elle est bleue. L'image seule pourrait confondre avec une piscine.**

**Mais la fusion combine les deux : le texte dit 'DVD', 'Luc Besson'. L'image dit 'objet bleu'. Le texte corrige. Resultat : DVD. Correct.**

**C'est ca, la complementarite multimodale. L'un sauve l'autre."**

**[ACTION : Bouger le slider vers 90% image]**

**"Regardez : si je pousse l'image a 90%... la prediction change. Le texte perd son pouvoir correcteur.**

**[ACTION : Remettre le slider a 60%]**

**60/40 — c'est l'equilibre qu'on a calibre. Le texte a 83% est en fait notre modalite la plus fiable. L'image a 79% apporte la robustesse. Ensemble : F1 0.85."**

---

## 3:35 — EXPLICABILITE : "COMMENT LE MODELE VOIT" (55 secondes)

**[ACTION : Cliquer "Explicabilite" dans le menu lateral gauche]**

> Le jury voit : Page Explicabilite, Tab "Vision (Grad-CAM)".
> Image explainability_drive.png : 4 colonnes (original, DINOv3 attention, EfficientNet activation, XGBoost features).

**"Un F1 de 0.85 ne suffit pas. Il faut prouver que le modele comprend — et pas qu'il triche.**

**Voici du Grad-CAM et des Attention Maps. Chaque colonne, c'est un de nos 3 modeles. Meme image, mais regardez : ils ne regardent PAS les memes zones."**

**[ACTION : Pointer/montrer les colonnes a l'ecran]**

**"DINOv3, le Transformer, capte la structure globale — l'objet entier. EfficientNet, le CNN, se concentre sur les textures et les bords. XGBoost analyse des features statistiques — dimensions, couleurs dominantes.**

**Des erreurs differentes, des forces differentes. C'est POUR CA que le vote fonctionne."**

**[ACTION : Scroller vers "Focus Battle" si le temps le permet]**

> Le jury voit : focus_battle.png — 4 quadrants, meme produit, DINOv3 90.7%, EfficientNet 77.8%, XGBoost 31.7%.

**"Exemple concret : DINOv3 est a 90% de confiance, EfficientNet a 77%, XGBoost seulement 31%. XGBoost se trompe — mais avec un poids de 1 sur 7, il ne contamine pas le vote. Le systeme est anti-fragile."**

---

## 4:30 — PERFORMANCE + PRODUCTION (30 secondes)

**[ACTION : Cliquer "Performance" dans le menu lateral]**

> Le jury voit : 5 metriques en haut (F1 0.83 / F1~0.79 / F1~0.85 / 40+ / 27)

**[ACTION : Cliquer onglet "Benchmark CPU/GPU"]**

> Le jury voit : 3 graphiques — CPU (gris), GPU (rouge), acceleration (vert x24).

**"En production : DINOv3 passe de 2 secondes en CPU a 82 millisecondes en GPU — facteur x24.**

**Le Voting complet : 170 millisecondes par produit. Soit 500 000 classifications par jour sur un seul serveur GPU."**

---

## 5:00 — QUALITE & RIGUEUR (45 secondes)

**[ACTION : Cliquer "Qualite" dans le menu lateral]**

> Le jury voit : 5 metriques (210+ tests / 85% couverture / 50+ securite / 40+ ML / < 2 min), pyramide des tests, gauges Accuracy 79.3 / F1 78.5.

**"On ne deploie pas un modele sans filet. Notre pipeline de qualite : 210 tests — unitaires, integration, performance, ET securite OWASP. 85% de couverture de code.**

**En bas, nos Quality Gates ML : accuracy image 79.3%, F1 macro 78.5%. Six gates sur six en vert."**

**[PAUSE — regarder le jury]**

**"Un point d'honneur : en cours de projet, Johan a decouvert un data leakage sur l'evaluation image. Les scores etaient gonfles — le Voting affichait 92% au lieu de 79%. On a TOUT corrige. Chaque fichier, chaque slide, chaque metrique. C'est la difference entre un projet academique et une demarche d'ingenieur."**

> *Ce moment est crucial — il montre l'honnetete intellectuelle et la rigueur. Les jurys ADORENT ca.*

---

## 5:45 — IMPACT BUSINESS (35 secondes)

**[ACTION : Cliquer "Conclusions" dans le menu lateral]**
**[ACTION : Scroller vers "Impact Business"]**

> Le jury voit : 2 colonnes Avant/Apres + ROI chart.

**"Impact business. Avant : un operateur met 5 minutes par produit, 10 a 15% d'erreur. Apres : moins d'une seconde, 27 categories, avec un seuil de confiance a 80%.**

**Les 70% de produits au-dessus du seuil sont classes automatiquement. Les 30% restants partent en revue humaine. Zero automatisation aveugle — le systeme sait dire 'je ne suis pas sur'."**

---

## 6:20 — TRANSITION VERS MICHAEL (40 secondes)

**[ECRAN : rester sur Conclusions]**
**[REGARD : se tourner vers le jury, puis vers Michael]**

**"Recapitulons ce que vous venez de voir en 6 minutes :**

**Un systeme qui classifie en temps reel — texte, image, ou les deux. Un Voting de 3 architectures complementaires. De l'explicabilite avec Grad-CAM et SHAP. Un pipeline de 210 tests. Et une correction de data leakage en cours de projet.**

**Tout ca, c'est le resultat. La question : comment on construit ca ?**

**Michael va vous plonger dans la matiere premiere — 85 000 produits, 5 langues, 35% de descriptions manquantes, et un desequilibre de 13 contre 1 entre les classes. C'est la que tout commence."**

**[GESTE : passer la parole a Michael]**

---

## 7:00 — FIN. Michael prend la parole.

---

## TABLEAU RECAPITULATIF DES 7 MINUTES

| Chrono | Section | Page Streamlit | Duree | WOW Factor |
|--------|---------|---------------|-------|------------|
| 0:00 | Hook probleme concret | Slide cover | 30s | Accroche emotionnelle |
| 0:30 | Equipe rapide | Slide cover | 30s | Confiance, roles clairs |
| 1:00 | Accueil app | Accueil | 20s | Chiffres en gros |
| 1:20 | Demo Texte | Demo > Texte | 30s | 99% confiance, instantane |
| 1:50 | Demo Image + Voting | Demo > Image | 50s | Conseil des Sages, 3 modeles |
| 2:40 | Demo Fusion + Slider | Demo > Fusion | 55s | Texte corrige l'image, live |
| 3:35 | Grad-CAM & Attention | Explicabilite | 55s | Le modele "voit" — visuel fort |
| 4:30 | Benchmark GPU | Performance | 30s | x24, 500K/jour |
| 5:00 | Qualite + Data Leakage | Qualite | 45s | 210 tests + honnetete |
| 5:45 | Impact Business | Conclusions | 35s | 5min → <1sec |
| 6:20 | Recap + Transition | Conclusions | 40s | Boucle fermee, suspense |

---

## PRODUITS DE DEMO (prepares a l'avance sur le Bureau)

| # | Texte | Image | Resultat attendu | Pourquoi ce cas |
|---|-------|-------|----------|-----------------|
| 1 | "Harry Potter edition poche" | *(pas d'image)* | Livres 99% | Cas facile — texte decisif |
| 2 | *(pas de texte)* | Photo bouee piscine | Piscine ~79% | Image decisive — montre le Voting |
| 3 | "DVD Le Grand Bleu Luc Besson" | Jaquette bleue | DVD 87% | Texte corrige l'image — fusion |

---

## CHECKLIST AVANT LA DEMO

- [ ] Streamlit lance et modeles charges (**lancer 10 min avant**, premier chargement = 30-60 sec)
- [ ] 2 images preparees sur le Bureau :
  - `bouee_piscine.jpg` (pour le test Image)
  - `jaquette_dvd_bleu.jpg` (pour le test Fusion)
- [ ] Navigateur ouvert sur http://localhost:8502, page Accueil visible
- [ ] URL de backup : https://huggingface.co/spaces/akiroussama/rakuten-classifier
- [ ] Faire un test rapide avant (1 prediction texte) pour confirmer que les modeles sont charges
- [ ] Verifier que les images Grad-CAM/Explicabilite sont presentes (page Explicabilite > Tab Vision)

## SI LA DEMO PLANTE

> **Scenario 1 — Delai de chargement :** "Le systeme charge les modeles — ca prend quelques secondes. Pendant ce temps, laissez-moi vous montrer l'architecture sur les slides."
> Basculer sur les slides HTML (PRESENTATION_RAKUTEN_SOUTENANCE.html).
>
> **Scenario 2 — Crash complet :** Ouvrir la version Hugging Face en backup. Si meme HF ne repond pas, rester sur les slides et dire : "L'application est deployee sur Hugging Face, je vous montrerai apres la presentation si le reseau revient."
>
> **Regle d'or :** Ne JAMAIS s'excuser plus de 5 secondes. Pivoter, continuer, sourire.

---

## LES 5 MOMENTS WOW (a jouer a fond)

1. **0:00 — Le Hook** : Le jury ne s'attend pas a un probleme concret en ouverture. Ca casse le pattern "bonjour je m'appelle".

2. **2:40 — Fusion "Le Grand Bleu"** : Le moment ou le texte sauve l'image. Manipuler le slider en live. Le jury VOIT la decision changer.

3. **3:35 — Grad-CAM** : Les 3 modeles ne regardent pas la meme zone. C'est visuel, c'est spectaculaire, ca montre une vraie comprehension du Deep Learning.

4. **5:00 — Data Leakage** : "On a decouvert un probleme, on a TOUT corrige." L'honnetete intellectuelle impressionne plus que des scores gonfles.

5. **6:20 — La boucle** : On a ouvert avec "jaquette bleue — DVD ou piscine ?" et on ferme en ayant MONTRE que le systeme sait repondre. Le jury retient cette coherence.
