# DISCOURS OUSSAMA — 0:00 a 5:00

> Fichier a lire tel quel pendant la soutenance.
> Les **[ACTIONS]** indiquent ce que tu fais a l'ecran.
> Les textes en gras sont les phrases a prononcer.

---

## 0:00 — COVER (40 secondes)

**[ECRAN : Slide de couverture]**

**"Bonjour. Nous sommes l'equipe Rakuten.**

**Je suis Oussama, j'ai travaille sur la partie image et le deploiement de l'application.**

**A ma droite, Johan, notre architecte image — c'est lui qui a construit le Voting System a 3 modeles.**

**Michael a prepare les donnees et tout le preprocessing texte.**

**Et Liviu, data scientist senior, a developpe les modeles NLP et la strategie de fusion.**

**Notre mentor est Antoine, qui nous a guides tout au long du projet.**

**Notre mission : classifier automatiquement 85 000 produits e-commerce en 27 categories, en combinant texte et image."**

---

## 0:40 — SOMMAIRE (20 secondes)

**[ECRAN : Slide sommaire]**

**"Notre presentation suit une logique simple : je commence par vous montrer le resultat en live, puis Michael explique les donnees, Johan detaille la partie image, et Liviu conclut avec le texte, la fusion, et l'explicabilite.**

**Avant toute theorie... laissez-moi vous montrer ce qu'on a construit."**

---

## 1:00 — PAGE ACCUEIL STREAMLIT (25 secondes)

**[ACTION : Ouvrir le navigateur sur http://localhost:8502 — Page Accueil]**

> Le jury voit : 4 metriques en gros (84 916 / 27 / Texte+Image / ~94%), le pipeline en 3 colonnes, et la grille des 27 categories avec emojis.

**"Voici notre application deployee sur Hugging Face Spaces.**

**En haut, les chiffres cles — 84 916 produits, 27 categories, deux modalites texte et image, et ~94% de precision globale.**

**En dessous, notre pipeline : le texte passe par un nettoyage et TF-IDF, l'image par DINOv3, et la fusion combine les deux.**

**Et voila nos 27 categories — des livres aux piscines, des jeux video au bricolage."**

**[ACTION : Cliquer sur le bouton "Classifier un produit" → Page Demo]**

---

## 1:25 — DEMO TEXTE (35 secondes)

**[ECRAN : Page Demo > Onglet "Analyse Texte" — zone de saisie a gauche, resultats a droite]**

**"Premier test : le texte seul."**

**[ACTION : Taper dans la zone : "Harry Potter edition poche"]**
**[ACTION : Cliquer "Analyser le Texte"]**

> Le jury voit : mots-cles surlignés (harry, potter, edition, poche), barre de confiance 99%, top 5 dans un tableau.

**"Le modele identifie les mots-cles — harry, potter, edition, poche — et predit avec 99% de confiance : Livres.**

**Le top 5 confirme : aucune hesitation. Cas facile — le texte suffit."**

---

## 2:00 — DEMO IMAGE (45 secondes)

**[ACTION : Cliquer sur l'onglet "Analyse Image"]**
**[ACTION : Cliquer "Browse files" et selectionner la photo de bouee piscine]**

> Le jury voit : l'apercu de la photo a gauche.

**"Deuxieme test : l'image seule. J'uploade une photo de bouee de piscine, sans aucun texte."**

**[ACTION : Cliquer "Lancer le Voting Image"]**

> Le jury voit : Le "Conseil des Sages" apparait — 3 colonnes avec DINOv3 (poids 4, barre 57%), EfficientNet (poids 2, barre 28%), XGBoost (poids 1, barre 14%). Puis le resultat de prediction.

**"Regardez l'architecture du vote : notre Conseil des Sages.**

**DINOv3, le patron, a poids 4 — c'est un Vision Transformer qui voit le contexte global.**

**EfficientNet, l'expert, poids 2 — un CNN qui capte les details fins.**

**XGBoost, le statisticien, poids 1 — il corrige.**

**Ensemble, ils votent : Piscine. C'est le cas ou l'image suffit."**

---

## 2:45 — DEMO FUSION (45 secondes)

**[ACTION : Cliquer sur l'onglet "FUSION Multimodale"]**

> Le jury voit : le slider a 60% image / 40% texte, les barres de poids, la zone texte et l'upload image.

**"Troisieme test : la fusion. C'est le cas le plus interessant."**

**[ACTION : Taper dans la zone texte : "DVD Le Grand Bleu Luc Besson"]**
**[ACTION : Cliquer "Browse files" et selectionner la jaquette bleue]**
**[ACTION : Cliquer "Calculer la Fusion"]**

**"Je tape 'DVD Le Grand Bleu Luc Besson' et j'uploade une jaquette bleue.**

**L'image seule hesiterait — c'est bleu, ca pourrait etre une piscine. Mais le texte dit 'DVD Luc Besson'. La fusion corrige : DVD.**

**C'est exactement la complementarite texte-image."**

**[ACTION : Bouger le slider vers 90% image]**

**"Si je pousse l'image a 90%... le texte perd son influence. C'est pour ca qu'on a calibre a 60/40 — le meilleur equilibre."**

---

## 3:30 — ARCHITECTURE VISUELLE (30 secondes)

**[ACTION : Scroller vers le haut de la page Demo]**
**[ACTION : Cliquer sur l'expander "Architecture du systeme de classification"]**

> Le jury voit : le schema d'explicabilite (comment les 3 modeles analysent l'image) + le graphique de comparaison d'accuracy des modeles.

**"Sous le capot : voici comment les 3 modeles du Voting analysent chaque image differemment.**

**DINOv3 regarde l'ensemble, EfficientNet les details, XGBoost les statistiques.**

**Et le resultat — le Voting a 92.4% surpasse chaque modele individuel. En fusionnant avec le texte : ~94%."**

---

## 4:00 — BENCHMARK RAPIDE (25 secondes)

**[ACTION : Cliquer sur "Performance" dans le menu lateral gauche]**
**[ACTION : Cliquer sur l'onglet "Benchmark CPU/GPU"]**

> Le jury voit : 3 graphiques cote a cote — temps CPU (gris), temps GPU (rouge), facteur d'acceleration (vert). DINOv3 passe de 2 secondes a 82ms (x24).

**"Un mot sur la production. DINOv3 prend 2 secondes en CPU, mais 82 millisecondes en GPU — acceleration de x24.**

**Le Voting complet : moins de 200 millisecondes par produit. Soit plus de 100 000 produits par jour sur un seul serveur."**

---

## 4:25 — TRANSITION (35 secondes)

**[ECRAN : rester sur la page Performance]**
**[REGARD : se tourner vers Michael]**

**"Vous venez de voir le systeme en action : texte, image, fusion. ~94% d'accuracy, en temps reel.**

**La question maintenant : comment on y arrive ?**

**Michael va vous plonger dans la matiere premiere — les donnees brutes, le desequilibre des classes, et le preprocessing qui rend tout ca possible."**

**[GESTE : passer la parole a Michael]**

---

## 5:00 — FIN. Michael prend la parole.

---

## CHECKLIST AVANT LA DEMO

- [ ] Streamlit lance et modeles charges (lancer 5 min avant)
- [ ] 3 images preparees sur le Bureau :
  - `bouee_piscine.jpg` (pour le test Image)
  - `jaquette_dvd_bleu.jpg` (pour le test Fusion)
  - *(pas d'image pour le test Texte)*
- [ ] Navigateur ouvert sur http://localhost:8502
- [ ] URL de backup : https://huggingface.co/spaces/akiroussama/rakuten-classifier

## SI LA DEMO PLANTE

> Pas de panique. Dire : "Le systeme met quelques secondes a charger les modeles, pendant ce temps laissez-moi vous montrer les resultats sur les slides."
> Basculer sur les slides de la presentation HTML et montrer les metriques statiques.
> Si ca revient : reprendre la demo la ou elle s'est arretee.
