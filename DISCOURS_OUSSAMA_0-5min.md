# DISCOURS OUSSAMA — 7 minutes

**[ECRAN : Slide de couverture]**

"Bonjour Antoine. Je suis Oussama Akir, promotion octobre 2025 chez DataScientest, parcours Machine Learning Engineer. Aujourd'hui je vous presente notre projet de certification : la classification automatique de produits pour Rakuten France.

Imaginez : vous etes responsable catalogue chez Rakuten France. Chaque jour, des milliers de produits arrivent. Un vendeur poste une jaquette bleue avec le titre 'Le Grand Bleu'. C'est un DVD ? Un accessoire piscine ? Un poster ?

Un humain met 5 minutes par produit, avec 10 a 15% d'erreur. Nous, on le fait en moins d'une seconde, sur 100 000 produits par jour.

Bonjour, je suis Oussama. Notre mission : 85 000 produits, 27 categories, texte plus image."

---

**[ACTION : Basculer sur le navigateur — localhost:8502]**

"Voici notre application, deployee en production sur Hugging Face Spaces. Les chiffres : 84 916 produits d'entrainement, 27 categories, F1-score 0.85 en fusion multimodale.

Maintenant... la preuve en direct."

**[ACTION : Cliquer "Classifier un produit" → Page Demo]**

---

**[ECRAN : Page Demo > Onglet "Analyse Texte"]**

"Test numero 1 : le texte seul."

**[ACTION : Taper "Harry Potter edition poche"]**
**[ACTION : Cliquer "Analyser le Texte"]**

"Le modele capte les mots-cles — harry, potter, poche — et predit : Livres, 99% de confiance. Zero hesitation. Quand le texte est clair, 10 millisecondes suffisent."

---

**[ACTION : Cliquer onglet "Analyse Image"]**
**[ACTION : Upload la photo de bouee piscine]**

"Test numero 2 : l'image seule, sans aucun texte. Une bouee de piscine."

**[ACTION : Cliquer "Lancer le Voting Image"]**

"Et la, c'est la ou ca devient interessant. On ne fait pas tourner UN modele, mais TROIS.

DINOv3, le patron — un Vision Transformer de Meta. Il voit l'image dans sa globalite. Poids 4 sur 7.

EfficientNet, l'expert des details — un CNN qui capte les textures. Poids 2.

XGBoost, le statisticien — du machine learning classique sur des features extraites. Poids 1. Son role : corriger quand les deux autres hesitent.

Trois architectures, trois facons de 'voir' une image. Resultat : Piscine.

C'est notre Conseil des Sages."

---

**[ACTION : Cliquer onglet "FUSION Multimodale"]**

"Test numero 3 — et c'est celui-ci qui justifie tout le projet."

**[ACTION : Taper "DVD Le Grand Bleu Luc Besson"]**
**[ACTION : Upload la jaquette bleue]**
**[ACTION : Cliquer "Calculer la Fusion"]**

"Je tape 'DVD Le Grand Bleu Luc Besson' et j'uploade cette jaquette. Elle est bleue. L'image seule pourrait confondre avec une piscine.

Mais la fusion combine les deux : le texte dit 'DVD', 'Luc Besson'. L'image dit 'objet bleu'. Le texte corrige. Resultat : DVD. Correct.

C'est ca, la complementarite multimodale. L'un sauve l'autre."

**[ACTION : Bouger le slider vers 90% image]**

"Regardez : si je pousse l'image a 90%... la prediction change. Le texte perd son pouvoir correcteur."

**[ACTION : Remettre le slider a 60%]**

"60/40 — c'est l'equilibre qu'on a calibre. Le texte a 83% est en fait notre modalite la plus fiable. L'image a 79% apporte la robustesse. Ensemble : F1 0.85."

---

**[ACTION : Cliquer "Explicabilite" dans le menu lateral]**

"Un F1 de 0.85 ne suffit pas. Il faut prouver que le modele comprend — et pas qu'il triche.

Voici du Grad-CAM et des Attention Maps. Chaque colonne, c'est un de nos 3 modeles. Meme image, mais regardez : ils ne regardent PAS les memes zones."

**[ACTION : Pointer les colonnes a l'ecran]**

"DINOv3, le Transformer, capte la structure globale — l'objet entier. EfficientNet, le CNN, se concentre sur les textures et les bords. XGBoost analyse des features statistiques — dimensions, couleurs dominantes.

Des erreurs differentes, des forces differentes. C'est POUR CA que le vote fonctionne."

**[ACTION : Scroller vers "Focus Battle"]**

"Exemple concret : DINOv3 est a 90% de confiance, EfficientNet a 77%, XGBoost seulement 31%. XGBoost se trompe — mais avec un poids de 1 sur 7, il ne contamine pas le vote. Le systeme est anti-fragile."

---

**[ACTION : Cliquer "Performance" dans le menu lateral]**
**[ACTION : Cliquer onglet "Benchmark CPU/GPU"]**

"En production : DINOv3 passe de 2 secondes en CPU a 82 millisecondes en GPU — facteur x24.

Le Voting complet : 170 millisecondes par produit. Soit 500 000 classifications par jour sur un seul serveur GPU."

---

**[ACTION : Cliquer "Qualite" dans le menu lateral]**

"On ne deploie pas un modele sans filet. Notre pipeline de qualite : 210 tests — unitaires, integration, performance, ET securite OWASP. 85% de couverture de code.

En bas, nos Quality Gates ML : accuracy image 79.3%, F1 macro 78.5%. Six gates sur six en vert."

**[PAUSE — regarder le jury]**

"Un point d'honneur : en cours de projet, Johan a decouvert un data leakage sur l'evaluation image. Les scores etaient gonfles — le Voting affichait 92% au lieu de 79%. On a TOUT corrige. Chaque fichier, chaque slide, chaque metrique. C'est la difference entre un projet academique et une demarche d'ingenieur."

---

**[ACTION : Cliquer "Conclusions" dans le menu lateral]**
**[ACTION : Scroller vers "Impact Business"]**

"Impact business. Avant : un operateur met 5 minutes par produit, 10 a 15% d'erreur. Apres : moins d'une seconde, 27 categories, avec un seuil de confiance a 80%.

Les 70% de produits au-dessus du seuil sont classes automatiquement. Les 30% restants partent en revue humaine. Zero automatisation aveugle — le systeme sait dire 'je ne suis pas sur'."

---

"Recapitulons ce que vous venez de voir :

Un systeme qui classifie en temps reel — texte, image, ou les deux. Un Voting de 3 architectures complementaires. De l'explicabilite avec Grad-CAM et SHAP. Un pipeline de 210 tests. Et une correction de data leakage en cours de projet.

Tout ca, c'est le resultat. La question maintenant : comment on construit ca ? On va plonger dans la matiere premiere — 85 000 produits, 5 langues, 35% de descriptions manquantes, et un desequilibre de 13 contre 1 entre les classes."
