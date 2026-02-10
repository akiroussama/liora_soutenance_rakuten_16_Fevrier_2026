# 60 Questions Q&A - Préparation Soutenance Rakuten
## 15 questions par étudiant, classées par difficulté

---

## OUSSAMA (15 questions) — Contexte, Présentation, Démo, Vision globale

### Niveau Facile (5)

**Q1. Pourquoi avoir choisi une approche multimodale plutôt que texte seul ou image seule ?**
> Le texte seul atteint 83%, l'image seule 92%. Mais ils échouent sur des cas différents. Quand l'image confond un DVD bleu avec une piscine, le texte "DVD Le Grand Bleu" corrige. La fusion atteint ~94% grâce à cette complémentarité. C'est le principe du multimodal : chaque modalité compense les faiblesses de l'autre.

**Q2. Combien de catégories classifiez-vous et quels sont les défis principaux ?**
> 27 catégories, allant des livres aux piscines, en passant par les jeux vidéo et les vêtements. Le défi principal est la variété intra-classe (un livre peut être rouge ou bleu, cartonné ou poche) et la similarité inter-classes (un poster de film ressemble à un DVD).

**Q3. Quelle est la taille du dataset et comment l'avez-vous séparé ?**
> 84 916 produits d'entraînement, 13 812 de test (fournis par Rakuten). On utilise un split stratifié 80/20 sur le train pour la validation, ce qui garantit la même proportion de chaque classe dans train et validation malgré le déséquilibre.

**Q4. Pourquoi Streamlit pour la démo et pas Flask ou FastAPI ?**
> Streamlit est fait pour le prototypage rapide de démos ML : widgets interactifs, cache natif pour les modèles, pas besoin de front-end séparé. Pour une mise en production réelle, on passerait à FastAPI + Docker. Mais pour une soutenance, Streamlit montre les résultats en temps réel avec un effort de développement minimal.

**Q5. Comment fonctionne le slider de fusion dans votre démo ?**
> Le slider ajuste les poids w_image et w_text en temps réel. Par défaut c'est 60/40 (image/texte). Si on met 80/20, l'image domine et ça améliore les cas visuels mais dégrade les cas ambigus. Le jury peut voir l'impact live de chaque ratio.

### Niveau Moyen (5)

**Q6. Quel est l'impact business concret de votre solution ?**
> Avant : classification manuelle à ~5 min/produit, 10-15% d'erreurs, non scalable. Après : <1 seconde/produit, ~6% d'erreur, capable de traiter 100K+ produits/jour. Pour un marketplace comme Rakuten avec des millions de produits, c'est la différence entre des centaines d'opérateurs et un serveur GPU.

**Q7. Si vous deviez refaire le projet, que changeriez-vous ?**
> 1) On testerait CLIP dès le début (fusion native texte+image plutôt que late fusion). 2) On ferait du fine-tuning de DINOv3 sur nos données Rakuten plutôt que de l'utiliser comme extracteur figé. 3) On ajouterait de l'OCR pour lire le texte dans les images — beaucoup de produits ont des informations textuelles visibles sur la photo.

**Q8. Comment gérez-vous les produits multilingues (5 langues) ?**
> 85% des descriptions sont déjà en français. Pour le reste (anglais, allemand, etc.), on a choisi de ne pas traduire car TF-IDF avec des n-grams de caractères (char 3-5) capture naturellement les patterns morphologiques de chaque langue. Et les noms de marques (iPhone, PlayStation) sont identiques dans toutes les langues.

**Q9. Quel est le temps total d'inférence pour une prédiction complète (fusion) ?**
> Texte seul : ~10ms (TF-IDF + LinearSVC). Image seule : ~200ms (forward pass DINOv3 + EfficientNet + XGBoost + voting). Fusion : ~250ms au total. Le bottleneck est l'extraction de features image par les CNN/ViT. C'est compatible avec du temps réel.

**Q10. Comment déploieriez-vous cette solution en production ?**
> Architecture : FastAPI comme API REST, modèles sérialisés (joblib/PyTorch), Docker pour la conteneurisation, GPU pour l'inférence image. Monitoring : Prometheus + Grafana pour les métriques, détection de drift avec Evidently. Pipeline CI/CD : tests automatisés + déploiement continu. A/B testing pour valider les nouvelles versions de modèles.

### Niveau Difficile (5)

**Q11. Comment détecteriez-vous et géreriez-vous le drift temporel ?**
> Le drift peut être de 2 types : data drift (la distribution des produits change) et concept drift (les catégories évoluent). Pour le détecter : monitoring continu du F1-Score sur un échantillon étiqueté, test de Kolmogorov-Smirnov sur les distributions de features. Pour le corriger : réentraînement périodique (mensuel), active learning sur les cas à faible confiance.

**Q12. Votre accuracy de ~94% est en fusion. Quelle est la performance sur les classes minoritaires spécifiquement ?**
> Les classes minoritaires (Consoles, Cartes cadeaux) ont un F1 autour de 60-65%. C'est la principale limite. Le class_weight='balanced' aide mais ne suffit pas. En perspective, on envisage du data augmentation ciblé (SMOTE pour le texte, augmentation géométrique pour les images) et un seuil de confiance : en-dessous de 70%, le produit part en review humaine.

**Q13. Pourquoi Late Fusion et pas Early Fusion ou un modèle end-to-end comme CLIP ?**
> Late Fusion a 3 avantages : 1) chaque modalité a son propre classifieur optimisé indépendamment, 2) on peut débugger chaque branche séparément, 3) si une modalité est absente (35% de descriptions manquantes), l'autre prend le relais. CLIP serait plus élégant mais nécessite un fine-tuning coûteux et un dataset beaucoup plus grand. Notre Late Fusion est pragmatique et performante.

**Q14. Comment justifiez-vous le ratio 60/40 image/texte ? Avez-vous optimisé ces poids ?**
> Le ratio 60/40 a été choisi empiriquement : on a testé 50/50, 60/40, 70/30, 80/20 sur le set de validation. 60/40 donne le meilleur F1 pondéré. C'est logique : l'image (92%) est plus fiable que le texte (83%), donc elle mérite plus de poids. Mais pas trop : le texte est crucial pour les cas ambigus (DVD vs poster). On pourrait l'optimiser par classe (certaines catégories sont plus "visuelles").

**Q15. Comment vous êtes-vous répartis le travail et quelle a été la difficulté principale de coordination ?**
> Michael sur les données et le preprocessing texte, Liviu sur les modèles NLP, Johan sur toute la partie image et le Voting System, moi sur la mise en forme et l'intégration. La difficulté principale : la cohérence entre les pipelines texte et image — s'assurer que les mêmes splits de données sont utilisés partout, que les métriques sont calculées de la même manière, et que le format de sortie est compatible pour la fusion.

---

## MICHAEL (15 questions) — Données, Preprocessing, TF-IDF

### Niveau Facile (5)

**Q16. Quelles sont les étapes de votre pipeline de nettoyage texte ?**
> 1) Suppression des balises HTML, 2) Nettoyage des caractères spéciaux, 3) Normalisation des espaces et passage en minuscules, 4) Concaténation de designation et description, 5) Vectorisation TF-IDF. On ne fait PAS de lemmatisation pour préserver les noms de marques (iPhone, PlayStation).

**Q17. Pourquoi avoir concaténé designation et description plutôt que de les traiter séparément ?**
> La designation est courte (2-5 mots) mais toujours présente. La description est longue mais manquante dans 35% des cas. En concaténant, on crée un seul vecteur riche quand la description existe, et on garde au moins la designation quand elle manque. C'est plus robuste qu'un traitement séparé qui double le nombre de features.

**Q18. Comment avez-vous géré les 35% de descriptions manquantes ?**
> On remplace les NaN par une chaîne vide. Ainsi, pour ces produits, seule la designation contribue au vecteur TF-IDF. C'est suffisant pour les cas non ambigus ("iPhone 15 Pro" est classifiable sans description). Pour les cas ambigus, c'est l'image qui compense via la fusion multimodale.

**Q19. Que signifie "TF-IDF 280K features" concrètement ?**
> TF-IDF (Term Frequency-Inverse Document Frequency) transforme chaque texte en un vecteur de 280 000 dimensions. Chaque dimension correspond à un n-gram (mot ou séquence de caractères). TF mesure la fréquence locale, IDF pénalise les termes trop communs ("le", "de"). On combine des n-grams de mots (1-2) et de caractères (3-5), d'où les 280K dimensions.

**Q20. Comment avez-vous traité le déséquilibre des classes ?**
> 3 stratégies combinées : 1) class_weight='balanced' dans le classifieur (pénalise plus les erreurs sur les classes rares), 2) F1-Score pondéré comme métrique (pas accuracy brute qui favorise les classes majoritaires), 3) Stratified split pour garder la même proportion dans train et validation. On n'a PAS fait d'oversampling car ça peut créer du surapprentissage sur le texte.

### Niveau Moyen (5)

**Q21. Pourquoi TF-IDF et pas Word2Vec ou des embeddings pré-entraînés ?**
> On a testé les deux. TF-IDF donne une performance équivalente à Word2Vec sur notre dataset, avec deux avantages : 1) Interprétabilité — on peut voir quels mots contribuent à la décision, 2) Les n-grams de caractères capturent les sous-mots et les marques (le "Pho" de "iPhone" est un signal). Word2Vec perd cette granularité.

**Q22. Pourquoi des n-grams de caractères (3-5) en plus des n-grams de mots ?**
> Les n-grams de caractères capturent des patterns morphologiques que les mots entiers ratent : "PlayStation" contient "Play" et "Station", "iPhone" contient "Phone". C'est aussi robuste aux fautes d'orthographe ("Playstation" vs "PlayStation") et aux langues non françaises. Les ranges 3-5 sont un sweet spot entre granularité et volume.

**Q23. Quel est le ratio train/validation/test et pourquoi ?**
> Rakuten fournit 84 916 produits d'entraînement et 13 812 de test. On split le train en 80/20 stratifié pour la validation. Le stratified split est crucial avec le déséquilibre : sinon une classe rare (0.9%) pourrait être absente du set de validation. Le test set officiel Rakuten garantit une évaluation comparable aux autres équipes.

**Q24. Comment savez-vous que vos données ne contiennent pas de fuite (data leakage) ?**
> Le test set est fourni séparément par Rakuten, sans recouvrement avec le train. Pour le split train/validation, on utilise sklearn train_test_split avec stratification. Le TF-IDF est fit uniquement sur le train (pas le validation ni le test). Le pipeline sklearn garantit que le fit et le transform sont correctement séparés.

**Q25. Si une nouvelle catégorie apparaît (ex: "Drones"), que se passe-t-il ?**
> Le modèle actuel ne peut pas prédire une catégorie absente de l'entraînement. Il classera le drone dans la catégorie la plus proche (probablement "Jeux vidéo" ou "Électronique"). Solutions : 1) Réentraîner avec la nouvelle catégorie, 2) Détection de nouveauté (anomaly detection) pour flaguer les produits inclassables, 3) Active learning : les cas à faible confiance vont en review humaine.

### Niveau Difficile (5)

**Q26. Vous avez 280K features TF-IDF. N'est-ce pas un problème de dimensionnalité (curse of dimensionality) ?**
> En théorie oui, mais en pratique non pour un SVM linéaire. Les SVM sont conçus pour la haute dimension — c'est leur force. La matrice TF-IDF est très sparse (chaque document n'utilise qu'une fraction des 280K termes), ce qui la rend efficace en mémoire. On a testé une réduction par SVD (LSA) mais sans gain significatif.

**Q27. Avez-vous envisagé une stratégie de pondération TF-IDF différente (BM25, sublinéaire) ?**
> On utilise le TF sublinéaire (sublinear_tf=True dans sklearn), qui applique 1+log(tf) au lieu de tf brut. Ça empêche les termes très fréquents de dominer. BM25 est une alternative intéressante, surtout pour la recherche d'information, mais pour la classification, le gain par rapport à TF-IDF sublinéaire est marginal. On a privilégié la simplicité.

**Q28. Comment validez-vous que le preprocessing ne perd pas d'information discriminante ?**
> On a comparé les performances avec et sans chaque étape : sans nettoyage HTML (83% → 81%), sans concat description (83% → 78%), sans n-grams char (83% → 80%). Chaque étape contribue positivement. La seule étape qu'on a retirée est la lemmatisation (83% → 82%) car elle détruit les noms de marques.

**Q29. 85% du texte est en français. Comment TF-IDF gère-t-il les 15% restants sans traduction ?**
> TF-IDF est agnostique à la langue : il voit des séquences de caractères, pas des "mots" au sens linguistique. Un produit en anglais "Swimming pool for children" aura des n-grams distincts des produits français, ce qui crée implicitement des features spécifiques par langue. Les n-grams de caractères ("swi", "imm", "ing") capturent les patterns morphologiques anglais. Et les marques internationales sont identiques dans toutes les langues.

**Q30. Si le jury vous demande : "Montrez-moi les 10 features les plus importantes pour la classe Livres", comment faites-vous ?**
> Avec LinearSVC, on accède directement aux coefficients du modèle. Pour chaque classe, le vecteur de poids w donne l'importance de chaque feature. Les 10 features avec les poids les plus élevés pour la classe "Livres" sont les n-grams les plus discriminants. On peut aussi utiliser SHAP pour une analyse plus fine. Ce serait des termes comme "roman", "auteur", "éditeur", "pages", "édition".

---

## JOHAN (15 questions) — Image, DINOv3, EfficientNet, Voting, Démo

### Niveau Facile (5)

**Q31. Pourquoi DINOv3 et pas ResNet50 ou VGG16 ?**
> DINOv3 est un Vision Transformer (ViT) entraîné en self-supervised par Meta. Contrairement à ResNet50 (entraîné supervisé sur ImageNet), DINOv3 apprend des représentations visuelles sans labels, ce qui le rend plus généraliste. Résultat : 91.4% vs ~75% pour ResNet50 sur nos données. Le self-supervised learning capture la structure visuelle profonde des images.

**Q32. Que signifie "1024 features" extraites par DINOv3 ?**
> DINOv3 prend une image 224x224 et produit un vecteur de 1024 nombres réels. Chaque nombre encode un aspect visuel de l'image (forme, texture, couleur, structure). C'est le "CLS token" du ViT — un résumé global de l'image. Ces 1024 features sont ensuite envoyées à un MLP pour la classification.

**Q33. Comment fonctionne le Voting System ?**
> 3 classifieurs votent sur chaque image : DINOv3+MLP (poids 4/7, 91.4%), EfficientNet-B0 (poids 2/7, ~75%), XGBoost calibré (poids 1/7, 76.5%). On pondère les probabilités de chaque modèle par son poids, puis on additionne. La classe avec le score total le plus élevé gagne. C'est un "Soft Voting" pondéré.

**Q34. Pourquoi avoir choisi ces 3 modèles spécifiquement pour le Voting ?**
> Diversité maximale : DINOv3 est un Vision Transformer (attention globale), EfficientNet est un CNN classique (convolutions locales), XGBoost est du ML tabulaire (statistiques sur les features). Ils "voient" l'image différemment et font des erreurs différentes. C'est cette complémentarité qui fait la force du vote.

**Q35. Quel est le rôle du XGBoost dans le Voting puisqu'il est le moins bon (76.5%) ?**
> XGBoost apporte une vision statistique indépendante des réseaux de neurones. Sa matrice de corrélation avec DINOv3 est faible (0.3), ce qui signifie qu'il se trompe sur des cas différents. Même avec 76.5%, sa correction statistique améliore le score global de 91.4% à 92%. C'est le principe des ensembles : la diversité prime sur la performance individuelle.

### Niveau Moyen (5)

**Q36. Qu'est-ce que le "Sharpening" (p³/Σp³) et pourquoi l'appliquer à XGBoost ?**
> XGBoost produit des probabilités "molles" : au lieu de dire 80% classe A / 5% classe B, il dit 35% / 25%. Le sharpening élève chaque probabilité au cube puis renormalise. Résultat : les probabilités deviennent plus tranchées (80% / 2%), ce qui évite de "diluer" le vote. Sans sharpening, XGBoost noierait les décisions confiantes de DINOv3.

**Q37. Comment avez-vous déterminé les poids 4/7, 2/7, 1/7 ?**
> Par validation croisée sur le set de validation. On a testé des grilles de poids (de 1/1/1 à 6/2/1) et mesuré le F1-Score pondéré. Le ratio 4/2/1 donne le meilleur compromis. C'est logique : DINOv3 (91.4%) mérite 4 fois plus de poids que XGBoost (76.5%). EfficientNet (~75%) est au milieu.

**Q38. Pourquoi utiliser ImageNet pour la normalisation alors que vos images sont des produits e-commerce ?**
> DINOv3 et EfficientNet sont pré-entraînés sur ImageNet. Leurs couches internes "s'attendent" à des images normalisées avec la moyenne et l'écart-type d'ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]). Si on normalise différemment, les activations sont décalées et les features extraites sont dégradées. C'est une contrainte du transfer learning.

**Q39. Avez-vous essayé le fine-tuning de DINOv3 au lieu de l'utiliser comme extracteur figé ?**
> Non, par manque de ressources GPU et de temps. DINOv3 comme extracteur figé donne déjà 91.4%. Le fine-tuning nécessiterait : 1) un GPU puissant pendant des heures, 2) un learning rate très faible pour ne pas détruire les poids pré-entraînés, 3) un risque d'overfitting sur notre dataset de 85K images. C'est en perspective pour une V2.

**Q40. Comment avez-vous géré les images de mauvaise qualité ou non pertinentes ?**
> Le dataset Rakuten contient des images de qualité variable : photos floues, fonds blancs, logos à la place de produits. On n'a PAS fait de filtrage car chaque image contient un signal, même faible. DINOv3 est robuste à la qualité car il a été entraîné sur des millions d'images variées. Pour les cas extrêmes, c'est la fusion avec le texte qui compense.

### Niveau Difficile (5)

**Q41. DINOv3 utilise un Vision Transformer. Expliquez le mécanisme d'attention et pourquoi c'est pertinent ici.**
> Le ViT découpe l'image en patches 16x16, les traite comme des "tokens" (comme des mots en NLP), et applique de l'attention multi-têtes. Chaque patch "regarde" tous les autres pour comprendre le contexte global. Pour un produit e-commerce, ça signifie que le modèle peut relier un logo en haut à gauche avec la forme du produit au centre — une capacité que les CNN classiques (champ réceptif local) n'ont pas aussi bien.

**Q42. Votre Voting passe de 91.4% à 92%. Ce gain de 0.6% justifie-t-il la complexité supplémentaire (3 modèles) ?**
> Oui, pour 2 raisons : 1) Le gain n'est pas uniforme — le Voting améliore surtout les classes difficiles (F1 +3-5% sur les classes minoritaires), 2) La robustesse — si DINOv3 échoue sur un type d'image (ex: photo très sombre), les 2 autres modèles compensent. En production, la fiabilité est plus importante que la performance moyenne. Le coût marginal d'inférence est faible (<50ms de plus).

**Q43. Pourquoi ne pas avoir utilisé un ensemble de plusieurs DINOv3 (même architecture, seeds différentes) ?**
> Un ensemble de modèles identiques manque de diversité. Tous les DINOv3 font les mêmes erreurs car ils "voient" les images de la même façon. Notre Voting combine 3 architectures DIFFERENTES (ViT, CNN, ML tabulaire) qui ont des biais inductifs différents. La diversité architecturale est la clé de l'ensemble learning efficace.

**Q44. Le self-supervised learning de DINOv3 n'utilise pas de labels. Comment savez-vous que les features sont pertinentes pour la classification de produits ?**
> DINO est entraîné par distillation : un "student" apprend à reproduire les représentations d'un "teacher". Ce processus force le modèle à capturer les structures visuelles fondamentales (formes, textures, compositions) qui sont universellement utiles. Pour la classification de produits, ces features sont pertinentes car un livre a une structure visuelle distincte d'une console ou d'une piscine. Le MLP en tête fine-tune la projection vers nos 27 classes.

**Q45. Si vous deviez réduire le temps d'inférence image de moitié, quelles compromis feriez-vous ?**
> Option 1 : Retirer EfficientNet du Voting (gain ~80ms, perte ~0.5% accuracy). Option 2 : Utiliser DINOv3-Small au lieu de DINOv3-Base (gain ~50%, perte ~1-2%). Option 3 : Quantization INT8 des modèles PyTorch (gain ~30%, perte négligeable). Option 4 : Batch les inférences plutôt que 1 par 1 (gain massif en throughput). En production, on combinerait les options 3 et 4.

---

## LIVIU (15 questions) — NLP, Fusion, Architecture, Explicabilité, Technique avancée

### Niveau Facile (5)

**Q46. Pourquoi LinearSVC et pas un modèle deep comme CamemBERT pour le texte ?**
> LinearSVC atteint 83% de F1-Score, CamemBERT ~81% sur notre dataset. Le SVM gagne car : 1) TF-IDF à 280K dimensions crée un espace déjà très discriminant, 2) CamemBERT nécessite du fine-tuning coûteux et un GPU, 3) LinearSVC infère en <10ms vs ~200ms pour CamemBERT. Pour ce dataset spécifique, le ML classique bat le deep learning en texte.

**Q47. Expliquez le Late Fusion en termes simples.**
> C'est comme un jury de 2 experts : un lit le texte, l'autre regarde l'image. Chacun donne son avis (probabilités par classe). Puis on combine : 60% de poids pour l'expert image (plus fiable à 92%) et 40% pour l'expert texte (83%). Le verdict final est la classe avec le score combiné le plus élevé. ~94% de bonnes décisions.

**Q48. Qu'est-ce que SHAP et comment l'utilisez-vous ?**
> SHAP (SHapley Additive exPlanations) attribue à chaque feature sa contribution à la prédiction. Pour un produit classé "Livres", SHAP montre que le mot "roman" contribue +0.15, "auteur" +0.10, "DVD" -0.08. C'est de l'explicabilité globale et locale. Ça permet de comprendre et d'auditer les décisions du modèle, ce qui est important pour la confiance et la conformité AI Act.

**Q49. Qu'est-ce que Grad-CAM et en quoi c'est utile pour les images ?**
> Grad-CAM (Gradient-weighted Class Activation Mapping) génère une heatmap sur l'image montrant les zones qui influencent le plus la décision. Pour un livre, la heatmap sera rouge sur la couverture. Pour une console, sur les manettes. Ça permet de vérifier que le modèle regarde "au bon endroit" et pas le fond blanc de l'image.

**Q50. Pourquoi avoir choisi le F1-Score pondéré comme métrique principale et pas l'accuracy ?**
> Avec un déséquilibre 13:1, un modèle qui prédit toujours "Piscines" aurait 12% d'accuracy — pas terrible mais l'accuracy ne pénalise pas assez les erreurs sur les classes rares. Le F1-Score pondéré donne à chaque classe un poids proportionnel à sa taille, mais pénalise beaucoup plus les faux négatifs sur les classes rares. C'est la métrique officielle du challenge Rakuten.

### Niveau Moyen (5)

**Q51. Comment le class_weight='balanced' fonctionne-t-il mathématiquement ?**
> Il attribue un poids inversement proportionnel à la fréquence de la classe : w_i = n_total / (n_classes × n_i). Pour une classe rare (0.9%), le poids sera ~13x plus élevé que pour la classe majoritaire (12%). Concrètement, une erreur sur un produit "Console" coûte 13 fois plus cher que sur un produit "Piscine" dans la fonction de perte. Ça force le classifieur à bien classer les classes rares.

**Q52. Pourquoi le ratio de fusion est-il 60/40 et pas optimisé dynamiquement par classe ?**
> On utilise un ratio global car c'est plus robuste et simple. Mais vous avez raison : certaines catégories sont plus "visuelles" (Piscines → image forte) et d'autres plus "textuelles" (Livres → titre discriminant). Une fusion par classe est en perspective. L'implémentation nécessiterait d'apprendre 27 ratios séparés, ce qui risque l'overfitting sur notre taille de dataset.

**Q53. Quelle est la différence entre SHAP et LIME pour l'explicabilité ?**
> SHAP est basé sur la théorie des jeux (valeurs de Shapley) et donne des attributions globalement cohérentes. LIME est une approximation locale : il perturbe l'entrée et observe l'effet sur la prédiction. SHAP est plus rigoureux théoriquement mais plus lent. LIME est plus rapide et intuitif. On utilise SHAP pour l'analyse globale (quels mots comptent en général) et LIME pour l'analyse locale (pourquoi CE produit est classé ainsi).

**Q54. Comment la conformité AI Act s'applique-t-elle à votre système ?**
> Le AI Act européen exige la transparence des systèmes IA à risque moyen. Un système de classification de produits affectant le référencement e-commerce tombe dans cette catégorie. Nos outils d'explicabilité (SHAP, LIME, Grad-CAM) répondent à l'exigence de transparence : pour chaque prédiction, on peut expliquer POURQUOI le modèle a décidé ainsi. C'est un argument différenciant.

**Q55. Comment le GridSearch a-t-il été configuré pour optimiser le LinearSVC ?**
> On a optimisé le paramètre C (régularisation) avec un GridSearch sur [0.01, 0.1, 0.5, 1.0, 5.0] en cross-validation 5-fold stratifiée. Le F1-Score pondéré est la métrique d'optimisation. C=0.5 donne le meilleur compromis biais-variance. Un C plus petit sous-apprend (frontière trop simple), un C plus grand sur-apprend (frontière trop complexe).

### Niveau Difficile (5)

**Q56. Comparez les avantages théoriques de SVM vs réseaux de neurones pour la classification de texte en haute dimension.**
> En haute dimension sparse (280K features TF-IDF), les SVM ont un avantage structural : ils cherchent l'hyperplan à marge maximale, ce qui est optimal quand les données sont linéairement séparables en haute dimension (théorème de Cover : en dimension suffisante, les données deviennent linéairement séparables). Les réseaux de neurones excellent quand les features sont denses et de basse dimension (embeddings). C'est pourquoi LinearSVC bat CamemBERT ici : TF-IDF 280K est le terrain de jeu idéal du SVM.

**Q57. Votre Voting System utilise un soft voting pondéré. Pourquoi pas un stacking (meta-learner) ?**
> Le stacking entraîne un méta-classifieur sur les sorties des modèles de base. Avantage : il apprend automatiquement les poids et peut capter des interactions non linéaires. Inconvénient : risque d'overfitting (on entraîne un modèle sur les prédictions d'autres modèles), et besoin d'un split supplémentaire pour éviter le leakage. Notre soft voting pondéré est plus simple, plus robuste, et le gain du stacking serait marginal vu la qualité de DINOv3 (91.4%).

**Q58. Si vous ajoutiez l'OCR comme 3ème modalité, comment l'intégreriez-vous dans votre architecture ?**
> L'OCR extrairait le texte visible dans les images (marques, descriptions sur packaging). On l'ajouterait comme un 3ème vecteur TF-IDF dans la fusion. Architecture : Image → DINOv3 (features visuelles) + OCR → TF-IDF (texte visuel) + Description → TF-IDF (texte saisi). Late Fusion à 3 branches avec poids appris. Le texte OCR serait complémentaire car il contient de l'information que ni la description ni l'image seule ne capturent (ex: "100% coton" visible sur l'étiquette).

**Q59. Comment évalueriez-vous la calibration de vos modèles (pas juste l'accuracy) ?**
> Un modèle bien calibré dit "90% confiance" et a raison 90% du temps. On mesure ça avec : 1) Le Brier Score (MSE entre probabilité prédite et label réel), 2) Le Reliability Diagram (courbe de calibration vs diagonale idéale), 3) L'Expected Calibration Error (ECE). Pour XGBoost, la calibration était mauvaise (d'où le sharpening). Pour DINOv3+MLP, elle est bonne car le softmax de la dernière couche produit des probabilités bien calibrées.

**Q60. Question de synthèse : Si vous aviez un GPU illimité et 6 mois de plus, quel serait votre pipeline idéal ?**
> Pipeline V2 : 1) **Texte** : Fine-tuning CamemBERT-large sur nos données (+5-8% vs LinearSVC), 2) **Image** : Fine-tuning DINOv3 end-to-end avec augmentation (+2-3%), 3) **OCR** : Extraction texte dans les images avec PaddleOCR, 4) **Fusion** : CLIP-like (encodeur joint texte+image, pas de late fusion), 5) **Entraînement** : Curriculum learning (commencer par les classes faciles, ajouter les difficiles), 6) **Production** : Quantization ONNX, serveur TorchServe, monitoring Evidently, A/B testing continu. Objectif : >97% accuracy avec confiance calibrée.

---

## STRATEGIE DE REPONSE AUX QUESTIONS

### Règle universelle :
> "Bonne question. [Réponse directe en 1 phrase]. [Explication technique en 2-3 phrases]. [Exemple concret si possible]."

### Si tu ne connais pas la réponse :
> "C'est un point intéressant que nous n'avons pas exploré dans ce projet. Notre intuition serait que [hypothèse raisonnée]. C'est quelque chose que nous envisageons en perspective."

### Si la question concerne un autre membre :
> Ne pas dire "je ne sais pas", mais : "C'est le domaine de [Johan/Liviu/Michael]. [Prénom], tu peux détailler ?"

### Questions pièges classiques :
- "Pourquoi pas plus simple ?" → "On a commencé simple (baseline RF) et ajouté de la complexité uniquement quand c'était justifié par le gain de performance."
- "C'est pas trop complexe pour de la classification ?" → "Chaque composant est justifié : le Voting gagne 0.6%, mais surtout améliore la robustesse sur les classes difficiles."
- "Votre accuracy est-elle reproductible ?" → "Oui, nous utilisons des random seeds fixes (42) et le test set est fourni par Rakuten."
