# Plan d'Amélioration pour 18+/20 — Félicitations du Jury
## Classification Multimodale Rakuten — Soutenance RNCP ML Engineer

---

## Etat actuel vs Objectif

| Critère | Etat actuel | Objectif 18+/20 | Action |
|---------|------------|-----------------|--------|
| Cohérence 3 livrables | CORRIGEE (tous les chiffres alignés) | Parfaite | Relecture croisée finale |
| Présentation | 13 slides professionnelles | 20 min chrono, 5 min/personne | Répétitions avec chrono |
| Rapport | Complet, bien structuré, HTML | Enrichir interprétation + limites | Ajouter paragraphes d'analyse |
| Repo Git | Nettoyé, structuré | Clean + documenté | README + architecture claire |
| Démo Streamlit | 5 pages fonctionnelles | Fluide, 3 cas préparés | Test pré-soutenance |
| Q&A | 60 questions préparées | Chacun maîtrise ses 15 questions | Relecture individuelle |

---

## LIVRABLE 1 : RAPPORT (RAPPORT_FINAL_RAKUTEN.html)

### Ce qui est bien (garder) :
- Structure complète respectant la méthodologie DataScientest
- Design HTML professionnel
- Résumé exécutif clair
- Benchmarks avec tableaux comparatifs

### Améliorations recommandées :

| # | Action | Priorité | Impact Jury |
|---|--------|----------|-------------|
| R1 | Ajouter une section "Analyse des erreurs" : matrice de confusion commentée, exemples d'erreurs et hypothèses | HAUTE | Montre l'esprit critique |
| R2 | Enrichir les "Limites" : classes minoritaires F1~60%, drift temporel, dépendance qualité texte | HAUTE | Montre la maturité |
| R3 | Ajouter un diagramme de Gantt (planning du projet) en annexe | MOYENNE | Requis par la méthodologie |
| R4 | Ajouter une section "Description des fichiers de code" en annexe | MOYENNE | Requis par la méthodologie |
| R5 | Vérifier la bibliographie : citer DINOv3 (Oquab et al. 2024), EfficientNet (Tan & Le 2019), LinearSVC (sklearn) | MOYENNE | Rigueur académique |
| R6 | Ajouter les courbes d'apprentissage (loss/accuracy par epoch) pour DINOv3+MLP | BASSE | Preuve du training |

### Points de la méthodologie à vérifier (rapports.docx) :
- [ ] Rendu 1 : Intro/contexte, pertinence données, preprocessing, visualisations
- [ ] Rendu 2 : Choix modèles, optimisation, interprétation (SHAP/LIME/Grad-CAM)
- [ ] Rapport Final : Conclusions, difficultés, bilan, perspectives, biblio, annexes (Gantt + code)

---

## LIVRABLE 2 : PRESENTATION (PRESENTATION_RAKUTEN_SOUTENANCE.html)

### Ce qui est bien (garder) :
- Design moderne, professionnel
- 13 slides couvrant tout le pipeline
- Métriques clés visibles
- Exemples de prédictions convaincants
- Slide Explicabilité (différenciant)

### Améliorations recommandées :

| # | Action | Priorité | Impact Jury |
|---|--------|----------|-------------|
| P1 | Ajouter une slide "Qui a fait quoi" (attribution claire des rôles) | HAUTE | Requis par la méthodologie |
| P2 | Vérifier que le texte sur chaque slide est lisible à 5 mètres | HAUTE | Confort du jury |
| P3 | Ajouter un slide "Architecture technique" simplifié (schéma clair du pipeline complet) | MOYENNE | Déjà présent, vérifier clarté |
| P4 | Préparer une slide de backup "Matrice de confusion" au cas où le jury demande | BASSE | Réactivité en Q&A |

### Points de la méthodologie à vérifier (soutenance.docx) :
- [ ] Présentation du sujet et de l'équipe (qui a fait quoi)
- [ ] Exploration des données et conclusions intéressantes
- [ ] Preprocessing détaillé
- [ ] Modèles comparés et justification des choix
- [ ] Analyse du meilleur modèle (pourquoi il gagne)
- [ ] Conclusion et perspectives
- [ ] Démo Streamlit fonctionnelle

---

## LIVRABLE 3 : REPO GIT

### Ce qui est bien (garder) :
- Structure nettoyée (cleanup effectué)
- .gitignore complet
- Branches obsolètes supprimées
- Streamlit fonctionnel avec vrais modèles

### Améliorations recommandées :

| # | Action | Priorité | Impact Jury |
|---|--------|----------|-------------|
| G1 | Mettre à jour le README.md principal : description projet, structure, instructions d'installation, comment lancer la démo | HAUTE | Première chose que le jury voit |
| G2 | Ajouter un requirements.txt à la racine (liste des dépendances) | HAUTE | Reproductibilité |
| G3 | Vérifier que les notebooks sont propres (pas de cellules vides, outputs nettoyés) | MOYENNE | Propreté du code |
| G4 | S'assurer que src/streamlit fonctionne sans erreur au premier lancement | HAUTE | Démo sans bug |
| G5 | Documenter le config.py (chemins des modèles, paramètres) | BASSE | Maintenabilité |

---

## PLAN D'ACTION PAR PRIORITE

### URGENT (faire cette semaine) :

1. **Répétition individuelle** : chacun chrono ses 5 minutes seul
2. **Répétition collective** : les 4 ensemble, 20 min chrono
3. **Tester la démo** sur le poste de présentation (ou un poste similaire)
4. **Relire les 60 questions** : chacun ses 15

### IMPORTANT (faire avant la soutenance) :

5. **README.md** : ajouter instructions claires (git clone → pip install → streamlit run)
6. **Rapport** : ajouter section "Analyse des erreurs" + "Limites" enrichie
7. **Présentation** : ajouter slide "Equipe et rôles"
8. **requirements.txt** : générer avec pip freeze (nettoyer les dépendances inutiles)

### BONUS (si le temps le permet) :

9. Diagramme de Gantt en annexe du rapport
10. Description des fichiers de code en annexe
11. Bibliographie vérifiée
12. Slides de backup (matrice de confusion, courbes d'apprentissage)

---

## CRITERES D'EVALUATION PROBABLES

| Critère | Poids estimé | Notre niveau | Action pour monter |
|---------|-------------|-------------|-------------------|
| Qualité technique du travail | 30% | FORT (92% image, ~94% fusion) | Déjà excellent |
| Présentation orale | 25% | MOYEN → FORT | Répétitions + timing |
| Rapport écrit | 20% | FORT | Enrichir analyse + annexes |
| Démo fonctionnelle | 10% | FORT | Tester, préparer backup |
| Réponses aux questions | 10% | MOYEN → FORT | 60 questions préparées |
| Travail en équipe | 5% | MOYEN | Slide "qui a fait quoi" |

### Points différenciants pour 18+/20 :
1. **Innovation** : Le Voting System avec Sharpening est original
2. **Multimodal** : Peu de projets combinent texte + image
3. **Explicabilité** : SHAP + LIME + Grad-CAM + AI Act
4. **Performance** : ~94% est excellent pour 27 classes
5. **Démo live** : Pas juste des slides, un vrai système fonctionnel
6. **Esprit critique** : Mentionner les limites proactivement

### Pièges à éviter :
1. Dépasser les 20 minutes → coupe = mauvaise impression
2. Lire ses slides → manque de maîtrise
3. Inconsistance des chiffres entre livrables → manque de rigueur
4. Démo qui plante → préparer un backup screenshot
5. Ne pas savoir répondre → dire "bonne question, c'est en perspective" plutôt que "je ne sais pas"
6. Un seul membre monopolise la parole → répartir équitablement

---

## VERDICT FINAL

| Livrable | Prêt pour 18+/20 ? | Actions restantes |
|----------|--------------------|--------------------|
| Présentation | OUI (après répétitions) | Ajouter slide rôles, 2 répétitions chrono |
| Rapport | OUI (avec enrichissements) | Ajouter analyse erreurs + annexes méthodologie |
| Repo Git | OUI (après README) | README.md + requirements.txt + test démo |
| Q&A | OUI (60 questions prêtes) | Chacun relit ses 15 questions |
| Démo | OUI (si testée) | 3 produits préparés, backup screenshots |

> **Pronostic : 17-19/20 si les répétitions sont faites et la démo fonctionne.**
> Les points techniques sont solides (92% image, ~94% fusion, Voting innovant).
> Le différenciant sera la qualité de la présentation orale et la fluidité de la démo.
