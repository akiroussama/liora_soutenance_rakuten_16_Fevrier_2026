"""
Classifieur mock pour le développement et les tests UI.

Ce classifieur génère des prédictions simulées réalistes sans
nécessiter de vrais modèles ML. Il est utilisé pour:
- Développer et tester l'interface utilisateur
- Démontrer le fonctionnement de l'application
- Servir de fallback si les vrais modèles ne sont pas disponibles

Les probabilités générées sont basées sur un hash du contenu d'entrée
pour assurer la reproductibilité des résultats.
"""
import hashlib
import numpy as np
from typing import Optional
from PIL import Image

from .model_interface import BaseClassifier, ClassificationResult
from .category_mapping import CATEGORY_CODES, get_category_name


class MockClassifier(BaseClassifier):
    """
    Classifieur simulé pour le développement de l'interface.

    Ce classifieur génère des prédictions pseudo-aléatoires mais
    déterministes basées sur le hash des entrées. Cela permet:
    - Des résultats cohérents pour les mêmes entrées
    - Une distribution réaliste des probabilités
    - Un fonctionnement sans dépendances ML lourdes

    Attributes:
        _ready: Indique si le classifieur est initialisé
        _seed: Graine pour la génération pseudo-aléatoire
    """

    def __init__(self, seed: int = 42):
        """
        Initialise le classifieur mock.

        Args:
            seed: Graine de base pour la génération aléatoire
        """
        self._ready = True
        self._seed = seed

    def predict(
        self,
        image: Optional[Image.Image] = None,
        text: Optional[str] = None,
        top_k: int = 5
    ) -> ClassificationResult:
        """
        Génère une prédiction simulée basée sur les entrées.

        La prédiction est déterministe: les mêmes entrées produisent
        toujours les mêmes résultats.

        Args:
            image: Image PIL du produit (optionnel)
            text: Texte du produit (optionnel)
            top_k: Nombre de prédictions à retourner

        Returns:
            ClassificationResult avec des probabilités simulées

        Raises:
            ValueError: Si ni image ni texte n'est fourni
        """
        if image is None and (text is None or text.strip() == ""):
            raise ValueError("Au moins une image ou un texte est requis")

        # Générer une graine déterministe basée sur les entrées
        hash_input = self._generate_hash(image, text)
        rng = np.random.RandomState(hash_input)

        # Générer des probabilités pseudo-aléatoires avec une distribution réaliste
        # Utilise une distribution Dirichlet pour des probabilités qui somment à 1
        # avec quelques pics de confiance
        alpha = np.ones(self.NUM_CLASSES) * 0.5
        # Augmenter alpha pour quelques catégories (créer des pics)
        peak_indices = rng.choice(self.NUM_CLASSES, size=3, replace=False)
        alpha[peak_indices] = rng.uniform(2.0, 5.0, size=3)

        probabilities = rng.dirichlet(alpha)

        # Déterminer la source
        if image is not None and text and text.strip():
            source = "mock_multimodal"
        elif image is not None:
            source = "mock_image"
        else:
            source = "mock_text"

        # Construire le résultat
        top_predictions = self._probabilities_to_predictions(probabilities, top_k)
        best_category, best_confidence = top_predictions[0]

        return ClassificationResult(
            category=best_category,
            confidence=best_confidence,
            top_k_predictions=top_predictions,
            source=source,
            raw_probabilities=probabilities
        )

    def load_model(self, path: str) -> None:
        """
        Simule le chargement d'un modèle (no-op pour le mock).

        Args:
            path: Chemin ignoré pour le mock
        """
        # Le mock n'a pas besoin de charger de modèle
        self._ready = True

    @property
    def is_ready(self) -> bool:
        """Le mock est toujours prêt."""
        return self._ready

    def _generate_hash(
        self,
        image: Optional[Image.Image],
        text: Optional[str]
    ) -> int:
        """
        Génère un hash déterministe à partir des entrées.

        Args:
            image: Image à hasher (utilise les dimensions et quelques pixels)
            text: Texte à hasher

        Returns:
            Entier utilisable comme graine pour numpy.random
        """
        hash_parts = [str(self._seed)]

        if image is not None:
            # Utiliser les dimensions et un échantillon de pixels
            hash_parts.append(f"{image.size}")
            # Réduire l'image pour un hash rapide
            small = image.resize((8, 8)).convert("L")
            hash_parts.append(small.tobytes().hex()[:32])

        if text and text.strip():
            hash_parts.append(text.strip()[:200])

        combined = "|".join(hash_parts)
        hash_bytes = hashlib.md5(combined.encode()).digest()
        return int.from_bytes(hash_bytes[:4], byteorder="big")


class DemoClassifier(MockClassifier):
    """
    Classifieur de démonstration avec des prédictions prédéfinies.

    Utilisé pour des démos contrôlées où on veut des résultats
    spécifiques et prévisibles.
    """

    # Prédictions prédéfinies pour certains mots-clés
    KEYWORD_PREDICTIONS = {
        "piscine": ("2583", 0.92),
        "pool": ("2583", 0.88),
        "livre": ("10", 0.85),
        "book": ("10", 0.82),
        "jeu vidéo": ("40", 0.90),
        "console": ("60", 0.87),
        "playstation": ("60", 0.95),
        "xbox": ("60", 0.94),
        "nintendo": ("60", 0.93),
        "figurine": ("1140", 0.88),
        "pokemon": ("1160", 0.91),
        "jouet": ("1280", 0.84),
        "bébé": ("1320", 0.86),
        "meuble": ("1560", 0.83),
        "jardin": ("2582", 0.89),
        "outil": ("2585", 0.85),
    }

    def predict(
        self,
        image: Optional[Image.Image] = None,
        text: Optional[str] = None,
        top_k: int = 5
    ) -> ClassificationResult:
        """
        Génère une prédiction basée sur des mots-clés ou le mock standard.

        Cherche d'abord des mots-clés connus dans le texte.
        Si aucun n'est trouvé, utilise le comportement mock standard.
        """
        if text:
            text_lower = text.lower()
            for keyword, (category, confidence) in self.KEYWORD_PREDICTIONS.items():
                if keyword in text_lower:
                    # Générer des probabilités cohérentes avec la prédiction
                    probabilities = self._generate_keyword_probabilities(
                        category, confidence
                    )
                    top_predictions = self._probabilities_to_predictions(
                        probabilities, top_k
                    )

                    return ClassificationResult(
                        category=category,
                        confidence=confidence,
                        top_k_predictions=top_predictions,
                        source="demo",
                        raw_probabilities=probabilities
                    )

        # Fallback sur le comportement mock standard
        return super().predict(image, text, top_k)

    def _generate_keyword_probabilities(
        self,
        main_category: str,
        main_confidence: float
    ) -> np.ndarray:
        """
        Génère des probabilités cohérentes avec une prédiction principale.

        Args:
            main_category: Code de la catégorie principale
            main_confidence: Confiance pour la catégorie principale

        Returns:
            Array de probabilités normalisées
        """
        probabilities = np.random.dirichlet(np.ones(self.NUM_CLASSES) * 0.3)

        # Ajuster pour la catégorie principale
        main_idx = CATEGORY_CODES.index(main_category)
        remaining = 1.0 - main_confidence
        scale = remaining / (probabilities.sum() - probabilities[main_idx])
        probabilities *= scale
        probabilities[main_idx] = main_confidence

        return probabilities
