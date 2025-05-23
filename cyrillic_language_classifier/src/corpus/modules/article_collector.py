"""Collecteur d'articles Wikipédia
pour la constitution de corpus multilingues

Ce module fournit la classe ArticleCollector qui encapsule toute la logique
de collecte d'articles Wikipédia selon différentes stratégies adaptatives.
Il optimise la collecte en fonction des ressources disponibles pour chaque
langue et respecte les contraintes de débit de l'API Wikipedia.

Fonctionnalités principales:
- Collecte par catégorie avec stratégies ordonnées et aléatoires
- Exploration hiérarchique des sous-catégories avec cache intelligent
- Collecte d'articles aléatoires avec validation
- Gestion adaptative des paramètres selon la richesse linguistique
- Optimisation des performances avec cache et filtrage intelligent

Le collecteur utilise des stratégies adaptatives où les langues bien dotées
utilisent des paramètres différents des langues moins dotées pour optimiser
la qualité et la diversité du corpus final.
"""

import random
import logging
from typing import List, Dict, Optional, Tuple, Set, Any, Union
from collections import Counter

# Imports depuis les modules locaux
from .config import CATEGORY_PREFIXES, get_adaptive_params
from .api_utils import (
    fetch_category_articles,
    fetch_subcategories,
    fetch_random_article
)
from .text_processing import (
    select_valid_articles,
    process_article,
    validate_article,
    ARTICLE_TYPES,
)


# === CONSTANTES DE COLLECTE ===

# Limites de sécurité pour éviter les boucles infinies
MAX_SUBCATEGORIES_CACHE = 800
MAX_SECONDARY_EXPLORATION = 300
MAX_RANDOM_ATTEMPTS_MULTIPLIER = 15
MAX_EMPTY_SUBCATEGORIES = 10

# Paramètres par défaut
DEFAULT_NUM_ARTICLES = 20
DEFAULT_FIXED_RATIO = 0.6
DEFAULT_SLEEP_TIME = (1, 3)
DEFAULT_MAX_DEPTH = 5
DEFAULT_SAMPLE_SIZE = 5
DEFAULT_ADDITIONAL_SUBCATS = 20

# Messages de log standardisés
LOG_MESSAGES = {
    "START_CATEGORY": "Récupération d'articles par catégorie pour {lang}, catégorie: {cat}",
    "END_CATEGORY": "Fin de la récupération pour {lang}, catégorie: {cat}, {count} articles trouvés",
    "START_SUBCATEGORIES": "Recherche dans les sous-catégories pour {lang}, catégorie: {cat}",
    "END_SUBCATEGORIES": "Récupération terminée, {count} articles trouvés dans les sous-catégories",
    "START_RANDOM": "Récupération d'articles aléatoires pour {lang}, objectif: {target} articles",
    "END_RANDOM": "Récupération d'articles aléatoires terminée pour {lang}: {count} articles trouvés",
    "DUPLICATE_WARNING": "ATTENTION: {count} doublons détectés dans la liste d'articles!",
    "TARGET_REACHED": "Objectif de tokens atteint ({current}/{target}), arrêt de la collecte.",
    "SECONDARY_EXPLORATION": "Tentative {attempt}: exploration secondaire à partir de '{subcat}'",
    "RANDOM_WARNING": "Attention: seulement {found}/{target} articles aléatoires trouvés pour {lang}",
}


class ArticleCollector:
    """
    Collecteur d'articles Wikipedia avec stratégies adaptatives.

    Cette classe encapsule toute la logique de collecte d'articles Wikipedia
    pour une langue donnée, en utilisant différentes stratégies selon les
    ressources disponibles et les objectifs de tokens.

    Attributes:
        language_code: code de la langue (ex: 'ru', 'uk')
        categories: liste des catégories à explorer
        params: paramètres adaptatifs de collecte
        already_collected_ids: ensemble des IDs d'articles déjà collectés
        api_url: URL de l'API Wikipedia construite automatiquement
    """

    def __init__(
        self,
        language_code: str,
        categories: List[str],
        params: Optional[Dict[str, Any]] = None,
        already_collected_ids: Optional[Set[int]] = None,
    ) -> None:
        """
        Initialise un collecteur d'articles pour une langue donnée.

        Args:
            language_code: code de la langue (ex: 'ru', 'uk')
            categories: liste des catégories à explorer
            params: paramètres adaptatifs
                (optionnel, récupérés automatiquement si non fournis)
            already_collected_ids: ensemble des IDs d'articles déjà collectés
                (optionnel)

        Raises:
            ValueError: si les paramètres sont invalides
        """
        # Validation des paramètres d'entrée
        if not isinstance(language_code, str) or not language_code.strip():
            raise ValueError("Le code de langue doit être une chaîne non vide")

        if not isinstance(categories, list) or not categories:
            raise ValueError("Les catégories doivent être une liste non vide")

        if not all(isinstance(cat, str) and cat.strip() for cat in categories):
            raise ValueError("Toutes les catégories doivent être des chaînes non vides")

        # Initialisation des attributs
        self.language_code = language_code.strip()
        self.categories = [cat.strip() for cat in categories]
        self.params = params or get_adaptive_params(language_code)
        self.already_collected_ids = already_collected_ids or set()
        self.api_url = f"https://{language_code}.wikipedia.org/w/api.php"

        # Validation des paramètres adaptatifs
        self._validate_adaptive_params()

        logging.info(
            f"ArticleCollector initialisé pour {language_code}"
            f" avec {len(categories)} catégories"
        )

    def _validate_adaptive_params(self) -> None:
        """
        Valide les paramètres adaptatifs du collecteur.

        Raises:
            ValueError: si les paramètres sont invalides
        """
        required_params = [
            "min_char_length",
            "max_token_length",
            "main_category_ratio",
            "subcategory_ratio",
            "random_ratio",
            "fixed_selection_ratio",
        ]

        for param in required_params:
            if param not in self.params:
                raise ValueError(f"Paramètre adaptatif manquant: {param}")

            if (
                not isinstance(self.params[param], (int, float))
                or self.params[param] < 0
            ):
                raise ValueError(
                    f"Paramètre adaptatif invalide {param}: "
                    "doit être un nombre positif"
                )

    def collect_by_category(
        self,
        category_name: str,
        category_target: int,
        num_articles: int = DEFAULT_NUM_ARTICLES,
        fixed_ratio: float = DEFAULT_FIXED_RATIO,
        sleep_time: Tuple[float, float] = DEFAULT_SLEEP_TIME,
    ) -> Tuple[List[Dict[str, Any]], int, int, int]:
        """
        Sélectionne des articles valides par grande catégorie
         avec des paramètres adaptatifs.

        Cette méthode combine la récupération d'articles ordonnés
        (selon le tri par défaut) et d'articles aléatoires
        (selon différents critères de tri) pour diversifier
        le corpus tout en maintenant une certaine cohérence.

        Args:
            category_name: nom de la catégorie à explorer
            category_target: nb de tokens cible pour cette catégorie
            num_articles: nb maximum d'articles à récupérer
            fixed_ratio: proportion d'articles ordonnés vs. aléatoires
                (de 0.0 à 1.0)
            sleep_time: délai entre les requêtes API
                (tuple min, max en secondes)

        Returns:
            tuple (articles_collectés, tokens_totaux, tokens_ordonnés, tokens_aléatoires)

        Raises:
            ValueError: si les paramètres sont invalides
        """
        # Validation des paramètres
        if not isinstance(category_name, str) or not category_name.strip():
            raise ValueError("Le nom de catégorie doit être une chaîne non vide")

        if not isinstance(category_target, int) or category_target <= 0:
            raise ValueError("L'objectif de catégorie doit être un entier positif")

        if not isinstance(num_articles, int) or num_articles <= 0:
            raise ValueError("Le nombre d'articles doit être un entier positif")

        if (
            not isinstance(fixed_ratio, (int, float))
            or not (0.0 <= fixed_ratio <= 1.0)
        ):
            raise ValueError("Le ratio fixe doit être un nombre entre 0.0 et 1.0")

        if not isinstance(sleep_time, tuple) or len(sleep_time) != 2:
            raise ValueError("sleep_time doit être un tuple de 2 éléments")

        category_name = category_name.strip()

        logging.info(
            LOG_MESSAGES["START_CATEGORY"].format(
                lang=self.language_code, cat=category_name
            )
        )

        articles = []

        try:
            # Obtenir le préfixe de catégorie approprié
            prefix = CATEGORY_PREFIXES.get(self.language_code, "Category:")
            full_category = f"{prefix}{category_name}"

            # Récupération d'articles ordonnés et aléatoires
            ordered_members, random_members = self._fetch_category_members(
                full_category, sleep_time
            )

            # Calculer la répartition des articles et tokens
            distribution = self._calculate_article_distribution(
                ordered_members,
                random_members,
                num_articles,
                category_target,
                fixed_ratio,
            )

            self._log_collection_plan(category_name, distribution)

            # Collecter les articles ordonnés
            ordered_articles, ordered_tokens = self._collect_ordered_articles(
                ordered_members, distribution, category_name, sleep_time
            )

            # Collecter les articles aléatoires
            # (en excluant ceux déjà collectés)
            random_articles, random_tokens = self._collect_random_articles(
                random_members,
                distribution,
                category_name,
                sleep_time,
                {article["pageid"] for article in ordered_articles},
            )

            # Combiner tous les articles
            articles = ordered_articles + random_articles
            total_tokens = ordered_tokens + random_tokens

            # Vérifier et signaler les doublons
            self._check_for_duplicates(articles)

            # Logger les statistiques finales
            self._log_collection_statistics(
                category_name,
                articles,
                total_tokens,
                ordered_tokens,
                random_tokens,
                fixed_ratio,
            )

        except Exception as e:
            logging.error(
                f"Erreur lors de la récupération des articles de {full_category}: {e}",
                exc_info=True,
            )
            return [], 0, 0, 0

        logging.info(
            LOG_MESSAGES["END_CATEGORY"].format(
                lang=self.language_code,
                cat=category_name,
                count=len(articles)
            )
        )

        return articles, total_tokens, ordered_tokens, random_tokens

    def _fetch_category_members(
        self,
        full_category: str,
        sleep_time: Tuple[float, float]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Récupère les articles ordonnés et aléatoires d'une catégorie.

        Args:
            full_category: nom complet de la catégorie avec préfixe
            sleep_time: délai entre les requêtes

        Returns:
            tuple (articles_ordonnés, articles_aléatoires)
        """
        # Récupération d'articles dans l'ordre
        ordered_members = fetch_category_articles(
            self.api_url,
            full_category,
            limit=250,
            sort_method="sortkey",
            sort_direction="asc",
            sleep_time=sleep_time,
        )

        # Récupération d'articles aléatoires avec stratégies variées
        random_strategies = [
            {"sort": "sortkey", "dir": "desc"},
            {"sort": "timestamp", "dir": "asc"},
            {"sort": "timestamp", "dir": "desc"},
        ]

        random_strategy = random.choice(random_strategies)

        random_members = fetch_category_articles(
            self.api_url,
            full_category,
            limit=250,
            sort_method=random_strategy["sort"],
            sort_direction=random_strategy["dir"],
            sleep_time=sleep_time,
        )

        return ordered_members, random_members

    def _calculate_article_distribution(
        self,
        ordered_members: List[Dict[str, Any]],
        random_members: List[Dict[str, Any]],
        num_articles: int,
        category_target: int,
        fixed_ratio: float,
    ) -> Dict[str, int]:
        """
        Calcule la distribution des articles et tokens selon les paramètres.

        Args:
            ordered_members: articles ordonnés disponibles
            random_members: articles aléatoires disponibles
            num_articles: nb total d'articles souhaité
            category_target: objectif de tokens pour la catégorie
            fixed_ratio: proportion d'articles ordonnés

        Returns:
            dictionnaire avec la distribution calculée
        """
        valid_members_count = len(ordered_members) + len(random_members)

        if valid_members_count == 0:
            return {
                "num_ordered": 0,
                "num_random": 0,
                "ordered_tokens_target": 0,
                "random_tokens_target": 0,
                "total_available": 0,
            }

        # Calculer le nb d'articles de chaque type
        num_ordered = min(int(num_articles * fixed_ratio), len(ordered_members))
        num_random = min(num_articles - num_ordered, len(random_members))

        # Calculer les objectifs de tokens
        ordered_tokens_target = int(category_target * fixed_ratio)
        random_tokens_target = category_target - ordered_tokens_target

        return {
            "num_ordered": num_ordered,
            "num_random": num_random,
            "ordered_tokens_target": ordered_tokens_target,
            "random_tokens_target": random_tokens_target,
            "total_available": valid_members_count,
        }

    def _log_collection_plan(
        self, category_name: str, distribution: Dict[str, int]
    ) -> None:
        """Affiche le plan de collecte pour la catégorie."""
        print(
            f"  {distribution['total_available']} "
            "articles disponibles dans la catégorie"
        )
        print(
            "  Objectifs en tokens: "
            f"{distribution['ordered_tokens_target']} ordonnés + "
            f"{distribution['random_tokens_target']} aléatoires"
        )

    def _collect_ordered_articles(
        self,
        ordered_members: List[Dict[str, Any]],
        distribution: Dict[str, int],
        category_name: str,
        sleep_time: Tuple[float, float],
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Collecte les articles ordonnés selon l'objectif défini.

        Args:
            ordered_members: liste des articles ordonnés disponibles
            distribution: distribution calculée des articles/tokens
            category_name: nom de la catégorie
            sleep_time: délai entre requêtes

        Returns:
            tuple (articles_collectés, tokens_collectés)
        """
        if distribution["num_ordered"] == 0:
            return [], 0

        fixed_articles = select_valid_articles(
            ordered_members,
            distribution["num_ordered"],
            self.already_collected_ids,
            self.params["min_char_length"],
            self.language_code,
            category_name,
            sleep_time,
            self.api_url,
            ARTICLE_TYPES["ORDERED"],
        )

        # Traiter les articles et collecter les tokens
        processed_articles = []
        total_tokens = 0

        for article in fixed_articles:
            processed_article = process_article(
                article, self.params["max_token_length"]
            )
            processed_articles.append(processed_article)
            total_tokens += processed_article["token_count"]
            self.already_collected_ids.add(processed_article["pageid"])

            print(
                f"  Article ajouté: {processed_article['title']}"
                f" ({processed_article['token_count']} tokens)"
            )

            # Vérifier si l'objectif est atteint
            if total_tokens >= distribution["ordered_tokens_target"]:
                break

        return processed_articles, total_tokens

    def _collect_random_articles(
        self,
        random_members: List[Dict[str, Any]],
        distribution: Dict[str, int],
        category_name: str,
        sleep_time: Tuple[float, float],
        used_ids: Set[int],
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Collecte les articles aléatoires selon l'objectif défini.

        Args:
            random_members: liste des articles aléatoires disponibles
            distribution: distribution calculée des articles/tokens
            category_name: nom de la catégorie
            sleep_time: délai entre requêtes
            used_ids: IDs déjà utilisés à exclure

        Returns:
            tuple (articles_collectés, tokens_collectés)
        """
        if distribution["num_random"] == 0:
            return [], 0

        # Filtrer les articles déjà utilisés
        filtered_random_members = [
            member for member in random_members if (
                member["pageid"] not in used_ids
            )
        ]

        if not filtered_random_members:
            return [], 0

        random_articles = select_valid_articles(
            filtered_random_members,
            distribution["num_random"],
            self.already_collected_ids,
            self.params["min_char_length"],
            self.language_code,
            category_name,
            sleep_time,
            self.api_url,
            ARTICLE_TYPES["RANDOM"],
        )

        # Traiter les articles et collecter les tokens
        processed_articles = []
        total_tokens = 0

        for article in random_articles:
            if total_tokens >= distribution["random_tokens_target"]:
                break

            processed_article = process_article(
                article, self.params["max_token_length"]
            )
            processed_articles.append(processed_article)
            total_tokens += processed_article["token_count"]
            self.already_collected_ids.add(processed_article["pageid"])

            print(
                f"    + Article: {processed_article['title']} "
                f"({processed_article['token_count']} tokens)"
            )

        return processed_articles, total_tokens

    def _check_for_duplicates(self, articles: List[Dict[str, Any]]) -> None:
        """Vérifie et signale les doublons dans la liste d'articles."""
        article_ids = [article["pageid"] for article in articles]
        unique_ids = set(article_ids)

        if len(article_ids) != len(unique_ids):
            duplicate_count = len(article_ids) - len(unique_ids)
            print(LOG_MESSAGES["DUPLICATE_WARNING"].format(
                count=duplicate_count)
            )

            # Identifier les IDs en double
            duplicate_ids = [
                item for item, count in Counter(article_ids).items() if count > 1
            ]
            print(f"IDs en double: {duplicate_ids}")

    def _log_collection_statistics(
        self,
        category_name: str,
        articles: List[Dict[str, Any]],
        total_tokens: int,
        ordered_tokens: int,
        random_tokens: int,
        fixed_ratio: float,
    ) -> None:
        """Affiche les statistiques de collecte pour la catégorie."""
        if not articles:
            return

        ordered_articles_count = sum(
            1 for a in articles if a.get("type") == ARTICLE_TYPES["ORDERED"]
        )
        random_articles_count = sum(
            1 for a in articles if a.get("type") == ARTICLE_TYPES["RANDOM"]
        )

        total_articles = len(articles)

        # Calculer les proportions
        ordered_articles_ratio = (
            ordered_articles_count / total_articles if (
                total_articles > 0
            ) else 0
        )
        random_articles_ratio = (
            random_articles_count / total_articles if total_articles > 0 else 0
        )
        ordered_tokens_ratio = ordered_tokens / total_tokens if total_tokens > 0 else 0
        random_tokens_ratio = random_tokens / total_tokens if (
            total_tokens > 0
        ) else 0

        print(
            f"    Détail: {ordered_tokens} tokens ordonnés "
            f"+ {random_tokens} tokens aléatoires"
        )
        print(f"    Batch: {total_tokens} tokens collectés "
              f"({total_articles} articles)")

        # Logger les statistiques détaillées
        logging.info(
            f"Distribution pour {category_name} "
            f"(langue: {self.language_code}):"
        )
        logging.info(
            f"- Articles: {ordered_articles_ratio:.2f} ordonnés "
            f"vs. {random_articles_ratio:.2f} aléatoires"
        )
        logging.info(
            f"- Tokens: {ordered_tokens_ratio:.2f} ordonnés "
            f"vs. {random_tokens_ratio:.2f} aléatoires"
        )
        logging.info(
            f"- Objectif initial: {fixed_ratio:.2f} ordonnés "
            f"vs. {1-fixed_ratio:.2f} aléatoires"
        )

    def collect_from_subcategories(
        self,
        category_name: str,
        num_articles: int = 10,
        max_depth: int = DEFAULT_MAX_DEPTH,
        sleep_time: Tuple[float, float] = DEFAULT_SLEEP_TIME,
        cached_subcategories: Optional[Set[str]] = None,
        attempt_number: int = 1,
        token_target: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], Optional[Set[str]], int]:
        """
        Récupère des articles aléatoires dans les sous-catégories
        d'une catégorie donnée.

        Cette méthode explore hiérarchiquement les sous-catégories et utilise
        un cache pour optimiser les tentatives successives
        et éviter la re-exploration.

        Args:
            category_name: nom de la catégorie principale à explorer
            num_articles: nb d'articles à récupérer
            max_depth: profondeur maximale de recherche
                dans les sous-catégories
            sleep_time: intervalle d'attente entre les requêtes API
            cached_subcategories: ensemble des sous-catégories déjà trouvées
                (cache)
            attempt_number: numéro de tentative actuelle
                (pour l'exploration secondaire)
            token_target: objectif de tokens à atteindre (optionnel)

        Returns:
            tuple (articles_collectés, cache_mis_à_jour, total_tokens)

        Raises:
            ValueError: si les paramètres sont invalides
        """
        # Validation des paramètres
        if not isinstance(category_name, str) or not category_name.strip():
            raise ValueError("Le nom de catégorie doit être une chaîne non vide")

        if not isinstance(num_articles, int) or num_articles <= 0:
            raise ValueError("Le nombre d'articles doit être un entier positif")

        if not isinstance(max_depth, int) or max_depth < 0:
            raise ValueError(
                "La profondeur maximale doit être un entier positif ou zéro"
            )

        if not isinstance(attempt_number, int) or attempt_number < 1:
            raise ValueError("Le numéro de tentative doit être un entier positif")

        if token_target is not None and (
            not isinstance(token_target, int) or token_target <= 0
        ):
            raise ValueError("L'objectif de tokens doit être un entier positif ou None")

        category_name = category_name.strip()

        logging.info(
            LOG_MESSAGES["START_SUBCATEGORIES"].format(
                lang=self.language_code, cat=category_name
            )
        )

        articles = []
        total_tokens = 0

        prefix = CATEGORY_PREFIXES.get(self.language_code, "Category:")
        full_category = f"{prefix}{category_name}"

        # Gérer le cache des sous-catégories
        if cached_subcategories is None:
            all_subcategories = self._explore_subcategories_hierarchy(
                full_category, max_depth, sleep_time
            )
            cached_subcategories = all_subcategories
        else:
            all_subcategories = cached_subcategories
            print(
                f"    Utilisation de {len(all_subcategories)} "
                f"sous-catégories déjà explorées pour '{category_name}'."
            )

        # Exploration secondaire pour les tentatives ultérieures
        if attempt_number > 1:
            additional_subcats = self._secondary_exploration(
                all_subcategories,
                max_depth,
                sleep_time,
                attempt_number
            )
            all_subcategories.update(additional_subcats)
            cached_subcategories = all_subcategories

        # Si aucune sous-catégorie n'est trouvée, renvoyer des résultats vides
        if not all_subcategories:
            return [], None, 0

        # Sélectionner et explorer les sous-catégories
        articles, total_tokens = self._explore_selected_subcategories(
            all_subcategories,
            category_name,
            num_articles,
            sleep_time,
            token_target
        )

        print(
            f"    Batch: {len(articles)} articles collectés, "
            f"total tokens: {total_tokens} tokens"
        )
        logging.info(LOG_MESSAGES["END_SUBCATEGORIES"].format(
            count=len(articles))
        )

        return articles, cached_subcategories, total_tokens

    def _explore_subcategories_hierarchy(
        self,
        full_category: str,
        max_depth: int,
        sleep_time: Tuple[float, float]
    ) -> Set[str]:
        """
        Explore la hiérarchie des sous-catégories en largeur d'abord (BFS).

        Args:
            full_category: nom complet de la catégorie racine
            max_depth: profondeur maximale d'exploration
            sleep_time: délai entre requêtes

        Returns:
            ensemble de toutes les sous-catégories trouvées
        """
        all_subcategories = set()
        subcategory_queue = [(full_category, 0)]  # (catégorie, profondeur)

        while subcategory_queue and (
            len(all_subcategories) < MAX_SUBCATEGORIES_CACHE
        ):
            current_category, depth = subcategory_queue.pop(0)

            # Éviter la re-exploration et respecter la profondeur max
            if current_category in all_subcategories or depth > max_depth:
                continue

            all_subcategories.add(current_category)

            # Chercher les sous-catégories si on n'est pas à la profondeur max
            if depth < max_depth:
                subcats = fetch_subcategories(
                    self.api_url, current_category, 50, sleep_time
                )

                for subcat in subcats:
                    if "title" in subcat:
                        subcategory_queue.append((subcat["title"], depth + 1))

        category_short_name = (
            full_category.split(":")[-1] if (
            ":" in full_category
            ) else full_category
        )
        print(
            f"    {len(all_subcategories)} sous-catégories trouvée(s) "
            f"pour '{category_short_name}'."
        )

        return all_subcategories

    def _secondary_exploration(
        self,
        existing_subcategories: Set[str],
        max_depth: int,
        sleep_time: Tuple[float, float],
        attempt_number: int,
    ) -> Set[str]:
        """
        Effectue une exploration secondaire à partir d'un point aléatoire.

        Args:
            existing_subcategories: sous-catégories déjà connues
            max_depth: profondeur max
            sleep_time: délai entre requêtes
            attempt_number: numéro de tentative

        Returns:
            nouvelles sous-catégories découvertes
        """
        if len(existing_subcategories) <= 10:
            return set()

        # Sélectionner un point de départ aléatoire
        subcats_list = list(existing_subcategories)
        random_subcat = random.choice(subcats_list)

        print(
            LOG_MESSAGES["SECONDARY_EXPLORATION"].format(
                attempt=attempt_number, subcat=random_subcat
            )
        )

        # Explorer à partir de ce point
        new_queue = [(random_subcat, 0)]
        new_explored = set([random_subcat])
        secondary_depth = max(2, max_depth // 2)

        while new_queue and len(new_explored) < MAX_SECONDARY_EXPLORATION:
            current_cat, depth = new_queue.pop(0)

            if depth > secondary_depth:
                continue

            new_subcats = fetch_subcategories(
                self.api_url,
                current_cat,
                50,
                sleep_time
            )

            for subcat in new_subcats:
                if "title" in subcat:
                    subcat_title = subcat["title"]
                    if (
                        subcat_title not in existing_subcategories
                        and subcat_title not in new_explored
                    ):
                        new_queue.append((subcat_title, depth + 1))
                        new_explored.add(subcat_title)

        discovered_count = len(new_explored) - 1  # -1 car on exclut le point de départ
        print(
            "  Exploration secondaire: "
            f"{discovered_count} sous-catégories découvertes"
        )

        return new_explored

    def _explore_selected_subcategories(
        self,
        all_subcategories: Set[str],
        category_name: str,
        num_articles: int,
        sleep_time: Tuple[float, float],
        token_target: Optional[int],
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Explore un échantillon de sous-catégories pour collecter des articles.

        Args:
            all_subcategories: toutes les sous-catégories disponibles
            category_name: nom de la catégorie principale
            num_articles: nb d'articles à collecter
            sleep_time: délai entre requêtes
            token_target: objectif de tokens (optionnel)

        Returns:
            tuple (articles_collectés, total_tokens)
        """
        # Sélectionner un échantillon de sous-catégories
        # avec rotation aléatoire
        subcategories_list = list(all_subcategories)
        selected_subcats = self._select_subcategories_sample(
            subcategories_list
        )

        articles = []
        total_tokens = 0
        articles_collected_count = 0
        subcats_explored = 0

        # Explorer les sous-catégories sélectionnées
        for subcat in selected_subcats:
            if articles_collected_count >= num_articles:
                break

            if token_target and total_tokens >= token_target:
                print(
                    LOG_MESSAGES["TARGET_REACHED"].format(
                        current=total_tokens,
                        target=token_target
                    )
                )
                break

            subcats_explored += 1
            (
                batch_articles,
                batch_tokens
            ) = self._collect_from_single_subcategory(
                subcat,
                category_name,
                sleep_time,
                min(5, num_articles - articles_collected_count),
            )

            articles.extend(batch_articles)
            total_tokens += batch_tokens
            articles_collected_count += len(batch_articles)

        # Explorer des sous-catégories supplémentaires si nécessaire
        if articles_collected_count < num_articles and subcats_explored < len(
            selected_subcats
        ):
            additional_articles, additional_tokens = (
                self._explore_additional_subcategories(
                    selected_subcats,
                    subcats_explored,
                    category_name,
                    num_articles - articles_collected_count,
                    sleep_time,
                    token_target,
                    total_tokens,
                )
            )

            articles.extend(additional_articles)
            total_tokens += additional_tokens

        return articles, total_tokens

    def _select_subcategories_sample(
            self,
            subcategories_list: List[str]
    ) -> List[str]:
        """
        Sélectionne un échantillon de sous-catégories avec rotation aléatoire.

        Args:
            subcategories_list: Liste complète des sous-catégories

        Returns:
            Liste des sous-catégories sélectionnées
        """
        if len(subcategories_list) > 50:
            # Rotation aléatoire pour diversifier les points de départ
            start_idx = random.randint(0, len(subcategories_list) - 1)
            reorganized_list = (
                subcategories_list[start_idx:] + subcategories_list[:start_idx]
            )
            return reorganized_list[:50]
        else:
            return subcategories_list

    def _collect_from_single_subcategory(
        self,
        subcat: str,
        category_name: str,
        sleep_time: Tuple[float, float],
        max_articles: int,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Collecte des articles depuis une seule sous-catégorie.

        Args:
            subcat: nom de la sous-catégorie
            category_name: nom de la catégorie principale
            sleep_time: délai entre requêtes
            max_articles: nb maximum d'articles à collecter

        Returns:
            tuple (articles_collectés, tokens_collectés)
        """
        members = fetch_category_articles(
            self.api_url,
            subcat,
            50,
            sleep_time=sleep_time
        )

        if not members:
            return [], 0

        # Sélectionner aléatoirement quelques articles
        sample_size = min(
            len(members),
            DEFAULT_SAMPLE_SIZE,
            max_articles
        )
        selected_members = random.sample(members, sample_size)

        articles = []
        total_tokens = 0

        for member in selected_members:
            validated_article = validate_article(
                member,
                self.language_code,
                f"{category_name} (Sous-catégorie)",
                self.params["min_char_length"],
                self.already_collected_ids,
                self.api_url,
                sleep_time,
            )

            if validated_article:
                processed_article = process_article(
                    validated_article,
                    self.params["max_token_length"]
                )
                articles.append(processed_article)
                self.already_collected_ids.add(processed_article["pageid"])
                total_tokens += processed_article["token_count"]

                print(
                    f"    + Article: {processed_article['title']} "
                    f"({processed_article['token_count']} tokens)"
                )
            else:
                # Logger les articles ignorés
                page_id = member.get("pageid")
                title = member.get("title", "Titre inconnu")

                if page_id and page_id not in self.already_collected_ids:
                    print(f"    - Article ignoré (trop court): {title}")

        return articles, total_tokens

    def _explore_additional_subcategories(
        self,
        selected_subcats: List[str],
        subcats_explored: int,
        category_name: str,
        articles_needed: int,
        sleep_time: Tuple[float, float],
        token_target: Optional[int],
        current_tokens: int,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Explore des sous-catégories supplémentaires
        si l'objectif n'est pas atteint.

        Args:
            selected_subcats: liste des sous-catégories sélectionnées
            subcats_explored: nb de sous-catégories déjà explorées
            category_name: nom de la catégorie principale
            articles_needed: nb d'articles encore nécessaires
            sleep_time: délai entre requêtes
            token_target: objectif de tokens (optionnel)
            current_tokens: tokens déjà collectés

        Returns:
            tuple (articles_supplémentaires, tokens_supplémentaires)
        """
        print(
            "  Objectif non atteint, exploration de sous-catégories supplémentaires..."
        )

        additional_subcats = selected_subcats[subcats_explored:]
        articles = []
        total_tokens = 0
        empty_subcats_count = 0

        for subcat in additional_subcats[:DEFAULT_ADDITIONAL_SUBCATS]:
            if len(articles) >= articles_needed:
                break

            if token_target and (
                (current_tokens + total_tokens) >= token_target
            ):
                print(
                    LOG_MESSAGES["TARGET_REACHED"].format(
                        current=current_tokens + total_tokens,
                        target=token_target
                    )
                )
                break

            members = fetch_category_articles(
                self.api_url,
                subcat,
                50,
                sleep_time=sleep_time
            )

            if members:
                empty_subcats_count = 0  # Réinitialiser le compteur

                (
                    batch_articles,
                    batch_tokens
                ) = self._collect_from_single_subcategory(
                    subcat,
                    category_name,
                    sleep_time,
                    min(5, articles_needed - len(articles)),
                )

                articles.extend(batch_articles)
                total_tokens += batch_tokens
            else:
                empty_subcats_count += 1

                if empty_subcats_count >= MAX_EMPTY_SUBCATEGORIES:
                    print(
                        "  Trop de sous-catégories vides consécutives, "
                        "arrêt de la recherche supplémentaire."
                    )
                    break

        if len(articles) < articles_needed:
            print(
                f"  Terminé avec {len(articles)} articles "
                f"sur {articles_needed} objectif"
            )

        return articles, total_tokens

    def collect_random(
        self,
        num_articles: int = DEFAULT_NUM_ARTICLES,
        sleep_time: Tuple[float, float] = DEFAULT_SLEEP_TIME,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Récupère aléatoirement des articles de Wikipédia
        dans la langue spécifiée.

        Cette méthode utilise l'endpoint "random" de l'API MediaWiki
        pour obtenir des articles complètement aléatoires,
        ce qui permet de diversifier le corpus
        au-delà des catégories spécifiques.

        Args:
            num_articles: nb d'articles à récupérer
            sleep_time: tuple (min, max) pour le temps d'attente
                entre les requêtes

        Returns:
            tuple (articles_collectés, total_tokens)

        Raises:
            ValueError: si les paramètres sont invalides
        """
        # Validation des paramètres
        if not isinstance(num_articles, int) or num_articles <= 0:
            raise ValueError("Le nombre d'articles doit être un entier positif")

        if not isinstance(sleep_time, tuple) or len(sleep_time) != 2:
            raise ValueError("sleep_time doit être un tuple de 2 éléments")

        if not all(isinstance(t, (int, float)) and (
                   t >= 0 for t in sleep_time)
        ):
            raise ValueError(
                "Les éléments de sleep_time doivent être des nombres positifs"
            )

        logging.info(
            LOG_MESSAGES["START_RANDOM"].format(
                lang=self.language_code,
                target=num_articles
            )
        )

        articles = []
        total_tokens = 0

        # Compteurs pour éviter les boucles infinies
        articles_collected = 0
        attempts = 0
        max_attempts = num_articles * MAX_RANDOM_ATTEMPTS_MULTIPLIER

        # Continuer jusqu'à obtenir le nombre d'articles demandé
        # ou atteindre le max de tentatives
        while articles_collected < num_articles and attempts < max_attempts:
            attempts += 1

            try:
                # Obtenir un article aléatoire
                random_article = fetch_random_article(self.api_url, sleep_time)

                if random_article:
                    # Valider l'article
                    validated_article = validate_article(
                        random_article,
                        self.language_code,
                        "Random",
                        self.params["min_char_length"],
                        self.already_collected_ids,
                        self.api_url,
                        sleep_time,
                    )

                    if validated_article:
                        # Traiter l'article pour limiter les tokens
                        processed_article = process_article(
                            validated_article,
                            self.params["max_token_length"]
                        )
                        articles.append(processed_article)
                        self.already_collected_ids.add(
                            processed_article["pageid"]
                        )

                        # Mettre à jour les compteurs
                        articles_collected += 1
                        total_tokens += processed_article["token_count"]

                        print(
                            f"    + Article aléatoire: {processed_article['title']}"
                            f" ({processed_article['token_count']} tokens)"
                        )

            except Exception as e:
                logging.warning(
                    "Erreur lors de la récupération d'un article aléatoire"
                    f" (tentative {attempts}): {e}"
                )
                continue

        # Vérifier si l'objectif a été atteint
        if articles_collected < num_articles:
            logging.warning(
                LOG_MESSAGES["RANDOM_WARNING"].format(
                    found=articles_collected,
                    target=num_articles,
                    lang=self.language_code,
                )
            )

        print(
            f"    Batch: {articles_collected} articles aléatoires collectés,"
            f" {total_tokens} tokens"
        )
        logging.info(
            LOG_MESSAGES["END_RANDOM"].format(
                lang=self.language_code,
                count=articles_collected
            )
        )

        return articles, total_tokens

    def get_collection_statistics(self) -> Dict[str, Any]:
        """
        Retourne des statistiques sur l'état actuel du collecteur.

        Returns:
            dictionnaire avec les statistiques de collecte
        """
        return {
            "language_code": self.language_code,
            "categories_count": len(self.categories),
            "categories": self.categories.copy(),
            "collected_articles_count": len(self.already_collected_ids),
            "collected_article_ids": self.already_collected_ids.copy(),
            "adaptive_params": self.params.copy(),
            "api_url": self.api_url,
        }

    def reset_collection_state(self) -> None:
        """
        Remet à zéro l'état de collecte du collecteur.

        Cette méthode vide la liste des articles déjà collectés,
        permettant de recommencer une collecte propre.
        """
        self.already_collected_ids.clear()
        logging.info(f"État de collecte remis à zéro pour {self.language_code}")

    def add_collected_ids(
            self,
            article_ids: Union[Set[int], List[int]]
    ) -> None:
        """
        Ajoute des IDs d'articles à la liste des articles déjà collectés.

        Args:
            article_ids: ensemble ou liste d'IDs d'articles à ajouter

        Raises:
            ValueError: si les IDs ne sont pas valides
        """
        if isinstance(article_ids, list):
            article_ids = set(article_ids)

        if not isinstance(article_ids, set):
            raise ValueError("article_ids doit être un ensemble ou une liste")

        # Valider que tous les IDs sont des entiers positifs
        invalid_ids = [
            id_ for id_ in article_ids if (
                not isinstance(id_, int)
            ) or id_ <= 0
        ]
        if invalid_ids:
            raise ValueError(f"IDs d'articles invalides: {invalid_ids}")

        initial_count = len(self.already_collected_ids)
        self.already_collected_ids.update(article_ids)
        added_count = len(self.already_collected_ids) - initial_count

        logging.info(
            f"{added_count} nouveaux IDs d'articles "
            f"ajoutés au collecteur {self.language_code}"
        )

    def __repr__(self) -> str:
        """
        Representation textuelle du collecteur.

        Returns:
            chaîne décrivant le collecteur
        """
        return (
            f"ArticleCollector(language='{self.language_code}', "
            f"categories={len(self.categories)}, "
            f"collected_articles={len(self.already_collected_ids)})"
        )

    def __str__(self) -> str:
        """
        Version lisible du collecteur.

        Returns:
            description textuelle du collecteur
        """
        return (
            f"Collecteur d'articles Wikipedia pour {self.language_code} "
            f"({len(self.categories)} catégories, "
            f"{len(self.already_collected_ids)} articles déjà collectés)"
        )
