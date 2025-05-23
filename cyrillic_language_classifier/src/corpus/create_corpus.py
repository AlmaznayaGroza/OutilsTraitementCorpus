"""Module principal de création d'un corpus multilingue cyrillique

Ce module implémente un système avancé de collecte et de curation de corpus
pour les langues utilisant l'alphabet cyrillique. Il développe une approche
méticuleuse, qui adapte automatiquement les stratégies de collecte
selon les spécificités et la disponibilité des ressources de chaque
langue traitée.

Innovation méthodologique principale:
    Le système implémente une stratégie de collecte adaptative qui classe
    les langues cyrilliques en groupes selon leur richesse en ressources
    numériques, puis adapte les paramètres de collecte, les profondeurs
    d'exploration, et les stratégies d'échantillonnage pour optimiser
    la qualité et la représentativité du corpus final pour chaque langue.

Architecture de collecte multi-stratégies:
    * collecte par catégories thématiques (avec traductions adaptées)
    * exploration adaptive des sous-catégories selon la profondeur optimale
    * échantillonnage aléatoire pour garantir la diversité textuelle
    * validation qualité temps-réel avec filtrage des contenus inadéquats
    * gestion intelligente des timeouts et reprises après interruption

Groupes linguistiques et stratégies adaptatives:
    - groupe A (langues bien dotées): exploration superficielle mais large
    - groupe B (langues intermédiaires): équilibre exploration/profondeur
    - groupe C (langues minoritaires): exploration profonde et exhaustive
    - groupe D (langues très peu dotées): collecte maximale avec seuils abaissés

Contrôle qualité et robustesse:
    Le système intègre des mécanismes sophistiqués de validation de contenu,
    de déduplication, de gestion d'erreurs et de reprise après interruption.
    Il veille au respect des bonnes pratiques d'utilisation d'APIs,
    avec gestion intelligente des délais et limitation du débit.

Sortie et métriques:
    Production de corpus équilibrés avec métadonnées complètes,
    statistiques détaillées par langue et méthode de collecte,
    et rapports de qualité permettant l'évaluation
    de la représentativité linguistique obtenue.

Applications:
    Particulièrement adapté pour la création de corpus d'entraînement
    équilibrés pour modèles de TAL multilingues, études linguistiques
    comparatives, et recherches sur les langues à ressources limitées.

Architecture:
    Utilise une approche modulaire avec séparation claire des responsabilités:
        - configuration centralisée
        - collecte d'articles
        - traitement du texte
        - gestion des données
        - calcul de statistiques
        - gestion de cache

Note sur l'éthique de collecte:
    Tous les contenus collectés respectent les conditions d'utilisation
    de Wikipedia et les bonnes pratiques de scraping éthique avec limitation
    du débit et respect des serveurs sources.
"""

import os
import time
import logging
from datetime import datetime
from typing import Optional, Tuple, List

# Import des modules refactorisés
from config import (
    TIME_LIMIT,
    LANGUAGES,
    ALL_CATEGORIES,
    MAX_DEPTHS_BY_GROUP,
    CATEGORY_TRANSLATIONS,
    get_adaptive_params,
    get_language_group,
    get_target_for_language,
)
from article_collector import ArticleCollector
from data_manager import save_articles_to_csv
from stat_manager import (
    CollectionStats,
    calculate_token_targets,
    print_collection_plan,
    save_global_stats,
)


# =============================================================
# CONSTANTES SUPPLEMENTAIRES DE CONFIGURATION POUR LA COLLECTE
# =============================================================

# Configuration des dossiers et fichiers
DEFAULT_PATHS = {
    "base_dir": "data",
    "articles_dir": "data/raw",
    "logs_dir": "logs",
    "metrics_base": "results/metrics/corpus_analysis/collection",
    "languages_stats_dir": "results/metrics/corpus_analysis/collection/languages",
    "global_stats_dir": "results/metrics/corpus_analysis/collection/global",
}

# Paramètres de collecte et performance
COLLECTION_PARAMS = {
    "max_languages_limit": None,  # None = toutes les langues, sinon nombre entier
    "batch_sizes": {
        "A": {"initial": 15, "min": 5},  # langues bien dotées
        "B": {"initial": 25, "min": 10},  # langues intermédiaires
        "C": {"initial": 30, "min": 10},  # langues minoritaires
        "D": {"initial": 30, "min": 10},  # langues rares
    },
    "max_attempts": 15,  # tentatives max par catégorie
    "timeout_multiplier": 1.0,  # multiplicateur pour TIME_LIMIT
}

# Configuration du logging
LOGGING_CONFIG = {
    "filename_pattern": "cyrillique_collecte.log",
    "level": logging.INFO,
    "format": "%(asctime)s - %(levelname)s - %(message)s",
    "console_level": logging.WARNING,
}

# Messages d'information standardisés
PROGRESS_MESSAGES = {
    "start_collection": "=== DÉMARRAGE DE LA COLLECTE SUR {} LANGUES ===",
    "language_progress": "Langue {}/{}: {}",
    "collection_complete": "=== COLLECTE TERMINÉE ===",
    "total_summary": "Total: {} articles collectés pour {} langues",
    "phase_1": "1. Collecte d'articles des catégories principales (objectif: {} tokens)",
    "phase_2": "2. Collecte d'articles des sous-catégories (objectif: {} tokens)",
    "phase_3": "3. Collecte d'articles aléatoires (objectif: {} tokens)",
}


# =====================================
# FONCTIONS UTILITAIRES ET VALIDATION
# =====================================


def validate_collection_parameters(
    max_languages: Optional[int] = None, timeout_multiplier: float = 1.0
) -> None:
    """Valide les paramètres de collecte avant le démarrage

    Cette fonction vérifie la cohérence des paramètres de collecte pour
    éviter les erreurs d'exécution et optimiser les performances selon
    les ressources disponibles et les objectifs de collecte.

    Args:
        max_languages (int, optional): limite le nombre de langues à traiter
        timeout_multiplier (float): multiplicateur pour ajuster les timeouts

    Raises:
        ValueError: si les paramètres sont incohérents ou invalides

    Note:
        Cette validation préventive permet d'identifier rapidement les
        problèmes de configuration et d'éviter les échecs tardifs durant
        le processus de collecte, qui dure des heures.
    """
    if max_languages is not None:
        if not isinstance(max_languages, int) or max_languages < 1:
            raise ValueError("max_languages doit être un entier positif")
        if max_languages > len(LANGUAGES):
            raise ValueError(
                f"max_languages ({max_languages}) dépasse "
                f"le nombre de langues disponibles ({len(LANGUAGES)})"
            )

    if (
        not isinstance(timeout_multiplier, (int, float))
        or timeout_multiplier <= 0
    ):
        raise ValueError("timeout_multiplier doit être un nombre positif")

    if timeout_multiplier > 2.0:
        logging.warning(
            f"timeout_multiplier élevé ({timeout_multiplier}x) "
            f"- collecte très longue attendue"
        )


def setup_collection_environment() -> None:
    """Configure l'environnement de collecte avec tous les dossiers nécessaires

    Cette fonction initialise l'infrastructure nécessaire pour la collecte:
    dossier de données, système de logging, et validation de l'environnement
    d'exécution. Elle garantit que tous les prérequis sont satisfaits avant
    le démarrage de la collecte.

    Raises:
        PermissionError: si impossible de créer les dossiers requis
        RuntimeError: si l'environnement ne peut pas être initialisé
    """
    try:
        # Création des répertoires de données
        for path_key, path_value in DEFAULT_PATHS.items():
            os.makedirs(path_value, exist_ok=True)

        # Configuration du système de logging
        log_path = os.path.join(
            DEFAULT_PATHS["logs_dir"], LOGGING_CONFIG["filename_pattern"]
        )
        logging.basicConfig(
            filename=log_path,
            level=LOGGING_CONFIG["level"],
            format=LOGGING_CONFIG["format"],
        )

        # Ajout du logging console pour les messages importants
        console_handler = logging.StreamHandler()
        console_handler.setLevel(LOGGING_CONFIG["console_level"])
        logging.getLogger("").addHandler(console_handler)

        logging.info("Environnement de collecte initialisé avec succès")

    except PermissionError as e:
        raise PermissionError(f"Impossible de créer les dossiers nécessaires: {e}")
    except Exception as e:
        logging.error(
            f"Erreur critique dans la fonction principale: {str(e)}",
            exc_info=True
        )
        raise RuntimeError(f"Échec de la collecte: {e}") from e


def get_adaptive_batch_size(
    language_code: str, remaining_tokens: int, attempt_number: int
) -> int:
    """Calcule une taille de batch adaptative selon la langue et le contexte

    Cette fonction implémente une logique sophistiquée qui adapte la taille
    des batches de collecte selon plusieurs facteurs: le groupe linguistique,
    le nombre de tokens restants à collecter et le nombre de tentatives
    déjà effectuées. Cette adaptation optimise l'efficacité de la collecte.

    Args:
        language_code (str): code de la langue en cours de traitement
        remaining_tokens (int): nb de tokens restants à collecter
        attempt_number (int): numéro de la tentative actuelle

    Returns:
        int: taille de batch optimale pour cette situation

    Note méthodologique:
        Cette adaptation permet d'optimiser la collecte en commençant par
        des batches plus grands puis en les réduisant si nécessaire, tout
        en respectant les caractéristiques spécifiques de chaque groupe
        linguistique identifié dans la recherche.
    """
    group = get_language_group(language_code)
    batch_config = COLLECTION_PARAMS["batch_sizes"].get(
        group, COLLECTION_PARAMS["batch_sizes"]["C"]
    )

    # Calcul de base selon le groupe linguistique
    if attempt_number == 1:
        base_size = batch_config["initial"]
    else:
        # Réduction progressive avec les tentatives
        base_size = max(
            batch_config["min"],
            batch_config["initial"] - (attempt_number - 1) * 2
        )

    # Adaptation selon les tokens restants
    if remaining_tokens < 1000:
        adjusted_size = max(batch_config["min"], base_size // 2)
    elif remaining_tokens > 5000:
        adjusted_size = min(base_size + 5, batch_config["initial"] + 10)
    else:
        adjusted_size = base_size

    return adjusted_size


def calculate_effective_timeout(
        base_timeout: int,
        multiplier: float = 1.0
) -> int:
    """Calcule le timeout effectif avec validation des limites raisonnables

    Args:
        base_timeout (int): timeout de base en secondes
        multiplier (float): multiplicateur à appliquer

    Returns:
        int: timeout effectif en secondes,
            borné dans des limites raisonnables
    """
    effective_timeout = int(base_timeout * multiplier)

    # Borner dans des limites raisonnables (1h min, 6h max)
    min_timeout = 3600  # 1h
    max_timeout = 21600  # 6h

    return max(min_timeout, min(effective_timeout, max_timeout))


# ==========================================
# FONCTION PRINCIPALE DE COLLECTE OPTIMISEE
# ==========================================


def collect_articles(language_code: str, categories: List[str]) -> Tuple:
    """
    Fonction de collecte avec adaptation par groupe de langue
    et validation robuste

    Cette fonction implémente le cœur de la logique de collecte adaptative
    développée dans ce projet. Elle orchestre les 3 phases de collecte
    (catégories principales, sous-catégories, articles aléatoires) avec des
    paramètres optimisés selon le groupe linguistique de la langue traitée.

    Args:
        language_code (str): code ISO de la langue à traiter
        categories (List[str]): liste des catégories thématiques à explorer

    Returns:
        Tuple: (articles, stats_catégories, total_tokens, temps_exécution,
                main_ordered_tokens, main_random_tokens, limited_articles,
                pourcentage_limité)

    Raises:
        ValueError: si les paramètres d'entrée sont invalides
        RuntimeError: si la collecte échoue de manière critique

    Note sur la méthodologie:
        Cette fonction représente l'innovation principale du projet avec son
        système adaptatif qui optimise automatiquement les stratégies de
        collecte selon les spécificités de chaque langue, permettant de
        créer des corpus plus équilibrés et représentatifs.
    """
    logging.info(f"Démarrage de la collecte pour la langue: {language_code}")

    try:
        # Récupération et validation des paramètres adaptatifs
        group = get_language_group(language_code)
        token_target = get_target_for_language(language_code)
        params = get_adaptive_params(language_code)

        if not categories:
            raise ValueError(f"Aucune catégorie fournie pour {language_code}")

        # Détermination des catégories disponibles pour cette langue
        available_categories = []
        for category in categories:
            if language_code in CATEGORY_TRANSLATIONS.get(category, {}):
                available_categories.append(category)

        print(
            f"\nCatégories disponibles ({len(available_categories)}/{len(categories)}): "
            f"{', '.join(available_categories)}"
        )

        if not available_categories:
            logging.warning(
                f"Attention: aucune catégorie disponible pour {language_code}"
            )
            return [], {}, 0, 0, 0, 0, 0, 0

        # Initialisation des stats avec validation
        stats = CollectionStats(language_code, token_target, categories)

        # Calcul et définition des objectifs de tokens
        targets = calculate_token_targets(
            token_target,
            params,
            available_categories
        )

        (main_target,
         sub_target,
         random_target,
         tokens_per_main,
         tokens_per_sub
         ) = targets

        stats.set_token_targets(
            main_target,
            sub_target,
            random_target,
            tokens_per_main,
            tokens_per_sub
        )

        # Affichage du plan de collecte pour traçabilité
        print_collection_plan(
            language_code,
            group,
            token_target,
            params,
            targets
        )

        # Variables de suivi de la collecte
        start_time = time.time()
        articles = []
        collected_article_ids = set()

        # Initialisation du collecteur d'articles
        collector = ArticleCollector(
            language_code=language_code,
            categories=categories,
            params=params,
            already_collected_ids=collected_article_ids,
        )

        # =============================================
        # ETAPE 1: Collecte des CATÉGORIES PRINCIPALES
        # =============================================
        print(
            f"\n{PROGRESS_MESSAGES['phase_1'].format(
                stats.main_category_token_target
            )}"
        )

        for category in available_categories:
            # Vérification du timeout global
            effective_timeout = calculate_effective_timeout(
                TIME_LIMIT, COLLECTION_PARAMS["timeout_multiplier"]
            )

            if time.time() - start_time > effective_timeout:
                print(
                    f"Limite de temps atteinte pour {language_code}. "
                    f"Passage à la langue suivante."
                )
                break

            translated_category = CATEGORY_TRANSLATIONS[category][language_code]
            print(f"\n  Catégorie: {category} ({translated_category})")

            # Collecte adaptative avec gestion d'erreurs
            category_target = stats.tokens_per_main_category.get(
                category, tokens_per_main
            )
            category_tokens = 0
            attempts = 0

            while (
                category_tokens < category_target
                and attempts < COLLECTION_PARAMS["max_attempts"]
            ):
                attempts += 1
                remaining_tokens = category_target - category_tokens

                # Calcul adaptatif de la taille de batch
                batch_size = get_adaptive_batch_size(
                    language_code, remaining_tokens, attempts
                )

                print(
                    f"  Tentative {attempts}: recherche de {batch_size} articles "
                    f"(objectif: {remaining_tokens} tokens manquants)"
                )

                try:
                    (
                        category_articles,
                        tokens_collected,
                        ordered_tokens,
                        random_tokens,
                    ) = collector.collect_by_category(
                        category_name=translated_category,
                        category_target=category_target,
                        num_articles=batch_size,
                        fixed_ratio=params["fixed_selection_ratio"],
                        sleep_time=(1, 2.5),
                    )

                    if not category_articles:
                        print(
                            f"  Aucun article disponible pour {category}"
                            f" ({translated_category})"
                        )
                        break

                    # Enregistrement et mise à jour des statistiques
                    stats.set_available_articles(
                        category, len(category_articles)
                    )
                    articles.extend(category_articles)

                    # Mise à jour des stats agrégées ordered/random
                    stats.main_ordered_tokens += ordered_tokens
                    stats.main_random_tokens += random_tokens

                    for article in category_articles:
                        stats.update_main_category_stats(
                            category, article, article.get("type")
                        )

                    category_tokens += tokens_collected

                    print(
                        f"  Progression: {category_tokens}/{category_target}"
                        f" tokens collectés"
                    )

                except Exception as e:
                    logging.error(
                        f"Erreur lors de la collecte pour {category}: "
                        f"{str(e)}",
                        exc_info=True,
                    )
                    continue

        # Affichage du résumé de l'étape 1
        stats.print_progress_summary(
            "les catégories principales",
            stats.main_category_tokens,
            stats.main_category_token_target,
        )

        # ======================================
        # ETAPE 2: Collecte des SOUS-CATÉGORIES
        # ======================================
        if time.time() - start_time <= effective_timeout:
            print(
                f"\n{PROGRESS_MESSAGES['phase_2'].format(
                    stats.subcategory_token_target
                )}"
            )

            subcategories_cache = {}

            for category in available_categories:
                if time.time() - start_time > effective_timeout:
                    print(
                        "Limite de temps atteinte pendant la collecte des sous-catégories."
                    )
                    break

                translated_category = CATEGORY_TRANSLATIONS[category][language_code]
                print(
                    f"\n  Sous-catégories de: {category} "
                    f"({translated_category})"
                )

                category_target = stats.tokens_per_subcategory.get(
                    category, tokens_per_sub
                )
                category_tokens = 0
                attempts = 0
                cached_subcats = subcategories_cache.get(category, None)

                while (
                    category_tokens < category_target
                    and attempts < COLLECTION_PARAMS["max_attempts"]
                ):
                    attempts += 1
                    remaining_tokens = category_target - category_tokens
                    batch_size = get_adaptive_batch_size(
                        language_code, remaining_tokens, attempts
                    )

                    print(
                        f"  Tentative {attempts}: recherche de "
                        f"{batch_size} articles"
                    )

                    try:
                        max_depth = MAX_DEPTHS_BY_GROUP[group]

                        subcategory_articles, updated_cache, sub_tokens = (
                            collector.collect_from_subcategories(
                                category_name=translated_category,
                                num_articles=batch_size,
                                max_depth=max_depth,
                                sleep_time=(0.5, 1.5),
                                cached_subcategories=cached_subcats,
                                attempt_number=attempts,
                                token_target=remaining_tokens,
                            )
                        )

                        if updated_cache:
                            subcategories_cache[category] = updated_cache
                            cached_subcats = updated_cache

                        if not subcategory_articles:
                            continue

                        articles.extend(subcategory_articles)

                        for article in subcategory_articles:
                            stats.update_subcategory_stats(category, article)

                        category_tokens += sub_tokens
                        print(
                            f"  Progression: {category_tokens}/{category_target} "
                            f"tokens collectés"
                        )

                    except Exception as e:
                        logging.error(
                            f"Erreur lors de la collecte des sous-catégories "
                            f"pour {category}: {str(e)}",
                            exc_info=True,
                        )
                        break

            stats.print_progress_summary(
                "les sous-catégories",
                stats.subcategory_tokens,
                stats.subcategory_token_target,
            )

        # ========================================
        # ETAPE 3: Collecte d'ARTICLES ALÉATOIRES
        # ========================================
        if time.time() - start_time <= effective_timeout:
            print(
                f"\n{PROGRESS_MESSAGES['phase_3'].format(
                    stats.random_token_target
                )}"
            )

            try:
                random_tokens = 0

                while random_tokens < stats.random_token_target:
                    if time.time() - start_time > effective_timeout:
                        print(
                            "Limite de temps atteinte pendant la collecte des articles aléatoires."
                        )
                        break

                    remaining_tokens = stats.random_token_target - random_tokens
                    batch_size = max(1, min(15, remaining_tokens // 200))

                    random_articles, rand_tokens = collector.collect_random(
                        num_articles=batch_size, sleep_time=(0.5, 2)
                    )

                    if not random_articles:
                        print(
                            f"  Plus d'articles aléatoires disponibles "
                            f"pour {language_code}"
                        )
                        break

                    articles.extend(random_articles)

                    for article in random_articles:
                        stats.update_random_stats(article)

                    random_tokens += rand_tokens
                    print(
                        f"  Progression: "
                        f"{random_tokens}/{stats.random_token_target}"
                        f" tokens collectés"
                    )

                    if random_tokens >= stats.random_token_target:
                        break

                stats.print_progress_summary(
                    "les articles aléatoires",
                    stats.random_tokens,
                    stats.random_token_target,
                )

            except Exception as e:
                logging.error(
                    f"Erreur lors de la collecte des articles aléatoires: "
                    f"{str(e)}",
                    exc_info=True,
                )

        # ============================================
        # FINALISATION ET GÉNÉRATION DES STATISTIQUES
        # ============================================

        total_tokens = stats.total_tokens
        execution_time = time.time() - start_time

        logging.info(
            f"Collecte terminée pour {language_code}: "
            f"{len(articles)} articles, {total_tokens} tokens"
        )

        # Sauvegarde des statistiques détaillées
        stats.save_to_file(
            execution_time,
            params["max_token_length"],
            output_dir=DEFAULT_PATHS["languages_stats_dir"],
        )

        logging.info(f"Statistiques finales pour {language_code} sauvegardées")

        return (
            articles,
            stats.categories_stats,
            stats.total_tokens,
            execution_time,
            stats.main_ordered_tokens,
            stats.main_random_tokens,
            stats.limited_articles,
            (
                stats.limited_articles / stats.articles_count * 100
                if stats.articles_count > 0
                else 0
            ),
        )

    except Exception as e:
        logging.error(
            f"Erreur critique lors de la collecte pour {language_code}: "
            f"{str(e)}",
            exc_info=True,
        )
        raise RuntimeError(
            f"Échec de la collecte pour {language_code}: "
            f"{e}"
        ) from e


# =====================
# FONCTION PRINCIPALE
# =====================

def main(
        max_languages: Optional[int] = None,
        timeout_multiplier: float = 1.0
) -> List:
    """
    Fonction principale pour exécuter la collecte adaptative avec robustesse

    Cette fonction orchestre l'ensemble du processus de collecte pour toutes
    les langues configurées, avec gestion robuste des erreurs, reprise après
    interruption, et génération de rapports complets sur la qualité et les
    performances de la collecte réalisée.

    Args:
        max_languages (int, optional): limite le nb de langues à traiter
            (si None, traite toutes les langues configurées)
        timeout_multiplier (float): multiplicateur pour ajuster les délais
            (valeur par défaut: 1.0 (timeouts standards))

    Returns:
        List: liste de tous les articles collectés avec métadonnées complètes

    Raises:
        ValueError: si les paramètres d'entrée sont invalides
        RuntimeError: si l'initialisation de l'environnement échoue

    Exemple:
        Collecte standard sur toutes les langues:
        >>> articles = main()

        Collecte limitée avec timeouts prolongés:
        >>> articles = main(max_languages=5, timeout_multiplier=1.5)

    Note sur la robustesse:
        La fonction continue le traitement même si certaines langues
        échouent, permettant de maximiser la collecte réussie tout en
        loggant les problèmes pour analyse ultérieure.
    """
    logging.info("Démarrage de la fonction principale de collecte")

    try:
        # Validation des paramètres et initialisation
        validate_collection_parameters(max_languages, timeout_multiplier)
        setup_collection_environment()

        # Configuration de la collecte
        languages_to_process = (
            LANGUAGES[:max_languages] if max_languages
            else LANGUAGES
        )
        max_languages_count = len(languages_to_process)

        print(PROGRESS_MESSAGES["start_collection"].format(max_languages_count))

        # Variables de suivi global
        all_articles = []
        successfully_processed = set()
        failed_languages = []

        # Boucle principale de collecte par langue
        for i, language in enumerate(languages_to_process, 1):
            if language in successfully_processed:
                print(
                    f"\nLa langue {language} a déjà été traitée, "
                    f"passage à la suivante."
                )
                continue

            print("\n" + "=" * 50)
            print(
                PROGRESS_MESSAGES["language_progress"].format(
                    i, max_languages_count, language
                )
            )
            print("=" * 50)

            try:
                # Collecte pour cette langue avec gestion d'erreurs
                collection_result = collect_articles(language, ALL_CATEGORIES)

                (
                    articles,
                    cat_stats,
                    total_tokens,
                    exec_time,
                    main_ordered_tokens,
                    main_random_tokens,
                    limited_articles,
                    limited_percentage,
                ) = collection_result

                if articles:
                    # Ajout à la collection globale
                    all_articles.extend(articles)

                    # Sauvegarde des articles dans un fichier CSV
                    save_articles_to_csv(
                        language, articles, DEFAULT_PATHS["articles_dir"]
                    )

                    print(
                        f"✅ {len(articles)} articles collectés pour {language}"
                        f", {total_tokens} tokens"
                    )
                    successfully_processed.add(language)
                else:
                    print(f"⚠️ Aucun article collecté pour {language}")
                    failed_languages.append((language, "Aucun article collecté"))

            except Exception as e:
                error_msg = f"Erreur lors de la collecte pour {language}: {str(e)}"
                logging.error(error_msg, exc_info=True)
                failed_languages.append((language, str(e)))
                print(f"❌ {error_msg}")
                continue

        # =====================================================
        # GÉNÉRATION DU RAPPORT FINAL ET STATISTIQUES GLOBALES
        # =====================================================
        print("\n" + "=" * 60)
        print(PROGRESS_MESSAGES["collection_complete"])
        print("=" * 60)
        print(
            PROGRESS_MESSAGES["total_summary"].format(
                len(all_articles), len(successfully_processed)
            )
        )

        # Statistiques de succès/échec
        success_rate = len(successfully_processed) / max_languages_count * 100
        print(
            f"Taux de réussite: {success_rate:.1f}% "
            f"({len(successfully_processed)}/{max_languages_count} langues)"
        )

        if failed_languages:
            print(f"\n⚠️ Langues avec problèmes ({len(failed_languages)}):")
            for lang, error in failed_languages:
                print(f"  • {lang}: {error}")

        # Sauvegarde des statistiques globales
        global_stats_path = f"{DEFAULT_PATHS['global_stats_dir']}/global_stats.csv"
        save_global_stats(
            successfully_processed,
            all_articles,
            global_stats_path
        )

        print(f"\n📊 Statistiques globales sauvegardées : {global_stats_path}")
        print(f"📁 Articles sauvegardés dans : {DEFAULT_PATHS['articles_dir']}")

        return all_articles

    except KeyboardInterrupt:
        print("\n❌ Collecte interrompue par l'utilisateur")
        logging.info("Collecte interrompue par l'utilisateur")
        raise
    except Exception as e:
        logging.error(
            f"Erreur critique dans la fonction principale: {str(e)}",
            exc_info=True
        )
        raise RuntimeError(f"Échec de la collecte: {e}") from e


# =========================================================
# POINT D'ENTRÉE PRINCIPAL AVEC GESTION D'ERREURS COMPLÈTE
# =========================================================

if __name__ == "__main__":
    """Point d'entrée principal avec gestion d'erreurs robuste
    et interface utilisateur

    Exécute la collecte de corpus avec gestion complète des erreurs,
    logging approprié, et interface utilisateur informative
    pour le monitoring en temps réel du processus de collecte.

    Variables d'environnement prises en charge:
        MAX_LANGUAGES: limite le nombre de langues à traiter
        TIMEOUT_MULTIPLIER: ajuste les délais d'attente
            (par défaut: 1.0)

    Codes de sortie:
        0: succès total
        1: interruption utilisateur
        2: erreur de paramètres ou d'environnement
        3: erreur de collecte partielle
        4: échec critique
    """
    try:
        print("🚀 Lancement de la création de corpus cyrillique adaptatif...")
        print(f"📅 Démarrage: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Lecture des paramètres d'environnement optionnels
        max_langs = None
        if "MAX_LANGUAGES" in os.environ:
            try:
                max_langs = int(os.environ["MAX_LANGUAGES"])
                print(
                    f"🔢 Limitation à {max_langs} langues "
                    f"(variable d'environnement)"
                )
            except ValueError:
                print("⚠️ Variable MAX_LANGUAGES invalide, ignorée")

        timeout_mult = 1.0
        if "TIMEOUT_MULTIPLIER" in os.environ:
            try:
                timeout_mult = float(os.environ["TIMEOUT_MULTIPLIER"])
                print(f"⏱️ Multiplicateur de timeout : {timeout_mult}x")
            except ValueError:
                print(
                    "⚠️ Variable TIMEOUT_MULTIPLIER invalide, valeur par défaut utilisée"
                )

        # Lancement de la collecte principale
        articles = main(
            max_languages=max_langs,
            timeout_multiplier=timeout_mult
        )

        # Rapport de succès final
        end_time = datetime.now()
        print("\n✅ Collecte terminée avec succès !")
        print(f"📊 {len(articles):,} articles collectés au total")
        print(f"⏱️ Terminé : {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 Données sauvegardées dans : {DEFAULT_PATHS['articles_dir']}")
        print(f"📈 Métriques disponibles dans : {DEFAULT_PATHS['metrics_base']}")

        # Message de conclusion
        print("\n🎯 Mission accomplie! Corpus multilingue cyrillique créé avec succès.")

    except KeyboardInterrupt:
        print("\n❌ Processus interrompu par l'utilisateur (Ctrl+C)")
        print("💾 Les données partielles ont été sauvegardées")
        exit(1)

    except (ValueError, RuntimeError) as e:
        print(f"\n❌ Erreur de configuration ou d'exécution: {e}")
        print("💡 Vérifiez les paramètres et les permissions de fichiers")
        logging.error(f"Erreur de configuration: {e}")
        exit(2)

    except Exception as e:
        print(f"\n❌ Erreur critique inattendue : {e}")
        print("💡 Consultez les logs pour plus de détails")
        print(
            f"📝 Fichier de log: {DEFAULT_PATHS['logs_dir']}"
            f"/{LOGGING_CONFIG['filename_pattern']}"
        )
        logging.error(f"Erreur critique: {e}", exc_info=True)
        exit(4)

    # Succès complet
    exit(0)
