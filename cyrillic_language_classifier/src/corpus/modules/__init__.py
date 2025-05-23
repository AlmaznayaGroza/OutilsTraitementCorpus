"""Package de collecte de corpus Wikipédia multilingue

Ce package fournit un ensemble complet d'outils pour la collecte, le traitement
et la gestion de corpus d'articles Wikipedia dans plusieurs langues.
Il est conçu pour une collecte adaptative
qui s'ajuste automatiquement selon les ressources
disponibles pour chaque langue.

Modules principaux:
- config: configuration et paramètres adaptatifs par langue
- api_utils: utilitaires pour l'interaction avec l'API Wikipedia
- text_processing: traitement et validation des textes
- article_collector: collecteur principal avec stratégies adaptatives
- data_manager: gestion des données et fusion de corpus
- stat_manager: statistiques et métriques de performance
- cache_manager: cache pour optimiser les performances

Classes principales:
- ArticleCollector: collecteur d'articles avec stratégies adaptatives
- CollectionStats: gestionnaire de statistiques de collecte

Fonctions utilitaires principales :
- get_adaptive_params: récupère les paramètres adaptatifs pour une langue
- save_articles_to_csv: sauvegarde des articles au format CSV
- calculate_token_targets: calcule la distribution d'objectifs de tokens

Example d'utilisation de base :
    >>> from corpus import ArticleCollector, get_adaptive_params
    >>> params = get_adaptive_params('ru')
    >>> collector = ArticleCollector('ru', ['Culture', 'History'], params)
    >>> articles, tokens, _, _ = collector.collect_by_category('Culture', 10000)
    >>> print(f"Collecté: {len(articles)} articles, {tokens} tokens")

Example de collecte complète:
    >>> from corpus import (
    ...      ArticleCollector, CollectionStats,
    ...      save_articles_to_csv, get_adaptive_params
    ... )
    >>> 
    >>> # Configuration
    >>> language = 'ru'
    >>> categories = ['Culture', 'History']
    >>> token_target = 100000
    >>> 
    >>> # Initialisation
    >>> params = get_adaptive_params(language)
    >>> collector = ArticleCollector(language, categories, params)
    >>> stats = CollectionStats(language, token_target, categories)
    >>> 
    >>> # Collecte
    >>> all_articles = []
    >>> for category in categories:
    ...     articles, tokens, _, _ = collector.collect_by_category(category, 20000)
    ...     all_articles.extend(articles)
    ...     for article in articles:
    ...         stats.update_main_category_stats(category, article, 'ordonné')
    >>> 
    >>> # Sauvegarde et statistiques
    >>> csv_path = save_articles_to_csv(language, all_articles, 'output/')
    >>> stats.finalize_collection()
    >>> stats_path = stats.save_to_file(stats.get_execution_time(), 1000)
    >>> print(f"Corpus sauvé: {csv_path}, Stats: {stats_path}")
"""

# Import des fonctions de configuration
from .config import (
    TIME_LIMIT,
    LANGUAGES,
    ALL_CATEGORIES,
    MAX_DEPTHS_BY_GROUP,
    CATEGORY_TRANSLATIONS,
    get_adaptive_params,
    get_language_group,
    get_target_for_language,
    validate_language_code,
    get_available_categories_for_language,
    get_category_translation
)

# Import du collecteur principal
from .article_collector import ArticleCollector

# Import des fonctions de traitement de texte
from .text_processing import (
    process_text,
    validate_article,
    select_valid_articles,
    process_article,
    estimate_token_count,
    calculate_processing_stats
)

# Import des fonctions de gestion de données
from .data_manager import (
    save_articles_to_csv,
    merge_with_existing_data,
    load_csv_with_fallback_encoding,
    remove_duplicates,
    validate_article_data,
    get_corpus_statistics,
    validate_corpus_integrity,
    export_corpus_summary
)

# Import des gestionnaires de statistiques
from .stat_manager import (
    CollectionStats,
    calculate_token_targets,
    print_collection_plan,
    save_global_stats,
    compare_language_performance,
    generate_performance_report
)

# Import des fonctions de cache
from .cache_manager import (
    mark_as_too_short,
    is_too_short,
    mark_as_collected,
    is_collected,
    get_cache_stats,
    clear_cache,
    print_cache_summary
)

# Import des utilitaires API
from .api_utils import (
    fetch_subcategories,
    fetch_category_articles,
    fetch_article_content,
    fetch_random_article,
    batch_fetch_articles_content,
    get_category_info
)

# Liste des langues supportées (réexportée pour facilité d'accès)
SUPPORTED_LANGUAGES = LANGUAGES

# Catégories disponibles (réexportées pour facilité d'accès)
AVAILABLE_CATEGORIES = ALL_CATEGORIES


# === FONCTIONS DE CONVENANCE ===

def get_package_info() -> dict:
    """
    Retourne les informations sur le package.
    
    Returns:
        dictionnaire avec les métadonnées du package
    """
    return {
        "name": "corpus",
        "supported_languages": len(SUPPORTED_LANGUAGES),
        "available_categories": len(AVAILABLE_CATEGORIES)
    }


def create_collector_with_stats(
        language_code: str,
        categories: list,
        token_target: int
) -> tuple:
    """
    Fonction de convenance pour créer un collecteur
    et son gestionnaire de stats.
    
    Args:
        language_code: code de la langue
        categories: liste des catégories
        token_target: objectif de tokens
        
    Returns:
        tuple (collector, stats) prêts à utiliser
    """
    params = get_adaptive_params(language_code)
    collector = ArticleCollector(language_code, categories, params)
    stats = CollectionStats(language_code, token_target, categories)
    
    return collector, stats


def quick_collect(
        language_code: str,
        categories: list,
        tokens_per_category: int = 10000
) -> dict:
    """
    Fonction de collecte rapide pour tests et prototypage
    
    Args:
        language_code: code de la langue
        categories: liste des catégories
        tokens_per_category: tokens à collecter par catégorie
        
    Returns:
        dictionnaire avec les résultats de collecte
    """
    # Validation des paramètres
    if not validate_language_code(language_code):
        raise ValueError(f"Langue non supportée: {language_code}")
    
    token_target = len(categories) * tokens_per_category
    collector, stats = create_collector_with_stats(
        language_code, categories, token_target
    )
    
    # Calculer les objectifs
    params = get_adaptive_params(language_code)
    targets = calculate_token_targets(token_target, params, len(categories))
    stats.set_token_targets(*targets)
    
    # Collecter les articles
    all_articles = []
    total_tokens = 0
    
    for category in categories:
        try:
            articles, tokens, _, _ = collector.collect_by_category(
                category, tokens_per_category)
            
            all_articles.extend(articles)
            total_tokens += tokens
            
            # Mettre à jour les stats
            for article in articles:
                article_type = article.get('type', 'ordonné')
                stats.update_main_category_stats(
                    category, article, article_type
                )
            
        except Exception as e:
            print(f"Erreur lors de la collecte pour {category}: {e}")
            continue
    
    # Finaliser
    stats.finalize_collection()
    
    return {
        "language": language_code,
        "articles": all_articles,
        "total_articles": len(all_articles),
        "total_tokens": total_tokens,
        "target_tokens": token_target,
        "completion_percentage": stats.get_completion_percentage(),
        "execution_time": stats.get_execution_time(),
        "stats": stats,
        "collector": collector
    }


def validate_setup() -> dict:
    """
    Valide que le package est correctement configuré
    
    Returns:
        dictionnaire avec les résultats de validation
    """
    validation_results = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "info": {}
    }
    
    # Vérifier les langues
    try:
        validation_results["info"]["languages_count"] = len(SUPPORTED_LANGUAGES)
        validation_results["info"]["sample_languages"] = SUPPORTED_LANGUAGES[:5]
    except Exception as e:
        validation_results["errors"].append(
            f"Erreur de configuration des langues: {e}"
        )
        validation_results["is_valid"] = False
    
    # Vérifier les catégories
    try:
        validation_results["info"]["categories_count"] = len(AVAILABLE_CATEGORIES)
        validation_results["info"]["sample_categories"] = AVAILABLE_CATEGORIES[:3]
    except Exception as e:
        validation_results["errors"].append(
            f"Erreur de configuration des catégories: {e}"
        )
        validation_results["is_valid"] = False
    
    # Vérifier les traductions
    try:
        sample_lang = 'ru' if 'ru' in SUPPORTED_LANGUAGES else SUPPORTED_LANGUAGES[0]
        sample_category = AVAILABLE_CATEGORIES[0] if AVAILABLE_CATEGORIES else 'Culture'
        translation = get_category_translation(sample_category, sample_lang)
        validation_results["info"]["translations_working"] = translation is not None
    except Exception as e:
        validation_results["warnings"].append(f"Problème de traductions: {e}")
    
    # Vérifier le cache
    try:
        cache_stats = get_cache_stats()
        validation_results["info"]["cache_initialized"] = True
        validation_results["info"]["cache_stats"] = cache_stats
    except Exception as e:
        validation_results["warnings"].append(f"Problème de cache: {e}")
    
    return validation_results


# === EXPORTS PRINCIPAUX ===

# Classes principales
__all__ = [
    # Classes
    'ArticleCollector',
    'CollectionStats',
    
    # Configuration
    'get_adaptive_params',
    'get_language_group', 
    'get_target_for_language',
    'validate_language_code',
    
    # Traitement
    'process_text',
    'validate_article',
    'select_valid_articles',
    
    # Gestion de données
    'save_articles_to_csv',
    'merge_with_existing_data',
    'validate_article_data',
    'get_corpus_statistics',
    
    # Statistiques
    'calculate_token_targets',
    'print_collection_plan',
    'save_global_stats',
    
    # Cache
    'mark_as_too_short',
    'is_too_short',
    'get_cache_stats',
    'clear_cache',
    
    # API
    'fetch_category_articles',
    'fetch_article_content',
    'fetch_random_article',
    
    # Constantes
    'SUPPORTED_LANGUAGES',
    'AVAILABLE_CATEGORIES',
    'TIME_LIMIT',
    
    # Fonctions de convenance
    'get_package_info',
    'create_collector_with_stats',
    'quick_collect',
    'validate_setup'
]


# Message d'initialisation (optionnel, pour debug)
import logging
logging.getLogger(__name__).debug(
    f"Package corpus initialisé - "
    f"{len(SUPPORTED_LANGUAGES)} langues supportées"
)