"""Gestionnaire de cache pour l'optimisation de la collecte d'articles

Ce module fournit un cache en mémoire simple et efficace pour éviter le
retraitement d'articles déjà évalués. Il maintient des ensembles globaux
partagés entre tous les modules de collecte pour optimiser les performances
et éviter les requêtes API redondantes.

Fonctionnalités principales:
- Cache des articles identifiés comme trop courts
- Cache des articles déjà collectés avec succès
- Fonctions utilitaires pour interroger et maintenir les caches
- Gestion automatique de la taille des caches pour éviter la surconsommation mémoire

Le cache utilise des ensembles (sets) Python pour des performances optimales
en recherche et insertion, avec une complexité O(1) pour toutes les opérations.
"""

import logging
from typing import Set, Union, List, Optional


# === CONSTANTES DE CONFIGURATION ===

# Limites de taille pour éviter la surconsommation mémoire
MAX_CACHE_SIZE = 100000
CACHE_CLEANUP_THRESHOLD = 80000  # nettoyer quand on atteint ce seuil

# Messages de log
LOG_MESSAGES = {
    "CACHE_CLEANUP": "Nettoyage du cache: {removed} IDs supprimés, {remaining} conservés",
    "CACHE_OVERFLOW": "Cache trop volumineux ({size} IDs), nettoyage automatique effectué",
    "INVALID_ID": "ID d'article invalide ignoré: {id}",
}


# === CACHES GLOBAUX ===

# Ensemble des IDs d'articles identifiés comme trop courts
too_short_article_ids: Set[int] = set()

# Ensemble des IDs d'articles déjà collectés avec succès
collected_article_ids: Set[int] = set()


# === FONCTIONS DE GESTION DU CACHE ===


def mark_as_too_short(article_id: Union[int, str]) -> bool:
    """
    Marque un article comme étant trop court pour éviter de le retraiter

    Args:
        article_id: ID de l'article à marquer (int ou str convertible)

    Returns:
        True si l'article a été ajouté au cache, False si déjà présent ou invalide
    """
    try:
        # Convertir en entier si nécessaire
        if isinstance(article_id, str):
            article_id = int(article_id)

        if not isinstance(article_id, int) or article_id <= 0:
            logging.warning(LOG_MESSAGES["INVALID_ID"].format(id=article_id))
            return False

        # Vérifier si déjà présent
        if article_id in too_short_article_ids:
            return False

        # Ajouter au cache
        too_short_article_ids.add(article_id)

        # Vérifier la taille du cache et nettoyer si nécessaire
        _cleanup_cache_if_needed()

        logging.debug(f"Article {article_id} marqué comme trop court")
        return True

    except (ValueError, TypeError):
        logging.warning(LOG_MESSAGES["INVALID_ID"].format(id=article_id))
        return False


def is_too_short(article_id: Union[int, str]) -> bool:
    """
    Vérifie si un article est déjà marqué comme trop court

    Args:
        article_id: ID de l'article à vérifier (int ou str convertible)

    Returns:
        True si l'article est marqué comme trop court, False sinon
    """
    try:
        # Convertir en entier si nécessaire
        if isinstance(article_id, str):
            article_id = int(article_id)

        if not isinstance(article_id, int) or article_id <= 0:
            return False

        return article_id in too_short_article_ids

    except (ValueError, TypeError):
        return False


def mark_as_collected(article_id: Union[int, str]) -> bool:
    """
    Marque un article comme ayant été collecté avec succès

    Args:
        article_id: ID de l'article collecté

    Returns:
        True si l'article a été ajouté au cache,
        False si déjà présent ou invalide
    """
    try:
        # Convertir en entier si nécessaire
        if isinstance(article_id, str):
            article_id = int(article_id)

        if not isinstance(article_id, int) or article_id <= 0:
            logging.warning(LOG_MESSAGES["INVALID_ID"].format(id=article_id))
            return False

        # Vérifier si déjà présent
        if article_id in collected_article_ids:
            return False

        # Ajouter au cache
        collected_article_ids.add(article_id)

        # Vérifier la taille du cache et nettoyer si nécessaire
        _cleanup_cache_if_needed()

        logging.debug(f"Article {article_id} marqué comme collecté")
        return True

    except (ValueError, TypeError):
        logging.warning(LOG_MESSAGES["INVALID_ID"].format(id=article_id))
        return False


def is_collected(article_id: Union[int, str]) -> bool:
    """
    Vérifie si un article a déjà été collecté

    Args:
        article_id: ID de l'article à vérifier

    Returns:
        True si l'article a été collecté, False sinon
    """
    try:
        # Convertir en entier si nécessaire
        if isinstance(article_id, str):
            article_id = int(article_id)

        if not isinstance(article_id, int) or article_id <= 0:
            return False

        return article_id in collected_article_ids

    except (ValueError, TypeError):
        return False


def add_multiple_too_short(article_ids: List[Union[int, str]]) -> int:
    """
    Ajoute plusieurs articles au cache des articles trop courts

    Args:
        article_ids: Liste des IDs d'articles à ajouter

    Returns:
        nombre d'articles ajoutés avec succès
    """
    if not isinstance(article_ids, list):
        logging.warning("article_ids doit être une liste")
        return 0

    added_count = 0
    for article_id in article_ids:
        if mark_as_too_short(article_id):
            added_count += 1

    logging.debug(
        f"{added_count}/{len(article_ids)} articles ajoutés au cache des trop courts"
    )
    return added_count


def add_multiple_collected(article_ids: List[Union[int, str]]) -> int:
    """
    Ajoute plusieurs articles au cache des articles collectés

    Args:
        article_ids: liste des IDs d'articles à ajouter

    Returns:
        nombre d'articles ajoutés avec succès
    """
    if not isinstance(article_ids, list):
        logging.warning("article_ids doit être une liste")
        return 0

    added_count = 0
    for article_id in article_ids:
        if mark_as_collected(article_id):
            added_count += 1

    logging.debug(
        f"{added_count}/{len(article_ids)} articles ajoutés au cache des collectés"
    )
    return added_count


def get_cache_stats() -> dict:
    """
    Retourne des statistiques sur l'état actuel des caches.

    Returns:
        dictionnaire avec les statistiques des caches
    """
    return {
        "too_short_count": len(too_short_article_ids),
        "collected_count": len(collected_article_ids),
        "total_cached_ids": len(too_short_article_ids) + len(collected_article_ids),
        "max_cache_size": MAX_CACHE_SIZE,
        "cleanup_threshold": CACHE_CLEANUP_THRESHOLD,
    }


def clear_cache(cache_type: Optional[str] = None) -> dict:
    """
    Vide un ou tous les caches

    Args:
        cache_type: type de cache à vider
            ('too_short', 'collected', ou None pour tous)

    Returns:
        statistiques avant et après le nettoyage
    """
    stats_before = get_cache_stats()

    if cache_type is None or cache_type == "too_short":
        too_short_cleared = len(too_short_article_ids)
        too_short_article_ids.clear()
        logging.info(f"Cache 'too_short' vidé: {too_short_cleared} IDs supprimés")
    else:
        too_short_cleared = 0

    if cache_type is None or cache_type == "collected":
        collected_cleared = len(collected_article_ids)
        collected_article_ids.clear()
        logging.info(f"Cache 'collected' vidé: {collected_cleared} IDs supprimés")
    else:
        collected_cleared = 0

    stats_after = get_cache_stats()

    return {
        "before": stats_before,
        "after": stats_after,
        "too_short_cleared": too_short_cleared,
        "collected_cleared": collected_cleared,
    }


def _cleanup_cache_if_needed() -> None:
    """
    Nettoie automatiquement les caches s'ils deviennent trop volumineux

    Cette fonction est appelée automatiquement lors des ajouts pour maintenir
    les performances et éviter une consommation mémoire excessive.
    """
    total_size = len(too_short_article_ids) + len(collected_article_ids)

    if total_size > CACHE_CLEANUP_THRESHOLD:
        # Nettoyer le cache le plus volumineux
        # en gardant les éléments les plus récents
        if len(too_short_article_ids) > len(collected_article_ids):
            _cleanup_single_cache(too_short_article_ids, "too_short")
        else:
            _cleanup_single_cache(collected_article_ids, "collected")


def _cleanup_single_cache(
    cache: Set[int],
    cache_name: str,
    keep_ratio: float = 0.6
) -> None:
    """
    Nettoie un cache individuel en gardant un pourcentage des éléments

    Args:
        cache: ensemble à nettoyer
        cache_name: nom du cache pour le logging
        keep_ratio: proportion d'éléments à conserver (0.0 à 1.0)
    """
    if not cache:
        return

    original_size = len(cache)
    target_size = int(original_size * keep_ratio)

    # Convertir en liste pour pouvoir faire du slicing
    cache_list = list(cache)

    # Garder les IDs les plus élevés (supposés plus récents)
    cache_list.sort(reverse=True)
    cache.clear()
    cache.update(cache_list[:target_size])

    removed = original_size - len(cache)

    logging.info(
        f"Cache '{cache_name}' nettoyé: "
        f"{removed} IDs supprimés, {len(cache)} conservés"
    )

    if original_size > MAX_CACHE_SIZE:
        logging.warning(LOG_MESSAGES["CACHE_OVERFLOW"].format(size=original_size))


def print_cache_summary() -> None:
    """
    Affiche un résumé des caches dans la console
    """
    stats = get_cache_stats()

    print(f"Cache des articles trop courts: {stats['too_short_count']:,} IDs")
    print(f"Cache des articles collectés: {stats['collected_count']:,} IDs")
    print(f"Total en cache: {stats['total_cached_ids']:,} IDs")

    # Alerte si proche de la limite
    if stats["total_cached_ids"] > CACHE_CLEANUP_THRESHOLD:
        print(f"⚠️  Cache proche de la limite ({CACHE_CLEANUP_THRESHOLD:,} IDs)")
