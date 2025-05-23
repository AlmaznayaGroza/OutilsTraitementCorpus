"""Utilitaires pour l'interaction avec l'API Wikipédia

Ce module fournit des fonctions standardisées pour interagir avec l'API MediaWiki
de Wikipédia. Il encapsule la logique répétitive de récupération de données
depuis les différentes versions linguistiques de Wikipédia.

Fonctionnalités principales :
- Récupération de sous-catégories avec exploration hiérarchique
- Collecte d'articles par catégorie avec différents critères de tri
- Extraction du contenu textuel des articles
- Sélection d'articles aléatoires
- Gestion automatique des délais et des erreurs API

Toutes les fonctions respectent les limitations de débit de l'API de Wikipédia
et incluent une gestion d'erreurs robuste pour assurer la stabilité
de la collecte de corpus.
"""

import requests
import time
import random
import logging
from typing import List, Dict, Optional, Tuple, Any, Union


# === CONSTANTES DE CONFIGURATION ===

# Paramètres par défaut pour les requêtes API
DEFAULT_API_LIMIT = 50
DEFAULT_SLEEP_RANGE = (0.5, 1.5)
DEFAULT_TIMEOUT = 30

# Paramètres de l'API MediaWiki
API_FORMAT = "json"
MAIN_NAMESPACE = "0"  # espace de noms principal (articles)

# Méthodes de tri disponibles
VALID_SORT_METHODS = ["sortkey", "timestamp"]
VALID_SORT_DIRECTIONS = ["asc", "desc"]

# Types de membres de catégorie
CATEGORY_MEMBER_TYPES = {
    "SUBCATEGORIES": "subcat",
    "PAGES": "page",
    "FILES": "file"
}


def validate_api_params(api_url: str, category_title: str, limit: int) -> None:
    """
    Valide les paramètres communs des requêtes API.

    Args:
        api_url: URL de l'API Wikipedia
        category_title: titre de la catégorie
        limit: limite du nombre de résultats

    Raises:
        ValueError: si les paramètres sont invalides
    """
    if not api_url or not isinstance(api_url, str):
        raise ValueError("L'URL de l'API doit être une chaîne non vide")

    if not api_url.startswith(("http://", "https://")):
        raise ValueError("L'URL de l'API doit commencer par http:// ou https://")

    if not category_title or not isinstance(category_title, str):
        raise ValueError("Le titre de la catégorie doit être une chaîne non vide")

    if not isinstance(limit, int) or limit <= 0:
        raise ValueError("La limite doit être un entier positif")

    if limit > 500:  # limite max de l'API MediaWiki
        raise ValueError(
            "La limite ne peut pas dépasser 500 (limitation API MediaWiki)"
        )


def execute_api_request(
    api_url: str,
    params: Dict[str, str],
    sleep_time: Tuple[float, float] = DEFAULT_SLEEP_RANGE,
    timeout: int = DEFAULT_TIMEOUT,
) -> Optional[Dict[str, Any]]:
    """
    Exécute une requête vers l'API Wikipedia avec gestion d'erreurs.

    Args:
        api_url: URL de l'API Wikipedia
        params: paramètres de la requête
        sleep_time: tuple (min, max) pour le délai aléatoire
        timeout: timeout de la requête en secondes

    Returns:
        réponse JSON de l'API ou None en cas d'erreur

    Raises:
        ValueError: si les paramètres sont invalides
    """
    if not isinstance(params, dict):
        raise ValueError("Les paramètres doivent être un dictionnaire")

    if len(sleep_time) != 2 or sleep_time[0] < 0 or sleep_time[1] < sleep_time[0]:
        raise ValueError(
            "sleep_time doit être un tuple (min, max)"
            " avec min >= 0 et max >= min"
        )

    # Respecter les délais pour éviter de surcharger l'API
    time.sleep(random.uniform(*sleep_time))

    try:
        logging.debug(f"Requête API vers {api_url} avec paramètres: {params}")
        response = requests.get(api_url, params=params, timeout=timeout)
        response.raise_for_status()  # lève une exception pour les codes d'erreur HTTP

        json_response = response.json()
        logging.debug(
            f"Requête API réussie, {len(str(json_response))}"
            f" caractères reçus"
        )

        return json_response

    except requests.exceptions.Timeout:
        logging.error(f"Timeout lors de la requête API vers {api_url}")
        return None
    except requests.exceptions.ConnectionError:
        logging.error(f"Erreur de connexion lors de la requête API vers {api_url}")
        return None
    except requests.exceptions.HTTPError as e:
        logging.error(f"Erreur HTTP lors de la requête API: {e}")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Erreur de requête API: {e}")
        return None
    except ValueError as e:
        logging.error(f"Erreur de parsing JSON: {e}")
        return None


def fetch_subcategories(
    api_url: str,
    category_title: str,
    limit: int = DEFAULT_API_LIMIT,
    sleep_time: Tuple[float, float] = DEFAULT_SLEEP_RANGE,
) -> List[Dict[str, Any]]:
    """
    Récupère la liste des sous-catégories d'une catégorie donnée.

    Cette fonction encapsule la logique répétitive pour récupérer
    les sous-catégories via l'API MediaWiki.
    Elle gère automatiquement les erreurs et les délais.

    Args:
        api_url: URL de l'API Wikipedia
            (ex: "https://ru.wikipedia.org/w/api.php" pour le russe)
        category_title: titre complet de la catégorie
            (ex: "Категория:Культура")
        limit: nb maximum de sous-catégories à récupérer (max 500)
        sleep_time: tuple (min, max) pour le délai aléatoire entre requêtes

    Returns:
        liste des sous-catégories trouvées, ou liste vide en cas d'erreur

    Raises:
        ValueError: si les paramètres sont invalides
    """
    try:
        validate_api_params(api_url, category_title, limit)
    except ValueError as e:
        logging.error(f"Paramètres invalides pour fetch_subcategories: {e}")
        return []

    params = {
        "action": "query",
        "format": API_FORMAT,
        "list": "categorymembers",
        "cmtitle": category_title,
        "cmlimit": str(limit),
        "cmtype": CATEGORY_MEMBER_TYPES["SUBCATEGORIES"],
    }

    response = execute_api_request(api_url, params, sleep_time)

    if (
        response and "query" in response
        and "categorymembers" in response["query"]
    ):
        subcategories = response["query"]["categorymembers"]
        logging.info(
            f"{len(subcategories)} sous-catégories trouvées"
            f" pour {category_title}"
        )
        return subcategories
    else:
        # Cas où la réponse est valide mais ne contient pas de sous-catégories
        logging.warning(f"Aucune sous-catégorie trouvée pour {category_title}")
        return []


def fetch_category_articles(
    api_url: str,
    category_title: str,
    limit: int = DEFAULT_API_LIMIT,
    sort_method: str = "sortkey",
    sort_direction: str = "asc",
    sleep_time: Tuple[float, float] = DEFAULT_SLEEP_RANGE,
) -> List[Dict[str, Any]]:
    """
    Récupère la liste des articles d'une catégorie donnée.

    Cette fonction standardise la récupération d'articles d'une catégorie
    avec différentes options de tri et de limite.

    Args:
        api_url: URL de l'API Wikipedia
        category_title: titre complet de la catégorie
        limit: nb max d'articles à récupérer (max 500)
        sort_method: méthode de tri ("sortkey", "timestamp")
        sort_direction: direction du tri ("asc", "desc")
        sleep_time: tuple (min, max) pour le délai aléatoire

    Returns:
        liste des articles trouvés, ou liste vide en cas d'erreur

    Raises:
        ValueError: si les paramètres sont invalides
    """
    try:
        validate_api_params(api_url, category_title, limit)
    except ValueError as e:
        logging.error(f"Paramètres invalides pour fetch_category_articles: {e}")
        return []

    # Validation des paramètres de tri
    if sort_method not in VALID_SORT_METHODS:
        raise ValueError(
            f"Méthode de tri invalide: {sort_method}. "
            f"Valeurs acceptées: {VALID_SORT_METHODS}"
        )

    if sort_direction not in VALID_SORT_DIRECTIONS:
        raise ValueError(
            f"Direction de tri invalide: {sort_direction}. "
            f"Valeurs acceptées: {VALID_SORT_DIRECTIONS}"
        )

    params = {
        "action": "query",
        "format": API_FORMAT,
        "list": "categorymembers",
        "cmtitle": category_title,
        "cmlimit": str(limit),
        "cmtype": CATEGORY_MEMBER_TYPES["PAGES"],
        "cmsort": sort_method,
        "cmdir": sort_direction,
    }

    response = execute_api_request(api_url, params, sleep_time)

    if (
        response and "query" in response
        and "categorymembers" in response["query"]
    ):
        articles = response["query"]["categorymembers"]
        logging.info(
            f"{len(articles)} articles trouvés pour {category_title} "
            f"(tri: {sort_method} {sort_direction})"
        )
        return articles
    else:
        logging.warning(f"Aucun article trouvé pour {category_title}")
        return []


def fetch_article_content(
    api_url: str,
    page_id: Union[int, str],
    sleep_time: Tuple[float, float] = DEFAULT_SLEEP_RANGE,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Récupère le contenu textuel d'un article Wikipédia.

    Cette fonction encapsule la logique pour récupérer le texte
    d'un article spécifique par son ID.

    Args:
        api_url: URL de l'API Wikipedia
        page_id: ID de la page Wikipedia (int ou str)
        sleep_time: tuple (min, max) pour le délai aléatoire

    Returns:
        tuple (texte_de_l_article, titre) si succès, (None, None) sinon

    Raises:
        ValueError: si les paramètres sont invalides
    """
    if not api_url or not isinstance(api_url, str):
        raise ValueError("L'URL de l'API doit être une chaîne non vide")

    # Convertir page_id en string et valider
    try:
        page_id_str = str(page_id)
        if not page_id_str or page_id_str == "0":
            raise ValueError(
                "L'ID de page doit être un identifiant valide (non vide et non zéro)"
            )
    except (TypeError, ValueError):
        raise ValueError("L'ID de page doit être convertible en chaîne")

    params = {
        "action": "query",
        "format": API_FORMAT,
        "prop": "extracts",
        "explaintext": "1",
        "pageids": page_id_str,
    }

    response = execute_api_request(api_url, params, sleep_time)

    if response and "query" in response and "pages" in response["query"]:
        page_data = response["query"]["pages"].get(page_id_str)

        if page_data and "extract" in page_data:
            # Récupérer aussi le titre si disponible
            title = page_data.get("title", f"Article_{page_id_str}")
            content = page_data["extract"]

            if content:  # Vérifier que le contenu n'est pas vide
                # Détecter les redirections explicites
                if content.strip().upper().startswith("#REDIRECT"):
                    logging.info(f"Article {page_id_str} est une redirection, ignoré")
                    return None, None

                logging.debug(
                    f"Contenu récupéré pour l'article {page_id_str}: "
                    f"{len(content)} caractères"
                )
                return content, title
            else:
                logging.warning(f"Contenu vide pour l'article {page_id_str}")
                return None, None
        else:
            logging.warning(f"Pas d'extrait disponible pour l'article {page_id_str}")
            return None, None
    else:
        logging.error(
            f"Erreur lors de la récupération du contenu pour l'article {page_id_str}"
        )
        return None, None


def fetch_random_article(
    api_url: str, sleep_time: Tuple[float, float] = DEFAULT_SLEEP_RANGE
) -> Optional[Dict[str, Any]]:
    """
    Récupère un article aléatoire depuis Wikipédia.

    Cette fonction encapsule la logique pour obtenir un article aléatoire
    dans l'espace de noms principal (articles).

    Args:
        api_url: URL de l'API Wikipedia
        sleep_time: tuple (min, max) pour le délai aléatoire

    Returns:
        dictionnaire avec 'pageid' et 'title' si succès, None sinon

    Raises:
        ValueError: si l'URL de l'API est invalide
    """
    if not api_url or not isinstance(api_url, str):
        raise ValueError("L'URL de l'API doit être une chaîne non vide")

    if not api_url.startswith(("http://", "https://")):
        raise ValueError("L'URL de l'API doit commencer par http:// ou https://")

    params = {
        "action": "query",
        "format": API_FORMAT,
        "list": "random",
        "rnnamespace": MAIN_NAMESPACE,  # articles principaux uniquement
        "rnlimit": "1",
    }

    response = execute_api_request(api_url, params, sleep_time)

    if response and "query" in response and "random" in response["query"]:
        random_pages = response["query"]["random"]

        if random_pages and len(random_pages) > 0:
            page = random_pages[0]

            # Vérifier que les champs requis sont présents
            if "id" in page and "title" in page:
                result = {"pageid": page["id"], "title": page["title"]}
                logging.debug(
                    f"Article aléatoire récupéré: {result['title']} "
                    f"(ID: {result['pageid']})"
                )
                return result
            else:
                logging.warning("Article aléatoire incomplet (champs manquants)")
                return None
        else:
            logging.warning("Aucun article aléatoire retourné par l'API")
            return None
    else:
        logging.error("Erreur lors de la récupération d'un article aléatoire")
        return None


def batch_fetch_articles_content(
    api_url: str,
    page_ids: List[Union[int, str]],
    sleep_time: Tuple[float, float] = DEFAULT_SLEEP_RANGE,
    batch_size: int = 20,
) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    """
    Récupère le contenu de plusieurs articles en lot
    pour optimiser les performances.

    Cette fonction permet de récupérer plusieurs articles
    en une seule requête API, ce qui est plus efficace
    que des requêtes individuelles.

    Args:
        api_url: URL de l'API Wikipedia
        page_ids: liste des IDs des pages à récupérer
        sleep_time: tuple (min, max) pour le délai aléatoire
        batch_size: nb d'articles par requête (max 50 recommandé)

    Returns:
        dictionnaire {page_id: (content, title)} pour chaque article

    Raises:
        ValueError: si les paramètres sont invalides
    """
    if not api_url or not isinstance(api_url, str):
        raise ValueError("L'URL de l'API doit être une chaîne non vide")

    if not isinstance(page_ids, list) or not page_ids:
        raise ValueError("page_ids doit être une liste non vide")

    if not isinstance(batch_size, int) or batch_size <= 0 or batch_size > 50:
        raise ValueError("batch_size doit être un entier entre 1 et 50")

    results = {}

    # Traiter les IDs par lots
    for i in range(0, len(page_ids), batch_size):
        batch_ids = page_ids[i:i+batch_size]

        # Convertir tous les IDs en chaînes et les joindre
        try:
            ids_str = "|".join(str(page_id) for page_id in batch_ids)
        except (TypeError, ValueError) as e:
            logging.error(f"Erreur lors de la conversion des IDs du lot {i}: {e}")
            continue

        params = {
            "action": "query",
            "format": API_FORMAT,
            "prop": "extracts",
            "explaintext": "1",
            "pageids": ids_str,
        }

        response = execute_api_request(api_url, params, sleep_time)

        if response and "query" in response and "pages" in response["query"]:
            pages = response["query"]["pages"]

            for page_id_str, page_data in pages.items():
                if "extract" in page_data and page_data["extract"]:
                    title = page_data.get("title", f"Article_{page_id_str}")
                    content = page_data["extract"]
                    results[page_id_str] = (content, title)
                else:
                    results[page_id_str] = (None, None)

        logging.info(f"Lot {i//batch_size + 1}: " f"{len(batch_ids)} articles traités")

    logging.info(
        f"Récupération en lot terminée: {len(results)} " f"articles traités au total"
    )
    return results


def get_category_info(
    api_url: str,
    category_title: str,
    sleep_time: Tuple[float, float] = DEFAULT_SLEEP_RANGE,
) -> Optional[Dict[str, Any]]:
    """
    Récupère les informations détaillées d'une catégorie.

    Args:
        api_url: URL de l'API Wikipedia
        category_title: titre complet de la catégorie
        sleep_time: tuple (min, max) pour le délai aléatoire

    Returns:
        dictionnaire avec les informations de la catégorie ou None
    """
    try:
        validate_api_params(
            api_url, category_title, 1
        )  # limite arbitraire pour la validation
    except ValueError as e:
        logging.error(f"Paramètres invalides pour get_category_info: {e}")
        return None

    params = {
        "action": "query",
        "format": API_FORMAT,
        "prop": "categoryinfo",
        "titles": category_title,
    }

    response = execute_api_request(api_url, params, sleep_time)

    if response and "query" in response and "pages" in response["query"]:
        pages = response["query"]["pages"]

        for page_data in pages.values():
            if "categoryinfo" in page_data:
                category_info = page_data["categoryinfo"]
                logging.debug(
                    f"Informations récupérées pour {category_title}:"
                    f" {category_info}"
                )
                return category_info

    logging.warning(f"Aucune information trouvée pour la catégorie {category_title}")
    return None
