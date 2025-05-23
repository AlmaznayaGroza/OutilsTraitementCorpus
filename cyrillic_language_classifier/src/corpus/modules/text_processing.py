"""Traitement et validation de textes pour la collecte de corpus Wikipédia

Ce module fournit des fonctions spécialisées pour le traitement et la validation
d'articles Wikipédia dans le cadre de la collecte de corpus multilingues.
Il encapsule la logique de validation, de sélection et de traitement textuel
des articles récupérés via l'API Wikipedia.

Fonctionnalités principales:
- Validation d'articles selon des critères de longueur et de qualité
- Traitement adaptatif de textes avec limitation de tokens
- Sélection intelligente d'articles valides depuis des listes de candidats
- Gestion des caches pour optimiser les performances
- Support pour différents types d'articles (ordonnés, aléatoires)

Le module respecte les contraintes de collecte adaptative et optimise
les performances en évitant le retraitement d'articles déjà évalués.
"""

import random
import logging
from typing import List, Dict, Optional, Tuple, Set, Any

# Imports depuis les modules locaux
from .api_utils import fetch_article_content
from .cache_manager import mark_as_too_short, is_too_short


# === CONSTANTES DE TRAITEMENT ===

# Types d'articles traités
ARTICLE_TYPES = {
    "ORDERED": "ordonné",
    "RANDOM": "aléatoire",
    "GENERIC": "article"
}

# Limites de validation
MIN_TITLE_LENGTH = 1
MAX_TITLE_LENGTH = 500
MIN_CONTENT_LENGTH = 50  # seuil absolu minimal

# Métriques de tokenisation
AVERAGE_CHARS_PER_TOKEN = 5  # estimation moyenne pour les langues prises en charge
TOKEN_ESTIMATION_BUFFER = 1.1  # marge d'erreur de 10% pour l'estimation

# Messages de log standardisés
LOG_MESSAGES = {
    "ARTICLE_VALID": "Article {type} valide trouvé: {title} ({chars} caractères)",
    "ARTICLE_TOO_SHORT": "Article {type} ignoré (trop court): {title}",
    "CONTENT_NOT_AVAILABLE": "Article {type} ignoré (contenu non disponible): {title}",
    "ARTICLE_INVALID": "Article {type} ignoré (pas valide): {title}",
    "ALREADY_PROCESSED": "Articles déjà traités ignorés: {count}",
    "CACHE_FILTERED": "Articles trop courts déjà identifiés ignorés: {count}",
}


def estimate_token_count(text: str) -> int:
    """
    Estime le nombre de tokens dans un texte.

    Utilise une heuristique simple basée sur la division
    par espaces avec un facteur de correction selon la langue.

    Args:
        text: texte à analyser

    Returns:
        estimation du nombre de tokens

    Raises:
        ValueError: si le texte est None
    """
    if text is None:
        raise ValueError("Le texte ne peut pas être None")

    if not text.strip():
        return 0

    # Tokenisation simple par espaces
    return len(text.split())


def validate_article_data(page_data: Dict[str, Any]) -> Tuple[int, str]:
    """
    Valide et extrait les données essentielles d'un article candidat.

    Args:
        page_data: dictionnaire contenant les données de l'article

    Returns:
        tuple (page_id, title) si valide

    Raises:
        ValueError: si les données sont invalides ou manquantes
    """
    if not isinstance(page_data, dict):
        raise ValueError("page_data doit être un dictionnaire")

    # Validation de l'ID de page
    if "pageid" not in page_data:
        raise ValueError("Le champ 'pageid' est requis")

    try:
        page_id = int(page_data["pageid"])
        if page_id <= 0:
            raise ValueError("L'ID de page doit être un entier positif")
    except (ValueError, TypeError):
        raise ValueError("L'ID de page doit être un entier valide")

    # Validation du titre
    if "title" not in page_data:
        raise ValueError("Le champ 'title' est requis")

    title = str(page_data["title"]).strip()
    if not title:
        raise ValueError("Le titre ne peut pas être vide")

    if len(title) < MIN_TITLE_LENGTH or len(title) > MAX_TITLE_LENGTH:
        raise ValueError(
            f"Le titre doit avoir entre {MIN_TITLE_LENGTH} "
            f"et {MAX_TITLE_LENGTH} caractères"
        )

    return page_id, title


def validate_article(
    page_data: Dict[str, Any],
    language_code: str,
    category_name: str,
    min_char_length: int,
    collected_ids: Set[int],
    api_url: str,
    sleep_time: Tuple[float, float],
    article_type: str = ARTICLE_TYPES["GENERIC"],
) -> Optional[Dict[str, Any]]:
    """
    Valide un article candidat selon les critères de qualité définis.

    Cette fonction vérifie si l'article :
    - n'a pas déjà été collecté
    - respecte la longueur minimale requise
    - possède un contenu valide et exploitable

    Args:
        page_data: dictionnaire avec 'pageid' et 'title'
        language_code: code ISO de la langue (ex: 'ru')
        category_name: nom de la catégorie d'origine
        min_char_length: longueur minimale en caractères
        collected_ids: ensemble des IDs d'articles déjà collectés
        api_url: URL de l'API Wikipedia
        sleep_time: délai pour les requêtes API
        article_type: type d'article ("ordonné", "aléatoire", "article")

    Returns:
        dictionnaire de l'article si valide, None sinon

    Raises:
        ValueError: si les paramètres sont invalides
    """
    # Validation des paramètres d'entrée
    if not isinstance(language_code, str) or not language_code:
        raise ValueError("Le code de langue doit être une chaîne non vide")

    if not isinstance(category_name, str) or not category_name:
        raise ValueError("Le nom de catégorie doit être une chaîne non vide")

    if (
        not isinstance(min_char_length, int)
        or min_char_length < MIN_CONTENT_LENGTH
    ):
        raise ValueError(
            f"La longueur minimale doit être un entier >= {MIN_CONTENT_LENGTH}"
        )

    if not isinstance(collected_ids, set):
        raise ValueError("collected_ids doit être un ensemble (set)")

    if article_type not in ARTICLE_TYPES.values():
        raise ValueError(
            f"Type d'article invalide: {article_type}. "
            f"Valeurs acceptées: {list(ARTICLE_TYPES.values())}"
        )

    try:
        page_id, title = validate_article_data(page_data)
    except ValueError as e:
        logging.warning(f"Données d'article invalides: {e}")
        return None

    # Vérifier si l'article a déjà été collecté
    if page_id in collected_ids:
        logging.debug(f"Article {page_id} déjà collecté, ignoré")
        return None

    # Récupérer le contenu de l'article
    try:
        extract, fetched_title = fetch_article_content(
            api_url, page_id, sleep_time
        )
        # Vérifier que le contenu est non vide et informatif
        if not extract or extract.strip() in ["", ".", "()", "[]", "{}"]:
            logging.warning(f"Contenu vide ou inutile pour l'article {page_id} : « {extract} »")
            return None
    except Exception as e:
        logging.error(
            f"Erreur lors de la récupération du contenu pour l'article {page_id}: {e}"
        )
        return None

    # Utiliser le titre récupéré si disponible, sinon le titre original
    final_title = fetched_title if fetched_title else title

    # Valider le contenu
    if not extract:
        logging.debug(f"Contenu non disponible pour l'article {page_id}: {final_title}")
        return None

    # Vérifier la longueur minimale
    if len(extract) < min_char_length:
        logging.debug(
            f"Article trop court ({len(extract)} < {min_char_length}): {final_title}"
        )
        mark_as_too_short(page_id)
        return None

    # Créer l'article validé
    validated_article = {
        "language": language_code,
        "title": final_title,
        "text": extract,
        "pageid": page_id,
        "url": f"https://{language_code}.wikipedia.org/?curid={page_id}",
        "category": category_name,
        "type": article_type,
        "token_count": estimate_token_count(extract),
        "char_count": len(extract),
    }

    logging.debug(
        f"Article validé: {final_title} "
        f"({len(extract)} caractères, "
        f"~{validated_article['token_count']} tokens)"
    )

    return validated_article


def process_article(
        article: Dict[str, Any],
        max_token_length: int
) -> Dict[str, Any]:
    """
    Traite un article validé en limitant son texte au nombre maximum de tokens.

    Cette fonction prend un article déjà validé et modifie son texte
    pour respecter la limite de tokens spécifiée, en utilisant un
    point de départ aléatoire pour préserver la diversité du contenu.

    Args:
        article: dictionnaire d'article validé contenant au minimum 'text'
        max_token_length: nombre maximum de tokens à conserver

    Returns:
        article modifié avec le texte traité et les métriques mises à jour

    Raises:
        ValueError: si les paramètres sont invalides
    """
    if not isinstance(article, dict):
        raise ValueError("L'article doit être un dictionnaire")

    if "text" not in article:
        raise ValueError("L'article doit contenir un champ 'text'")

    if not isinstance(max_token_length, int) or max_token_length <= 0:
        raise ValueError("La longueur maximale doit être un entier positif")

    original_text = article["text"]
    if not isinstance(original_text, str):
        raise ValueError("Le texte de l'article doit être une chaîne")

    # Traiter le texte
    processed_text, token_count = process_text(original_text, max_token_length)

    # Mettre à jour l'article avec les nouvelles données
    article_copy = article.copy()
    article_copy["text"] = processed_text
    article_copy["token_count"] = token_count
    article_copy["char_count"] = len(processed_text)
    article_copy["is_truncated"] = token_count < estimate_token_count(original_text)

    # Ajouter des métadonnées de traitement
    if article_copy["is_truncated"]:
        original_tokens = estimate_token_count(original_text)
        article_copy["truncation_ratio"] = (
            token_count / original_tokens if original_tokens > 0 else 0
        )
        logging.debug(
            f"Article tronqué: {article.get('title', 'Sans titre')} "
            f"({original_tokens} -> {token_count} tokens)"
        )

    return article_copy


def process_text(text: str, max_tokens: int) -> Tuple[str, int]:
    """
    Traite un texte en le limitant à un nombre maximum de tokens.

    Pour les textes longs, utilise un point de départ aléatoire
    pour préserver la diversité du contenu collecté.

    Args:
        text: texte à traiter
        max_tokens: nb maximum de tokens à conserver

    Returns:
        tuple (texte_traité, nombre_de_tokens)

    Raises:
        ValueError: si les paramètres sont invalides
    """
    if not isinstance(text, str):
        raise ValueError("Le texte doit être une chaîne")

    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("Le nombre maximum de tokens doit être un entier positif")

    # Tokenisation simple par espaces
    tokens = text.split()

    # Si le texte est plus court que max_tokens, le renvoyer tel quel
    if len(tokens) <= max_tokens:
        return text, len(tokens)

    # Sélectionner un point de départ aléatoire pour préserver la diversité
    max_start_idx = len(tokens) - max_tokens
    start_idx = random.randint(0, max_start_idx)

    # Extraire les tokens à partir du point de départ
    selected_tokens = tokens[start_idx:start_idx+max_tokens]
    processed_text = " ".join(selected_tokens)

    logging.debug(
        f"Texte tronqué: {len(tokens)} -> {len(selected_tokens)} tokens "
        f"(début à l'index {start_idx})"
    )

    return processed_text, len(selected_tokens)


def filter_processed_candidates(
    candidates: List[Dict[str, Any]], already_collected_ids: Set[int]
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Filtre les candidats pour exclure ceux déjà traités.

    Args:
        candidates: liste des articles candidats
        already_collected_ids: IDs d'articles déjà collectés

    Returns:
        tuple (candidats_filtrés, nb_trop_courts_ignorés, nb_déjà_collectés_ignorés)

    Raises:
        ValueError: si les paramètres sont invalides
    """
    if not isinstance(candidates, list):
        raise ValueError("Les candidats doivent être une liste")

    if not isinstance(already_collected_ids, set):
        raise ValueError("already_collected_ids doit être un ensemble")

    # Filtrer les candidats déjà identifiés comme trop courts
    filtered_candidates = []
    too_short_count = 0
    already_collected_count = 0

    for candidate in candidates:
        try:
            page_id, _ = validate_article_data(candidate)

            if is_too_short(page_id):
                too_short_count += 1
                continue

            if page_id in already_collected_ids:
                already_collected_count += 1
                continue

            filtered_candidates.append(candidate)

        except ValueError:
            # Ignorer les candidats avec des données invalides
            continue

    return filtered_candidates, too_short_count, already_collected_count


def select_valid_articles(
    candidates: List[Dict[str, Any]],
    num_needed: int,
    already_collected_ids: Set[int],
    min_length: int,
    language_code: str,
    category_name: str,
    sleep_time: Tuple[float, float],
    api_url: str,
    article_type: str = ARTICLE_TYPES["ORDERED"],
) -> List[Dict[str, Any]]:
    """
    Sélectionne et valide des articles en vérifiant leur qualité.

    Cette fonction optimise la sélection en filtrant d'abord les candidats
    déjà traités, puis en validant les articles restants selon l'ordre
    approprié (ordonné ou aléatoire).

    Args:
        candidates: liste des articles candidats
        num_needed: nombre d'articles valides à trouver
        already_collected_ids: IDs d'articles déjà collectés à éviter
        min_length: longueur minimale du texte pour qu'un article soit valide
        language_code: code de la langue
        category_name: nom de la catégorie
        sleep_time: temps d'attente entre les requêtes
        api_url: URL de l'API Wikipedia
        article_type: type d'article ("ordonné" ou "aléatoire")

    Returns:
        liste d'articles valides avec leur contenu complet

    Raises:
        ValueError: si les paramètres sont invalides
    """
    # Validation des paramètres
    if not isinstance(candidates, list):
        raise ValueError("Les candidats doivent être une liste")

    if not isinstance(num_needed, int) or num_needed <= 0:
        raise ValueError("Le nombre d'articles nécessaires doit être un entier positif")

    if not isinstance(already_collected_ids, set):
        raise ValueError("already_collected_ids doit être un ensemble")

    if article_type not in ARTICLE_TYPES.values():
        raise ValueError(f"Type d'article invalide: {article_type}")

    # Filtrer les candidats déjà traités
    filtered_candidates, too_short_ignored, already_collected_ignored = (
        filter_processed_candidates(candidates, already_collected_ids)
    )

    # Afficher les statistiques de filtrage
    if too_short_ignored > 0:
        print(f"  {LOG_MESSAGES['CACHE_FILTERED'].format(count=too_short_ignored)}")

    if already_collected_ignored > 0:
        print(
            f"  {LOG_MESSAGES['ALREADY_PROCESSED'].format(count=already_collected_ignored)}"
        )

    remaining_count = len(filtered_candidates)
    print(f"  {remaining_count} articles candidats restants à examiner")

    if remaining_count == 0:
        print(
            "  Tous les articles disponibles ont déjà été traités, abandon de la tentative."
        )
        return []

    # Déterminer l'ordre de parcours selon le type d'article
    candidates_to_check = organize_candidates_by_type(filtered_candidates, article_type)

    # Collecter les articles valides
    valid_articles = []
    processed_count = 0

    for candidate in candidates_to_check:
        if len(valid_articles) >= num_needed:
            break

        processed_count += 1

        # Valider l'article
        validated_article = validate_article(
            candidate,
            language_code,
            category_name,
            min_length,
            already_collected_ids,
            api_url,
            sleep_time,
            article_type,
        )

        if validated_article:
            valid_articles.append(validated_article)
            print(
                f"  {LOG_MESSAGES['ARTICLE_VALID'].format(type=article_type, title=validated_article['title'], chars=validated_article['char_count'])}"
            )
        else:
            # Analyser pourquoi l'article a été rejeté
            log_rejection_reason(
                candidate,
                already_collected_ids,
                api_url,
                sleep_time,
                article_type
            )

    logging.info(
        f"Sélection terminée: {len(valid_articles)}/{num_needed} articles valides trouvés "
        f"({processed_count} candidats examinés)"
    )

    return valid_articles


def organize_candidates_by_type(
    candidates: List[Dict[str, Any]],
    article_type: str
) -> List[Dict[str, Any]]:
    """
    Organise les candidats selon le type d'article demandé.

    Args:
        candidates: liste des candidats
        article_type: type d'article ("ordonné" ou "aléatoire")

    Returns:
        liste des candidats réorganisée
    """
    if article_type == ARTICLE_TYPES["RANDOM"]:
        # Pour les articles aléatoires, mélanger l'ordre
        candidates_copy = candidates.copy()
        random.shuffle(candidates_copy)
        return candidates_copy
    else:
        # Pour les articles ordonnés, garder l'ordre original
        return candidates


def log_rejection_reason(
    candidate: Dict[str, Any],
    already_collected_ids: Set[int],
    api_url: str,
    sleep_time: Tuple[float, float],
    article_type: str,
) -> None:
    """
    Analyse et log la raison du rejet d'un article candidat.

    Args:
        candidate: article candidat rejeté
        already_collected_ids: IDs déjà collectés
        api_url: URL de l'API
        sleep_time: délai pour les requêtes
        article_type: type d'article
    """
    try:
        page_id, title = validate_article_data(candidate)

        if page_id in already_collected_ids:
            # Ne pas spammer les logs pour les articles déjà collectés
            return

        # Vérifier la raison du rejet
        try:
            extract, _ = fetch_article_content(api_url, page_id, sleep_time)

            if not extract or extract.strip() in ["", ".", "()", "[]", "{}"]:
                print(
                    f"  {LOG_MESSAGES['CONTENT_NOT_AVAILABLE'].format(type=article_type, title=title)}"
                )
            elif len(extract) < MIN_CONTENT_LENGTH:
                print(
                    f"  {LOG_MESSAGES['ARTICLE_TOO_SHORT'].format(type=article_type, title=title)}"
                )
                mark_as_too_short(page_id)
            else:
                print(
                    f"  {LOG_MESSAGES['ARTICLE_INVALID'].format(type=article_type, title=title)}"
                )

        except Exception:
            print(
                f"  {LOG_MESSAGES['ARTICLE_INVALID'].format(type=article_type, title=title)}"
            )

    except ValueError:
        print(f"  Article {article_type} ignoré (données invalides)")


def calculate_processing_stats(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calcule des statistiques sur un ensemble d'articles traités.

    Args:
        articles: liste des articles traités

    Returns:
        dictionnaire avec les statistiques calculées
    """
    if not isinstance(articles, list):
        raise ValueError("Les articles doivent être une liste")

    if not articles:
        return {
            "total_articles": 0,
            "total_tokens": 0,
            "total_chars": 0,
            "avg_tokens_per_article": 0,
            "avg_chars_per_article": 0,
            "truncated_articles": 0,
            "truncation_rate": 0,
        }

    total_tokens = sum(article.get("token_count", 0) for article in articles)
    total_chars = sum(article.get("char_count", 0) for article in articles)
    truncated_count = sum(
        1 for article in articles if article.get("is_truncated", False)
    )

    return {
        "total_articles": len(articles),
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "avg_tokens_per_article": total_tokens / len(articles),
        "avg_chars_per_article": total_chars / len(articles),
        "truncated_articles": truncated_count,
        "truncation_rate": truncated_count / len(articles),
    }
