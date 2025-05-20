# Script de collecte avec stratégie adaptative par groupe de langues
import requests
import pandas as pd
import time
import random
import os
import sys
import json
import signal
import logging
from datetime import datetime

# Import des modules
from config import (
    TIME_LIMIT,
    LANGUAGE_GROUPS,
    LANGUAGES,
    ALL_CATEGORIES,
    TARGET_TOKENS_BY_GROUP,
    BELARUSIAN_TARGET,
    MAX_DEPTHS_BY_GROUP,
    ADAPTIVE_PARAMS,
    CATEGORY_PREFIXES,
    CATEGORY_TRANSLATIONS
)
from api_utils import (
    fetch_category_articles,
    fetch_subcategories,
    fetch_random_article,
    fetch_article_content
)


# Créer le dossier de logs s'il n'existe pas
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Configuration du logging
logging.basicConfig(
    filename=f"{log_dir}/cyrillique_collecte.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Pour voir les logs dans la console également
console = logging.StreamHandler()
console.setLevel(logging.WARNING)  # seulement les warnings et erreurs
logging.getLogger('').addHandler(console)

# Message de début de script
logging.info("Démarrage du script de collecte pour les langues cyrilliques")


# Ensemble des articles trop courts, sur lesquels ne pas repasser
too_short_article_ids = set()

def get_language_group(language_code):
    """Détermine le groupe d'une langue"""
    for group, languages in LANGUAGE_GROUPS.items():
        if language_code in languages:
            return group
    return 'C'  # par défaut, considérer comme groupe C (moins doté)


def get_target_for_language(language_code):
    # Exception pour les variantes du bélarussien
    if language_code in ["be", "be-tarask"]:
        return BELARUSIAN_TARGET
        
    # Sinon, utiliser l'objectif standard du groupe
    group = get_language_group(language_code)
    return TARGET_TOKENS_BY_GROUP[group]


def get_adaptive_params(language_code):
    """Récupère les paramètres adaptatifs pour une langue donnée."""
    group = get_language_group(language_code)
    return ADAPTIVE_PARAMS[group]


def process_text(text, max_tokens):
    """
    Traite le texte en le limitant à un nombre maximum de tokens,
    avec un point de départ aléatoire pour les longs textes.
    
    Args:
        text: Texte à traiter
        max_tokens: Nombre maximum de tokens à garder
        
    Returns:
        tuple (texte traité, nombre de tokens)
    """
    # Tokeniser le texte (approx. simple: split sur les espaces)
    tokens = text.split()
    
    # Si le texte est plus court que max_tokens, le renvoyer tel quel
    if len(tokens) <= max_tokens:
        return text, len(tokens)
    
    # Sinon, sélectionner un point de départ aléatoire
    max_start_idx = len(tokens) - max_tokens
    start_idx = random.randint(0, max_start_idx)
    
    # Extraire les tokens à partir du point de départ
    selected_tokens = tokens[start_idx:start_idx + max_tokens]
    processed_text = ' '.join(selected_tokens)
    
    return processed_text, len(selected_tokens)


def select_valid_articles(candidates, num_needed, already_collected_ids, min_length, language_code,
                          category_name, sleep_time, api_url, article_type="ordonné"):
    """
    Sélectionne et valide des articles en vérifiant leur longueur.
    
    Args:
        candidates: liste des articles candidats
        num_needed: nombre d'articles valides à trouver
        already_collected_ids: IDs d'articles déjà collectés à éviter
        min_length: longueur minimale du texte pour qu'un article soit valide
        language_code: code de la langue
        category_name: nom de la catégorie
        sleep_time: temps d'attente entre les requêtes
        api_url: URL de l'API Wikipedia
        article_type: "ordonné" ou "aléatoire"
    
    Returns:
        liste d'articles valides avec leur contenu
    """
    global too_short_article_ids
    valid_articles = []
    
    # Filtrer d'abord les candidats déjà connus comme trop courts
    filtered_candidates = [ c for c in candidates if c['pageid'] not in too_short_article_ids ]
    already_collected_candidates = [ c for c in filtered_candidates if c['pageid'] in already_collected_ids ]
    print(f"  {len(candidates) - len(filtered_candidates)} articles déjà identifiés comme trop courts ignorés")
    print(f"  {len(already_collected_candidates)} articles déjà collectés ignorés")
    print(f"  {len(filtered_candidates) - len(already_collected_candidates)} articles candidats restants à examiner")

    if len(filtered_candidates) - len(already_collected_candidates) == 0:
        print("  Tous les articles disponibles ont déjà été traités, abandon de la tentative.")
        return valid_articles

    # Déterminer l'ordre de parcours
    if article_type == "aléatoire":
        # Pour les articles aléatoires, créer une liste d'indices et la mélanger
        indices = list(range(len(filtered_candidates)))
        random.shuffle(indices)
        candidates_to_check = [ filtered_candidates[idx] for idx in indices ]
    else:
        # Pour les articles ordonnés, garder l'ordre original
        candidates_to_check = filtered_candidates
    
    # Parcourir les candidats (dans l'ordre déterminé ci-dessus)
    for member in candidates_to_check:
        if len(valid_articles) >= num_needed:
            break

        page_id = member['pageid']
        title = member['title']
            
        # Vérifier si l'article a déjà été collecté
        if page_id in already_collected_ids:
            continue
            
        extract, _ = fetch_article_content(api_url, page_id, sleep_time)
        
        if extract:
            # Vérifier la longueur
            if len(extract) >= min_length:
                # Article valide : ajouter à la liste
                valid_articles.append({
                    "pageid": page_id,
                    "title": title,
                    "text": extract,
                    "language": language_code,
                    "category": category_name,
                    "type": article_type,
                    "url": f"https://{language_code}.wikipedia.org/?curid={page_id}"
                })
                print(f"  Article {article_type} valide trouvé: {title} ({len(extract)} caractères)")
            else:
                print(f"  Article {article_type} ignoré (trop court): {title}")
                # Marquer l'article comme trop court
                too_short_article_ids.add(page_id)
    
    return valid_articles


def get_articles_by_category(language_code, category_name, category_target, num_articles=20, 
                            fixed_ratio=0.6, sleep_time=(1, 3), already_collected_ids=None):
    """
    Sélectionne des articles valides par grande catégorie avec des paramètres adaptatifs.
    
    Args:
        language_code: code de la langue
        category_name: nom de la catégorie
        category_target: nombre de tokens cible pour cette catégorie
        num_articles: nombre m d'articles à récupérer
        fixed_ratio: proportion d'articles ordonnés vs. aléatoires
        sleep_time: délai entre les requêtes API
        already_collected_ids: IDs d'articles déjà collectés
    """
    logging.info(f"Récupération d'articles par catégorie pour {language_code}, catégorie: {category_name}")

    # Récupérer les paramètres adaptatifs
    params = get_adaptive_params(language_code)
    
    if already_collected_ids is None:
        already_collected_ids = set()
        
    api_url = f"https://{language_code}.wikipedia.org/w/api.php"
    articles = []
    
    # Obtenir le préfixe adéquat
    prefix = CATEGORY_PREFIXES.get(language_code, 'Category:')
    full_category = f"{prefix}{category_name}"
    
    try:
        # === RÉCUPÉRATION D'ARTICLES DANS L'ORDRE ===
        ordered_members = fetch_category_articles(
            api_url, 
            full_category, 
            limit=250,
            sort_method="sortkey",
            sort_direction="asc",
            sleep_time=sleep_time
        )

        # === RÉCUPÉRATION D'ARTICLES ALÉATOIRES ===
        random_strategies = [
            {"sort": "sortkey", "dir": "desc"},
            {"sort": "timestamp", "dir": "asc"}, 
            {"sort": "timestamp", "dir": "desc"}
        ]

        random_strategy = random.choice(random_strategies)

        random_members = fetch_category_articles(
            api_url,
            full_category,
            limit=250,
            sort_method=random_strategy["sort"],
            sort_direction=random_strategy["dir"],
            sleep_time=sleep_time
        )

        # Calculer cb d'articles de chaque type on veut en utilisant fixed_ratio
        valid_members_count = len(ordered_members) + len(random_members)
        if valid_members_count == 0:
            return []
            
        num_fixed_selection = min(int(num_articles * fixed_ratio), len(ordered_members))
        num_random_selection = min(num_articles - num_fixed_selection, len(random_members))
        
        # Calculer combien de tokens nous voulons pour chaque type d'articles
        ordered_tokens_target = int(category_target * fixed_ratio)
        random_tokens_target = category_target - ordered_tokens_target
        
        print(f"  {valid_members_count} articles disponibles dans la catégorie")
        print(f"  Objectifs en tokens: {ordered_tokens_target} ordonnés + {random_tokens_target} aléatoires")

        # Variables pour suivre les tokens collectés
        ordered_tokens_collected = 0
        random_tokens_collected = 0
        category_tokens = 0          # total des tokens pour la catégorie

        # === COLLECTE ET TRAITEMENT DES ARTICLES ORDONNÉS ===
        fixed_articles = select_valid_articles(
            ordered_members,
            num_fixed_selection,
            already_collected_ids,
            params['min_char_length'],
            language_code,
            category_name,
            sleep_time,
            api_url,
            "ordonné"
        )

        # Ajouter les articles ordonnés jusqu'à atteindre l'objectif
        articles_to_add = []
        for article in fixed_articles:
            article_text, token_count = process_text(article["text"], params['max_token_length'])
            article["text"] = article_text
            article["token_count"] = token_count
            
            # Ajouter l'article et mettre à jour les compteurs
            articles_to_add.append(article)
            ordered_tokens_collected += token_count
            category_tokens += token_count
            
            print(f"  Article ajouté: {article['title']} ({token_count} tokens)")
            
            # Si on a atteint l'objectif pour les articles ordonnés, arrêter
            if ordered_tokens_collected >= ordered_tokens_target:
                break

        # Mettre à jour les IDs déjà collectés
        for article in articles_to_add:
            already_collected_ids.add(article["pageid"])

        used_ids = { article['pageid'] for article in articles_to_add }

        # Si on n'a pas atteint l'objectif avec les articles ordonnés,
        # augmenter l'objectif pour les articles aléatoires
        remaining_tokens = category_target - ordered_tokens_collected
        if remaining_tokens > random_tokens_target:
            random_tokens_target = remaining_tokens

        # === COLLECTE ET TRAITEMENT DES ARTICLES ALÉATOIRES ===
        filtered_random_members = [m for m in random_members if m['pageid'] not in used_ids]

        random_articles = select_valid_articles(
            filtered_random_members,
            num_random_selection,
            already_collected_ids,
            params['min_char_length'],
            language_code,
            category_name,
            sleep_time,
            api_url,
            "aléatoire"
        )

        # Ajouter les articles aléatoires jusqu'à atteindre l'objectif
        for article in random_articles:
            if random_tokens_collected >= random_tokens_target:
                break
                
            article_text, token_count = process_text(article["text"], params['max_token_length'])
            article["text"] = article_text
            article["token_count"] = token_count
            
            # Ajouter l'article et mettre à jour les compteurs
            articles_to_add.append(article)
            random_tokens_collected += token_count
            category_tokens += token_count
            already_collected_ids.add(article["pageid"])
            
            print(f"  Article ajouté: {article['title']} ({token_count} tokens)")

        # Résumé des résultats
        print(f"  Résultat: {ordered_tokens_collected} tokens ordonnés + {random_tokens_collected} tokens aléatoires")
        print(f"  Total: {category_tokens}/{category_target} tokens collectés ({len(articles_to_add)} articles)")

        # Retourner les articles collectés
        articles = articles_to_add

        # Vérifier s'il y a des doublons dans la liste d'articles
        article_ids = [ article["pageid"] for article in articles ]
        unique_ids = set(article_ids)
        if len(article_ids) != len(unique_ids):
            print(f"ATTENTION: {len(article_ids) - len(unique_ids)} doublons détectés dans la liste d'articles!")
            # Identifier les doublons
            from collections import Counter
            duplicate_ids = [ item for item, count in Counter(article_ids).items() if count > 1 ]
            print(f"IDs en double: {duplicate_ids}")
        
    except Exception as e:
        logging.error(f"Erreur lors de la récupération des articles de {full_category}: {e}", exc_info=True)
    
    logging.info(f"Fin de la récupération pour {language_code}, catégorie: {category_name}, {len(articles)} articles trouvés")

    ordered_articles_count = sum(1 for a in articles if a.get("type") == "ordonné")
    random_articles_count = sum(1 for a in articles if a.get("type") == "aléatoire")

    total_articles = len(articles)
    total_tokens = category_tokens

    # Calculer les proportions
    ordered_articles_ratio = ordered_articles_count / total_articles if total_articles > 0 else 0
    random_articles_ratio = random_articles_count / total_articles if total_articles > 0 else 0
    ordered_tokens_ratio = ordered_tokens_collected / total_tokens if total_tokens > 0 else 0
    random_tokens_ratio = random_tokens_collected / total_tokens if total_tokens > 0 else 0

    # Logger ces informations
    logging.info(f"Distribution pour {category_name} (langue: {language_code}):")
    logging.info(f"- Articles: {ordered_articles_ratio:.2f} ordonnés vs {random_articles_ratio:.2f} aléatoires")
    logging.info(f"- Tokens: {ordered_tokens_ratio:.2f} ordonnés vs {random_tokens_ratio:.2f} aléatoires")
    logging.info(f"- Objectif initial: {fixed_ratio:.2f} ordonnés vs {1-fixed_ratio:.2f} aléatoires")

    return articles


# Récupérer des articles aléatoires dans les sous-catégories
def get_articles_from_subcategories(language_code, category_name, num_articles=10, max_depth=3, 
                                    sleep_time=(1, 3), already_collected_ids=None, min_char_length=500,
                                    cached_subcategories=None, attempt_number=1):
    """
    Récupère des articles aléatoires dans les sous-catégories d'une catégorie donnée.
    
    Args:
        language_code: code de la langue
        category_name: nom de la catégorie principale
        num_articles: nombre d'articles à récupérer
        max_depth: profondeur max de recherche dans les sous-catégories
        sleep_time: intervalle d'attente entre les requêtes
        already_collected_ids: ensemble des IDs d'articles déjà collectés à exclure
        min_char_length: longueur minimale en caractères pour qu'un article soit valide
        cached_subcategories: ensemble des sous-catégories déjà trouvées
                              (pour éviter de refaire l'exploration à chaque nouvelle tentative)
        
    Returns:
        liste de dictionnaires contenant les articles
    """
    logging.info(f"Recherche dans les sous-catégories pour {language_code}, catégorie: {category_name}")

    # Initialiser l'ensemble des IDs déjà collectés si non fourni
    if already_collected_ids is None:
        already_collected_ids = set()
        
    api_url = f"https://{language_code}.wikipedia.org/w/api.php"
    articles = []
    
    prefix = CATEGORY_PREFIXES.get(language_code, 'Category:')
    full_category = f"{prefix}{category_name}"
    
    # Initialiser le cache des sous-catégories si non fourni
    if cached_subcategories is None:
        # Ensemble pour stocker toutes les sous-catégories trouvées
        all_subcategories = set()
        # File d'attente pour l'exploration en largeur des sous-catégories
        subcategory_queue = [(full_category, 0)]  # (catégorie, profondeur)
    
        # Explorer les sous-catégories en largeur d'abord (BFS)
        while subcategory_queue and len(all_subcategories) < 800:
            current_category, depth = subcategory_queue.pop(0)
            
            # Si on a déjà exploré cette catégorie ou si on a atteint la profondeur max
            if current_category in all_subcategories or depth > max_depth:
                continue
                
            all_subcategories.add(current_category)
            
            # Si on n'est pas à la profondeur maximale, chercher les sous-catégories
            if depth <= max_depth:
                subcats = fetch_subcategories(api_url, current_category, 50, sleep_time)
                
                for subcat in subcats:
                    if 'title' in subcat:
                        subcategory_queue.append((subcat['title'], depth + 1))
        
        print(f"{len(all_subcategories)} sous-catégories trouvée(s) pour '{category_name}'.")

        # Retourner le cache des sous-catégories pour les tentatives futures
        cached_subcategories = all_subcategories

    else:
        all_subcategories = cached_subcategories
        print(f"Utilisation de {len(all_subcategories)} sous-catégories déjà explorées pour '{category_name}'.")
    
    # Exploration secondaire à partir d'un point aléatoire lors des tentatives ultérieures
    if attempt_number > 1 and len(all_subcategories) > 10:
        # Convertir en liste pour pouvoir sélectionner une sous-catégorie aléatoire
        subcats_list = list(all_subcategories)
        random_subcat_idx = random.randint(0, len(subcats_list) - 1)
        random_subcat = subcats_list[random_subcat_idx]
        
        print(f"  Tentative {attempt_number}: exploration secondaire à partir de '{random_subcat}'")
        
        # Créer une file d'attente pour cette exploration secondaire
        new_queue = [(random_subcat, 0)]
        new_explored = set([random_subcat])  # sous-catégories déjà explorées dans cette session
        
        # Explorer à partir de ce point aléatoire (profondeur limitée)
        secondary_depth = max(2, max_depth // 2)  # profondeur réduite pour l'exploration secondaire
        
        while new_queue and len(new_explored) < 300:  # limite le nb de nouvelles sous-catégories
            current_cat, depth = new_queue.pop(0)
            
            # Si déjà à la profondeur maximale pour cette exploration secondaire, passer
            if depth > secondary_depth:
                continue
            
            new_subcats = fetch_subcategories(api_url, current_cat, 50, sleep_time)
                    
            # Ajouter les nouvelles sous-catégories à la file
            for subcat in new_subcats:
                if 'title' in subcat:
                    subcat_title = subcat['title']
                    if subcat_title not in all_subcategories and subcat_title not in new_explored:
                        new_queue.append((subcat_title, depth + 1))
                        new_explored.add(subcat_title)
        
        # Ajouter les nouvelles sous-catégories trouvées au cache principal
        print(f"  Exploration secondaire: {len(new_explored)} sous-catégories découvertes")
        all_subcategories.update(new_explored)
        cached_subcategories = all_subcategories  # Mettre à jour le cache
    
    # Après l'exploration secondaire et l'enrichissement de all_subcategories
    # Si aucune sous-catégorie n'est trouvée, retourner une liste vide
    if not all_subcategories:
        return [], None
        
    # Convertir l'ensemble en liste pour pouvoir utiliser random.sample
    subcategories_list = list(all_subcategories)
    
    # Rotation de la liste (démarrage à un point aléatoire)
    if len(subcategories_list) > 50:
        # Choisir un point de départ aléatoire
        start_idx = random.randint(0, len(subcategories_list) - 1)
        # Réorganiser la liste pour commencer à ce point
        reorganized_list = subcategories_list[start_idx:] + subcategories_list[:start_idx]
        # Prendre les 50 premiers éléments de cette liste réorganisée
        selected_subcats = reorganized_list[:50]
    else:
        # Pour les petites listes, prendre tous les éléments
        selected_subcats = subcategories_list
    
    articles_collected_count = 0
    subcats_explored = 0
    
    # Continuer à explorer plus de sous-catégories si nécessaire
    for subcat in selected_subcats:
        subcats_explored += 1
        
        # Si on a atteint l'objectif d'articles, on arrête
        if articles_collected_count >= num_articles:
            break
            
        members = fetch_category_articles(api_url, subcat, 50, sleep_time=sleep_time)
                
        # S'il y a des articles valides
        if members:
            sample_size = min(len(members), 5)  # (augmenté de 3 à 5)
            selected_members = random.sample(members, sample_size)
            
            for member in selected_members:
                # Si on a atteint le nombre d'articles souhaité, on arrête
                if len(articles) >= num_articles:
                    break
                    
                page_id = member['pageid']
                title = member['title']
                
                # Vérifier si l'article a déjà été collecté
                if page_id in already_collected_ids:
                    continue
                    
                extract, _ = fetch_article_content(api_url, page_id, sleep_time)
                    
                # Vérifier que l'article a un contenu suffisant
                if extract and len(extract) >= min_char_length:
                    # Ajouter l'article à la liste
                    articles.append({
                        "language": language_code,
                        "title": title,
                        "text": extract,
                        "page_id": page_id,
                        "url": f"https://{language_code}.wikipedia.org/?curid={page_id}",
                        "category": f"{category_name} (Sous-catégorie)"
                    })
                    already_collected_ids.add(page_id)
                    articles_collected_count += 1
                    print(f"Article récupéré: {title} ({language_code}, {category_name})")
                else:
                    print(f"Article ignoré (trop court): {title}")
    
    # Si on n'a pas atteint l'objectif, essayer plus de sous-catégories
    if articles_collected_count < num_articles and subcats_explored < len(selected_subcats):
        print(f"  Objectif non atteint, exploration de sous-catégories supplémentaires...")
        additional_subcats = selected_subcats[subcats_explored:]

        empty_subcats_count = 0
        max_empty_subcats = 10  # après 10 sous-catégories vides consécutives, on arrête
        
        for subcat in additional_subcats[:20]:  # essayer jusqu'à 20 sous-catégories supplémentaires
            if articles_collected_count >= num_articles:
                break
                
            subcats_explored += 1
            
            members = fetch_category_articles(api_url, subcat, 50, sleep_time=sleep_time)
                    
            # S'il y a des articles valides
            if members:
                # Sélectionner aléatoirement quelques articles
                sample_size = min(len(members), 5)
                selected_members = random.sample(members, sample_size)
                
                for member in selected_members:
                    # Si on a atteint le nombre d'articles souhaité, on arrête
                    if articles_collected_count >= num_articles:
                        break
                        
                    page_id = member['pageid']
                    title = member['title']
                    
                    # Vérifier si l'article a déjà été collecté
                    if page_id in already_collected_ids:
                        continue
                        
                    extract, _ = fetch_article_content(api_url, page_id, sleep_time)
                    
                    # Vérifier que l'article a un contenu suffisant
                    if extract and len(extract) >= min_char_length:
                        # Ajouter l'article à la liste
                        articles.append({
                            "language": language_code,
                            "title": title,
                            "text": extract,
                            "page_id": page_id,
                            "url": f"https://{language_code}.wikipedia.org/?curid={page_id}",
                            "category": f"{category_name} (Sous-catégorie)"
                        })
                        already_collected_ids.add(page_id)
                        articles_collected_count += 1
                        print(f"  Article supplémentaire {articles_collected_count}/{num_articles}: {title}")
                    else:
                        print(f"  Article ignoré (trop court): {title}")
            
            if not members:  # si la sous-catégorie est vide
                empty_subcats_count += 1
                if empty_subcats_count >= max_empty_subcats:
                    print(f"  Trop de sous-catégories vides consécutives, arrêt de la recherche supplémentaire.")
                    break
            else:
                empty_subcats_count = 0  # réinitialiser le compteur
        
        if articles_collected_count < num_articles:
            print(f"  Terminé avec {articles_collected_count} articles sur {num_articles} objectif")

    print(f"  Total: {articles_collected_count} articles collectés sur {num_articles} objectif")
    logging.info(f"Récupération terminée, {len(articles)} articles trouvés dans les sous-catégories")

    return articles, cached_subcategories


# Récupérer des articles aléatoirement (toutes catégories confondues)
def get_random_articles(language_code, num_articles=20, sleep_time=(1, 3),
                        already_collected_ids=None, min_char_length=500):
    """
    Récupère aléatoirement des articles de Wikipédia dans la langue spécifiée.
    
    Args:
        language_code: code de langue Wikipédia (par ex., 'ru', 'uk', 'be'...)
        num_articles: nb d'articles à récupérer
        sleep_time: tuple (min, max) pour le temps d'attente entre les requêtes
        already_collected_ids: ensemble des IDs d'articles déjà collectés à exclure
        min_char_length: longueur minimale en caractères pour qu'un article soit valide
        
    Returns:
        liste de dictionnaires contenant les articles avec leur texte et métadonnées
    """
    logging.info(f"Récupération d'articles aléatoires pour {language_code}, objectif: {num_articles} articles")

    # Initialiser l'ensemble des IDs déjà collectés si non fourni
    if already_collected_ids is None:
        already_collected_ids = set()
        
    api_url = f"https://{language_code}.wikipedia.org/w/api.php"
    articles = []
    
    # Compteur pour éviter les boucles infinies
    articles_collected = 0
    attempts = 0
    max_attempts = num_articles * 15  # limite pour éviter les boucles infinies
    
    # Continuer jusqu'à obtenir le nombre d'articles demandé ou atteindre le max de tentatives
    while articles_collected < num_articles and attempts < max_attempts:
        attempts += 1
        
        # 1. Obtenir un article aléatoire
        random_article = fetch_random_article(api_url, sleep_time)

        if not random_article:
            print(f"Impossible de récupérer un article aléatoire (tentative {attempts})")
            continue
            
        page_id = random_article['id']
        title = random_article['title']
        
        # Vérifier si l'article a déjà été collecté
        if page_id in already_collected_ids:
            print(f"Article déjà collecté, on passe: {title}")
            continue
        
        # 2. Récupérer le contenu de l'article
        extract, _ = fetch_article_content(api_url, page_id, sleep_time)
        
        # Si l'article est valide, l'ajouter et incrémenter le compteur
        if extract and len(extract) >= min_char_length:
            articles.append({
                "language": language_code,
                "title": title,
                "text": extract,
                "page_id": page_id,
                "url": f"https://{language_code}.wikipedia.org/?curid={page_id}",
                "category": "Random"  # marquer comme aléatoire pour les stats
            })
            already_collected_ids.add(page_id)
            articles_collected += 1
            print(f"Article {articles_collected}/{num_articles} récupéré: {title} ({language_code})")
        else:
            print(f"Article ignoré (trop court): {title} ({language_code})")
    
    if articles_collected < num_articles:
        logging.warning(f"Attention: Seulement {articles_collected}/{num_articles} articles aléatoires trouvés pour {language_code}")
    
    logging.info(f"Récupération d'articles aléatoires terminée pour {language_code}: {articles_collected} articles trouvés")

    return articles


def collect_articles(language_code, categories):
    """
        Version améliorée de la fonction de collecte avec adaptation par groupe de langue.
    """
    logging.info(f"Démarrage de la collecte pour la langue: {language_code}")

    # Récupérer le groupe de la langue
    group = get_language_group(language_code)
    # Récupérer l'objectif de tokens pour la langue
    token_target = get_target_for_language(language_code)
    
    print(f"\n\n===== COLLECTE POUR LA LANGUE: {language_code} =====")
    print(f"Groupe de langue: {group}")
    print(f"Objectif de tokens adaptatif: {token_target}")
    
    # Récupérer les paramètres adaptatifs pour la langue
    params = get_adaptive_params(language_code)
    
    print(f"Paramètres adaptatifs:")
    print(f"- Longueur minimale: {params['min_char_length']} caractères")
    print(f"- Longueur maximale: {params['max_token_length']} tokens")
    print(f"- Ratio catégories principales: {params['main_category_ratio']*100}%")
    print(f"- Ratio sous-catégories: {params['subcategory_ratio']*100}%")
    print(f"- Ratio articles aléatoires: {params['random_ratio']*100}%")
    print(f"- Proportion de sélection ordonnée: {params['fixed_selection_ratio']*100}%")
    
    # Calculer les objectifs de tokens pour chaque méthode
    main_category_token_target = int(token_target * params['main_category_ratio'])
    subcategory_token_target = int(token_target * params['subcategory_ratio'])
    # Calculer le reste pour la dernière méthode
    random_token_target = token_target - main_category_token_target - subcategory_token_target
    print(f"Objectif: {token_target} tokens | Démarré à {datetime.now().strftime('%H:%M:%S')}")
    
    main_ordered_tokens = 0
    main_random_tokens = 0

    start_time = time.time()
    available_categories = []
    
    # Déterminer quelles catégories sont disponibles pour cette langue
    for category in categories:
        if language_code in CATEGORY_TRANSLATIONS[category]:
            available_categories.append(category)
    
    print(f"\nCatégories disponibles ({len(available_categories)}/{len(categories)}): {', '.join(available_categories)}")
    
    if not available_categories:
        logging.warning(f"Attention: aucune catégorie disponible pour {language_code}")
        return []
    
    # Répartir les tokens entre les catégories disponibles
    tokens_per_main_category = main_category_token_target // len(available_categories)
    tokens_per_subcategory = subcategory_token_target // len(available_categories)
    
    print(f"Plan de collecte adaptatif:")
    print(f"- Tokens catégories principales: {main_category_token_target} ({tokens_per_main_category} par catégorie)")
    print(f"- Tokens sous-catégories: {subcategory_token_target} ({tokens_per_subcategory} par catégorie)")
    print(f"- Tokens articles aléatoires: {random_token_target}")

    # Initialiser categories_stats avant de commencer la collecte
    categories_stats = {}
    for category in categories:
        categories_stats[category] = {
            "main_articles": 0,
            "main_tokens": 0,
            "sub_articles": 0,
            "sub_tokens": 0,
            "available_articles": 0
        }
    
    articles = []
    collected_article_ids = set()
    
    total_tokens = 0
    main_category_tokens = 0
    subcategory_tokens = 0
    random_tokens = 0
    
    # 1. Collecter des articles des catégories principales
    print(f"\n1. Collecte d'articles des catégories principales (objectif: {main_category_token_target} tokens)")
    
    for category in available_categories:
        # Vérifier si on a dépassé la limite de temps
        if time.time() - start_time > TIME_LIMIT:
            print(f"Limite de temps atteinte pour {language_code}. Passage à la langue suivante.")
            break
            
        translated_category = CATEGORY_TRANSLATIONS[category][language_code]
        print(f"\n  Catégorie: {category} ({translated_category})")
        
        # Adapter le batch_size en fonction des besoins
        category_target = tokens_per_main_category
        category_tokens = 0
        attempts = 0
        max_attempts = 15 # nb maximum de tentatives
        
        while category_tokens < category_target and attempts < max_attempts:
            attempts += 1
            # Calculer le batch_size en fonction des tokens restants
            remaining_tokens = category_target - category_tokens

            if group in ['A', 'B']:
                batch_size = max(5, min(15, remaining_tokens // 200))  # plus petit batch initial
            else:
                batch_size = max(10, min(30, remaining_tokens // 100)) # plus grand pour le groupe C
            
            print(f"  Tentative {attempts}: recherche de {batch_size} articles (objectif: {remaining_tokens} tokens manquants)")
            
            try:
                category_articles = get_articles_by_category(
                    language_code,
                    translated_category,
                    category_target=category_target,
                    num_articles=batch_size,
                    fixed_ratio=params['fixed_selection_ratio'],
                    sleep_time=(1, 2.5),
                    already_collected_ids=collected_article_ids
                )
                
                # Enregistrer le nombre d'articles disponibles
                if category_articles:  # s'assurer qu'il y a des articles
                    categories_stats[category]["available_articles"] = len(category_articles)
                
                # Si pas d'articles disponibles, passer à la catégorie suivante
                if not category_articles:
                    print(f"  Aucun article disponible pour {category} ({translated_category})")
                    break
                
                # Ajouter les articles jusqu'à atteindre l'objectif pour cette catégorie
                for article in category_articles:
                    # Ajouter l'article à la liste principale
                    articles.append(article)
                    
                    # Récupérer le nombre de tokens du résultat
                    token_count = article.get("token_count", 0)
                    
                    # Mettre à jour les compteurs
                    category_tokens += token_count
                    main_category_tokens += token_count
                    total_tokens += token_count
                    
                    # Mettre à jour les statistiques
                    categories_stats[category]["main_articles"] += 1
                    categories_stats[category]["main_tokens"] += token_count
                    
                    if article.get("type") == "ordonné":
                        main_ordered_tokens += article.get("token_count", 0)
                    elif article.get("type") == "aléatoire":
                        main_random_tokens += article.get("token_count", 0)

                    # Si on a atteint l'objectif pour cette catégorie, passer à la suivante
                    if category_tokens >= category_target:
                        break
                    
                    # Vérifier si on a dépassé la limite de temps 
                    if time.time() - start_time > TIME_LIMIT:
                        print(f"Limite de temps atteinte pendant la collecte de {category}.")
                        break
                
                print(f"  Progression: {category_tokens}/{category_target} tokens collectés "
                      f"({category_target - category_tokens} tokens manquants)")
            
            except Exception as e:
                logging.error(f"Erreur lors de la collecte pour {category}: {str(e)}", exc_info=True)
                continue
    
    # Afficher le résumé de la collecte des catégories principales
    print(f"\nTotal collecté pour les catégories principales: {main_category_tokens}/{main_category_token_target} tokens "
          f"({main_category_tokens/main_category_token_target*100 if main_category_token_target > 0 else 0:.1f}%)")
    
    # Vérifier si on a dépassé la limite de temps
    if time.time() - start_time > TIME_LIMIT:
        print(f"Limite de temps atteinte pour {language_code}. Collecte des sous-catégories et articles aléatoires annulée.")
        # Même en cas de dépassement de temps, on continue pour collecter des statistiques partielles
        pass
    else:
        # 2. Collecter des articles des sous-catégories
        print(f"\n2. Collecte d'articles des sous-catégories (objectif: {subcategory_token_target} tokens)")

        # Pour chaque catégorie
        subcategories_cache = {}  # dictionnaire pour stocker les sous-catégories par catégorie

        for category in available_categories:
            # Vérifier si on a dépassé la limite de temps
            if time.time() - start_time > TIME_LIMIT:
                print(f"Limite de temps atteinte pendant la collecte des sous-catégories.")
                break
                
            translated_category = CATEGORY_TRANSLATIONS[category][language_code]
            print(f"\n  Sous-catégories de: {category} ({translated_category})")
            
            # Logique de collecte pour chaque catégorie
            category_target = tokens_per_subcategory
            category_tokens = 0
            attempts = 0
            max_attempts = 15

            # Récupérer le cache des sous-catégories si disponible
            cached_subcats = subcategories_cache.get(category, None)
            
            while category_tokens < category_target and attempts < max_attempts:
                attempts += 1
                remaining_tokens = category_target - category_tokens

                if group in ['A', 'B']:
                    batch_size = max(5, min(15, remaining_tokens // 200))  # plus petit batch initial
                else:
                    batch_size = max(10, min(30, remaining_tokens // 100)) # plus grand pour le groupe C
                
                print(f"  Tentative {attempts}: recherche de {batch_size} articles")
                
                try:
                    # Récupérer la profondeur adaptée au groupe de langue
                    max_depth = MAX_DEPTHS_BY_GROUP[group]

                    subcategory_articles, updated_cache = get_articles_from_subcategories(
                        language_code,
                        translated_category,
                        num_articles=batch_size,
                        max_depth=max_depth,                         # profondeur adaptative
                        sleep_time=(0.5, 1.5),
                        already_collected_ids=collected_article_ids,
                        min_char_length=params['min_char_length'],   # ajoute ce paramètre
                        cached_subcategories=cached_subcats,         # passer le cache
                        attempt_number=attempts
                    )

                    # Mettre à jour le cache pour les tentatives suivantes
                    if updated_cache:
                        subcategories_cache[category] = updated_cache
                        cached_subcats = updated_cache
                    
                    if not subcategory_articles:
                        print(f"  Point de départ infructueux, essai d'autres points...")
                        
                        # Essayer jusqu'à 5 autres points de départ dans cette même tentative
                        max_retry_points = 5
                        retry_success = False
                        
                        # Besoin d'accéder à la liste des sous-catégories
                        all_subcats_list = list(cached_subcats) if cached_subcats else []
                        
                        if len(all_subcats_list) > 10:
                            # Essayer plusieurs points de départ
                            for retry in range(max_retry_points):
                                # Choisir un nouveau point de départ aléatoire
                                new_start_idx = random.randint(0, len(all_subcats_list) - 1)
                                new_start_point = all_subcats_list[new_start_idx]
                                
                                print(f"  Nouvel essai {retry+1}/{max_retry_points} à partir de '{new_start_point}'")
                                
                                # Explorer récursivement à partir de ce point avec une profondeur limitée
                                retry_articles = []
                                explored_subcats = set()
                                subcats_to_explore = [(new_start_point, 0)]  # (catégorie, profondeur)
                                
                                # Profondeur maximale d'exploration
                                max_retry_depth = 3
                                
                                # Explorer jusqu'à un certain nombre de sous-catégories ou d'articles
                                max_subcats_to_explore = 50
                                api_url = f"https://{language_code}.wikipedia.org/w/api.php"
                                
                                while subcats_to_explore and len(explored_subcats) < max_subcats_to_explore:
                                    current_cat, depth = subcats_to_explore.pop(0)
                                    
                                    if current_cat in explored_subcats or depth > max_retry_depth:
                                        continue
                                    
                                    explored_subcats.add(current_cat)
                                    print(f"    Exploration de: {current_cat}, profondeur: {depth}")
                                    
                                    try:
                                        # D'abord, chercher des articles dans cette catégorie
                                        article_params = {
                                            "action": "query",
                                            "format": "json",
                                            "list": "categorymembers",
                                            "cmtitle": current_cat,
                                            "cmlimit": "50",
                                            "cmtype": "page"
                                        }
                                        
                                        time.sleep(random.uniform(0.5, 1.5))
                                        article_response = requests.get(api_url, params=article_params).json()
                                        
                                        if 'query' in article_response and 'categorymembers' in article_response['query']:
                                            members = article_response['query']['categorymembers']
                                            
                                            if members:
                                                sample_size = min(len(members), 3)  # limiter pour accélérer
                                                selected_members = random.sample(members, sample_size)
                                                
                                                for member in selected_members:
                                                    page_id = member['pageid']
                                                    title = member['title']
                                                    
                                                    if page_id in collected_article_ids:
                                                        continue
                                                    
                                                    # Récupérer le contenu
                                                    content_params = {
                                                        "action": "query",
                                                        "format": "json",
                                                        "prop": "extracts",
                                                        "explaintext": "1",
                                                        "pageids": str(page_id)
                                                    }
                                                    
                                                    time.sleep(random.uniform(0.5, 1.5))
                                                    content_response = requests.get(api_url, params=content_params).json()
                                                    
                                                    try:
                                                        extract = content_response["query"]["pages"][str(page_id)]["extract"]
                                                        
                                                        if len(extract) >= params['min_char_length']:
                                                            retry_articles.append({
                                                                "language": language_code,
                                                                "title": title,
                                                                "text": extract,
                                                                "page_id": page_id,
                                                                "url": f"https://{language_code}.wikipedia.org/?curid={page_id}",
                                                                "category": f"{category} (Sous-catégorie)"
                                                            })
                                                            collected_article_ids.add(page_id)
                                                            print(f"    Article trouvé: {title}")
                                                            
                                                            # Si on a trouvé suffisamment d'articles, arrêter
                                                            if len(retry_articles) >= batch_size // 2:
                                                                break
                                                    except KeyError:
                                                        print(f"    Erreur lors de l'extraction pour {title}")
                                        
                                        # Si on a trouvé suffisamment d'articles, arrêter l'exploration
                                        if len(retry_articles) >= batch_size // 2:
                                            break
                                        
                                        # Sinon, explorer les sous-catégories
                                        if depth < max_retry_depth:
                                            subcat_params = {
                                                "action": "query",
                                                "format": "json",
                                                "list": "categorymembers",
                                                "cmtitle": current_cat,
                                                "cmlimit": "50",
                                                "cmtype": "subcat"
                                            }
                                            
                                            time.sleep(random.uniform(0.5, 1.5))
                                            subcat_response = requests.get(api_url, params=subcat_params).json()
                                            
                                            if 'query' in subcat_response and 'categorymembers' in subcat_response['query']:
                                                subcats = subcat_response['query']['categorymembers']
                                                
                                                for subcat in subcats:
                                                    if 'title' in subcat:
                                                        # Ajouter à la file d'attente pour exploration
                                                        subcats_to_explore.append((subcat['title'], depth + 1))
                                    
                                    except Exception as e:
                                        print(f"    Erreur lors de l'exploration de {current_cat}: {str(e)}")
                                
                                # Après l'exploration complète à partir de ce point de départ
                                if retry_articles:
                                    subcategory_articles = retry_articles
                                    retry_success = True
                                    print(f"  Succès! {len(retry_articles)} articles trouvés après exploration profonde!")
                                    break
                            
                            if not retry_success:
                                print(f"  Tous les points d'entrée tentés sont infructueux pour {category}")
                                break  # abandonner cette catégorie
                        else:
                            print(f"  Pas assez de sous-catégories disponibles pour tenter d'autres points de départ")
                            break
                    
                    # Ajouter les articles collectés
                    for article in subcategory_articles:
                        # Traiter et limiter le texte
                        article_text, token_count = process_text(article["text"], params['max_token_length'])
                        article["text"] = article_text
                        article["token_count"] = token_count
                        
                        # Ajouter l'article et mettre à jour les compteurs
                        articles.append(article)
                        collected_article_ids.add(article["page_id"])
                        
                        category_tokens += token_count
                        subcategory_tokens += token_count
                        total_tokens += token_count
                        
                        # Mettre à jour les statistiques
                        categories_stats[category]["sub_articles"] += 1
                        categories_stats[category]["sub_tokens"] += token_count
                        
                        print(f"  Article ajouté: {article['title']} ({token_count} tokens)")
                        
                        # Si on a atteint l'objectif pour cette catégorie, arrêter
                        if category_tokens >= category_target:
                            break
                    
                    print(f"  → Sous-catégories de {category}: {category_tokens} tokens collectés sur {category_target} ciblés")
                    
                except Exception as e:
                    logging.error(f"  Erreur lors de la collecte des sous-catégories pour {category}: {str(e)}", exc_info=True)
                    break

        # Afficher le résumé de la collecte des sous-catégories
        print(f"\nTotal collecté pour les sous-catégories: {subcategory_tokens}/{subcategory_token_target} tokens "
            f"({subcategory_tokens/subcategory_token_target*100 if subcategory_token_target > 0 else 0:.1f}%)")
        
        # Vérifier si on a dépassé la limite de temps
        if time.time() - start_time > TIME_LIMIT:
            print(f"Limite de temps atteinte pour {language_code}. Collecte des articles aléatoires annulée.")
        else:
            # 3. Collecter des articles aléatoires
            print(f"\n3. Collecte d'articles aléatoires (objectif: {random_token_target} tokens)")
            
            try:
                while random_tokens < random_token_target:
                    # Vérifier si on a dépassé la limite de temps
                    if time.time() - start_time > TIME_LIMIT:
                        print(f"Limite de temps atteinte pendant la collecte des articles aléatoires.")
                        break
                    
                    # Calculer combien d'articles collecter pour cette itération
                    remaining_tokens = random_token_target - random_tokens
                    batch_size = max(1, min(5, remaining_tokens // 500))
                    
                    random_articles = get_random_articles(
                        language_code,
                        num_articles=batch_size,
                        sleep_time=(0.5, 2),
                        already_collected_ids=collected_article_ids,
                        min_char_length=params['min_char_length']
                    )
                    
                    if not random_articles:
                        print(f"  Plus d'articles aléatoires disponibles pour {language_code}")
                        break
                    
                    for article in random_articles:
                        # Traiter et limiter le texte
                        article_text, token_count = process_text(article["text"], params['max_token_length'])
                        article["text"] = article_text
                        article["token_count"] = token_count
                        
                        # Ajouter l'article et mettre à jour les compteurs
                        articles.append(article)
                        collected_article_ids.add(article["page_id"])
                        
                        random_tokens += token_count
                        total_tokens += token_count
                        
                        print(f"  Article aléatoire ajouté: {article['title']} ({token_count} tokens)")
                        
                        # Vérifier si on a atteint l'objectif
                        if random_tokens >= random_token_target:
                            break
                        
                        # Vérifier si on a dépassé la limite de temps
                        if time.time() - start_time > TIME_LIMIT:
                            print(f"Limite de temps atteinte pendant la collecte des articles aléatoires.")
                            break
                    
                    print(f"  Articles aléatoires: {random_tokens}/{random_token_target} tokens collectés "
                          f"({random_tokens/random_token_target*100 if random_token_target > 0 else 0:.1f}%)")
                    
            except Exception as e:
                logging.error(f"  Erreur lors de la collecte des articles aléatoires: {str(e)}", exc_info=True)
    
    logging.info(f"Collecte terminée pour {language_code}: {len(articles)} articles, {total_tokens} tokens")

    # Calculer le temps d'exécution total
    execution_time = time.time() - start_time

    # Créer le répertoire pour les statistiques
    stats_dir = "results/metrics/collection/language"
    os.makedirs(stats_dir, exist_ok=True)
    
    # Statistiques finales pour cette langue
    statistics_summary = f"""=== STATISTIQUES FINALES POUR {language_code} ===
Temps d'exécution: {execution_time/60:.1f} minutes
Total d'articles: {len(articles)}
Total de tokens: {total_tokens}/{token_target} ({total_tokens/token_target*100:.1f}%)
Répartition réelle des tokens:
- Catégories principales: {main_category_tokens} tokens ({main_category_tokens/total_tokens*100 if total_tokens > 0 else 0:.1f}%)
- Sous-catégories: {subcategory_tokens} tokens ({subcategory_tokens/total_tokens*100 if total_tokens > 0 else 0:.1f}%)
- Articles aléatoires: {random_tokens} tokens ({random_tokens/total_tokens*100 if total_tokens > 0 else 0:.1f}%)

Distribution dans les catégories principales:
- Articles ordonnés: {main_ordered_tokens} tokens ({main_ordered_tokens/main_category_tokens*100 if main_category_tokens > 0 else 0:.1f}%)
- Articles aléatoires: {main_random_tokens} tokens ({main_random_tokens/main_category_tokens*100 if main_category_tokens > 0 else 0:.1f}%)

Statistiques par catégorie:
"""
    # Ajouter les statistiques détaillées par catégorie
    for category in categories:
        if category in categories_stats:
            stats = categories_stats[category]
            total_cat_tokens = stats["main_tokens"] + stats["sub_tokens"]
            statistics_summary += f"- {category}: {total_cat_tokens} tokens ({total_cat_tokens/total_tokens*100 if total_tokens > 0 else 0:.1f}%)\n"
            statistics_summary += f"  * {stats['main_articles']} articles principaux ({stats['main_tokens']} tokens)\n"
            statistics_summary += f"  * {stats['sub_articles']} articles de sous-catégories ({stats['sub_tokens']} tokens)\n"
            statistics_summary += f"  * {stats['available_articles']} articles disponibles au total\n"
    
    # Ajouter la longueur moyenne des articles et le pourcentage d'articles limités
    if articles:
        avg_tokens = total_tokens / len(articles)
        token_limit = params['max_token_length']
        limited_articles = sum(1 for a in articles if a.get("token_count", 0) >= token_limit)
        limited_percentage = limited_articles / len(articles) * 100
        statistics_summary += f"\nLongueur moyenne des articles: {avg_tokens:.1f} tokens/article\n"
        statistics_summary += f"Articles limités à {token_limit} tokens: {limited_articles} ({limited_percentage:.1f}%)\n"

    # Sauvegarder les stats dans un fichier
    with open(f"{stats_dir}/{language_code}_stats.txt", "w", encoding="utf-8") as f:
        f.write(statistics_summary)

    logging.info(f"Statistiques finales pour {language_code} sauvegardées dans {stats_dir}/{language_code}_stats.txt")

    return (
        articles, 
        categories_stats, 
        total_tokens, 
        execution_time, 
        main_ordered_tokens, 
        main_random_tokens, 
        limited_articles, 
        limited_percentage
    )


def merge_with_existing_data():
    """ Fusionne les données collectées avec les données existantes """
    logging.info("Fusion des nouvelles données avec les données existantes...")
    
    temp_dir = "data/raw/temp_collection_final/intermediate_articles"
    target_dir = "data/raw/intermediate_articles"
    
    # S'assurer que le répertoire cible existe
    os.makedirs(target_dir, exist_ok=True)
    
    for language in MISSING_LANGUAGES:
        temp_file = os.path.join(temp_dir, f"{language}_articles.csv")
        target_file = os.path.join(target_dir, f"{language}_articles.csv")
        
        if not os.path.exists(temp_file):
            logging.warning(f"Pas de nouveau fichier pour {language}")
            continue
            
        # Charger les nouvelles données
        try:
            new_df = pd.read_csv(temp_file)
            logging.info(f"Nouvelles données pour {language}: {len(new_df)} articles")
            
            # Si des données existantes existent déjà, fusionner
            if os.path.exists(target_file):
                try:
                    existing_df = pd.read_csv(target_file)
                    logging.info(f"Données existantes pour {language}: {len(existing_df)} articles")
                    
                    # Fusionner en évitant les doublons (si possible sur la base du texte)
                    if "text" in new_df.columns and "text" in existing_df.columns:
                        combined_df = pd.concat([existing_df, new_df]).drop_duplicates(subset=["text"])
                    else:
                        # Sinon, fusionner simplement et espérer qu'il n'y a pas de doublons
                        combined_df = pd.concat([existing_df, new_df])
                    
                    combined_df.to_csv(target_file, index=False)
                    
                    logging.info(f"Fusion réussie pour {language}: {len(combined_df)} articles au total")
                except Exception as e:
                    logging.error(f"Erreur lors de la fusion pour {language}: {e}")
                    logging.info(f"Remplacement du fichier existant par le nouveau")
                    import shutil
                    shutil.copy2(temp_file, target_file)
            else:
                # Si pas de données existantes, copier directement
                import shutil
                shutil.copy2(temp_file, target_file)
                logging.info(f"Nouveau fichier créé pour {language}")
        except Exception as e:
            logging.error(f"Erreur lors du traitement pour {language}: {e}")
    
    logging.info("Fusion des données terminée")


def main():
    """
    Fonction principale pour exécuter la collecte adaptative avec sauvegarde progressive et reprise
    """
    logging.info("Démarrage de la fonction principale main")

    # Configuration des chemins de fichiers
    output_folder = "data/raw"
    os.makedirs(output_folder, exist_ok=True)
    
    resume_file = "resume_state.json"
    global_stats_dir = "results/metrics/collection/global"
    os.makedirs(global_stats_dir, exist_ok=True)
    global_stats_path = f"{global_stats_dir}/global_stats.csv"
    
    # Variables pour le suivi de la progression
    all_articles = []
    already_processed_languages = set()
    
    # Déterminer s'il s'agit d'une reprise ou d'un nouveau démarrage
    if os.path.exists(resume_file):
        try:
            with open(resume_file, 'r', encoding='utf-8') as f:
                resume_state = json.load(f)
                already_processed_languages = set(resume_state.get('processed_languages', []))
                start_time = resume_state.get('start_time', time.time())
                print(f"Reprise de la collecte. {len(already_processed_languages)} langues déjà traitées.")
                
                # Informations sur la dernière erreur si présente
                if 'last_error_language' in resume_state:
                    print(f"Dernière erreur sur la langue: {resume_state['last_error_language']}")
                    print(f"Message d'erreur: {resume_state.get('last_error', 'inconnu')}")
        except Exception as e:
            logging.error(f"Erreur lors du chargement de l'état de reprise: {str(e)}")
            start_time = time.time()
            resume_state = {'start_time': start_time}
    else:
        start_time = time.time()
        resume_state = {'start_time': start_time}
    
    # Charger les statistiques globales si elles existent déjà
    stats_columns = [
        "Language", "Total tokens", "Article count", "Avg tokens/article",
        "Main categories %", "Subcategories %", "Random %",
        "Execution time (min)", "Categories available", "Categories coverage %",
        "Main ordered %", "Main random %",
        "Limited articles count", "Limited articles %"
    ]
    
    # Ajouter des colonnes pour chaque catégorie
    for category in ALL_CATEGORIES:
        stats_columns.append(f"{category} tokens")
        stats_columns.append(f"{category} articles")
    
    if os.path.exists(global_stats_path):
        global_stats = pd.read_csv(global_stats_path)
        # Mettre à jour already_processed_languages avec les langues dans global_stats
        already_processed_languages.update(global_stats['Language'].values)
        logging.info(f"Statistiques globales chargées: {len(global_stats)} langues")
        
        # Charger les résumés d'articles déjà traités
        intermediate_dir = f"{output_folder}/intermediate_articles"
        if os.path.exists(intermediate_dir):
            for file in os.listdir(intermediate_dir):
                if file.endswith("_articles.csv"):
                    try:
                        lang_code = file.split("_")[0]
                        summary_path = os.path.join(intermediate_dir, file)
                        summaries_df = pd.read_csv(summary_path)
                        
                        # Créer des résumés pour chaque article
                        for _, row in summaries_df.iterrows():
                            summary = {
                                "language": row.get("language"),
                                "title": row.get("title"),
                                "category": row.get("category"),
                                "type": row.get("type", ""),
                                "token_count": row.get("token_count", 0),
                                "url": row.get("url", "")
                            }
                            all_articles.append(summary)
                        
                        logging.info(f"Résumés chargés pour {lang_code}: {len(summaries_df)} articles")
                    except Exception as e:
                        logging.error(f"Erreur lors du chargement des résumés pour {file}: {str(e)}")
    else:
        global_stats = pd.DataFrame(columns=stats_columns)
    
    max_languages = len(LANGUAGES)
    
    print(f"====== DÉMARRAGE DE LA COLLECTE SUR {max_languages} LANGUES ======")
    print(f"Catégories principales: {', '.join(ALL_CATEGORIES)}")
    print(f"Objectif en tokens: groupe A - {TARGET_TOKENS_BY_GROUP['A']}, groupe B - {TARGET_TOKENS_BY_GROUP['B']}, groupe C - {TARGET_TOKENS_BY_GROUP['C']}, groupe D - {TARGET_TOKENS_BY_GROUP['D']}")
    print(f"Limite de temps par langue: {TIME_LIMIT/60:.1f} minutes")
    print(f"Démarré à: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Langues déjà traitées: {len(already_processed_languages)}/{max_languages}")
    
    # Configuration du gestionnaire de signaux pour capture des interruptions
    def signal_handler(sig, frame):
        print("\nInterruption détectée, sauvegarde de l'état actuel...")
        # Sauvegarder l'état actuel
        with open(resume_file, 'w', encoding='utf-8') as f:
            json.dump(resume_state, f)
        # Sauvegarder les stats globales
        global_stats.to_csv(global_stats_path, index=False)
        print("État sauvegardé. Vous pourrez reprendre la collecte en relançant le script.")
        sys.exit(0)
    
    # Installer les gestionnaires de signaux
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Pour chaque langue
    for i, language in enumerate(LANGUAGES[:max_languages], 1):
        # Vérifier si la langue a déjà été traitée
        if language in already_processed_languages:
            print(f"\nLa langue {language} a déjà été traitée, passage à la suivante.")
            continue
        
        print(f"\n\n====== Langue {i}/{max_languages}: {language} ======")
        
        try:
            # Collecter des articles pour cette langue
            (
                articles, 
                cat_stats, 
                total_tokens, 
                exec_time, 
                main_ordered_tokens, 
                main_random_tokens,
                limited_articles, 
                limited_percentage
            ) = collect_articles(language, ALL_CATEGORIES)
            
            if articles:
                # 1. Sauvegarde immédiate des articles
                intermediate_dir = f"{output_folder}/intermediate_articles"
                os.makedirs(intermediate_dir, exist_ok=True)
                
                articles_df = pd.DataFrame(articles)
                articles_df.to_csv(f"{intermediate_dir}/{language}_articles.csv", index=False)
                
                # 2. Création des versions légères pour all_articles
                article_summaries = []
                for a in articles:
                    # Version légère sans le texte complet
                    summary = {
                        "language": a["language"],
                        "title": a["title"],
                        "category": a["category"],
                        "type": a.get("type", ""),
                        "token_count": a.get("token_count", 0),
                        "url": a.get("url", "")
                    }
                    article_summaries.append(summary)
                
                all_articles.extend(article_summaries)
                
                logging.info(f"Articles pour {language} sauvegardés dans {intermediate_dir}/{language}_articles.csv")
                print(f"✓ Articles pour {language} sauvegardés. Total actuel: {len(all_articles)} articles")
                
                # 3. Calcul des statistiques
                main_tokens = sum(a["token_count"] for a in articles if " (Sous-catégorie)" not in a["category"] and a["category"] != "Random")
                sub_tokens = sum(a["token_count"] for a in articles if " (Sous-catégorie)" in a["category"])
                random_tokens = sum(a["token_count"] for a in articles if a["category"] == "Random")
                
                # Calculer les pourcentages
                main_pct = main_tokens / total_tokens * 100 if total_tokens > 0 else 0
                sub_pct = sub_tokens / total_tokens * 100 if total_tokens > 0 else 0
                random_pct = random_tokens / total_tokens * 100 if total_tokens > 0 else 0
                
                # Calculer le nombre de catégories disponibles
                available_cats = sum(1 for c in ALL_CATEGORIES if c in cat_stats)
                coverage_pct = available_cats / len(ALL_CATEGORIES) * 100
                
                # Créer la ligne de stats pour la langue
                stats_row = {
                    "Language": language,
                    "Total tokens": total_tokens,
                    "Article count": len(articles),
                    "Avg tokens/article": total_tokens / len(articles) if articles else 0,
                    "Main categories %": main_pct,
                    "Subcategories %": sub_pct,
                    "Random %": random_pct,
                    "Execution time (min)": exec_time / 60,
                    "Categories available": available_cats,
                    "Categories coverage %": coverage_pct,
                    "Main ordered %": main_ordered_tokens / main_tokens * 100 if main_tokens > 0 else 0,
                    "Main random %": main_random_tokens / main_tokens * 100 if main_tokens > 0 else 0,
                    "Limited articles count": limited_articles,
                    "Limited articles %": limited_percentage
                }
                
                # Ajouter les statistiques par catégorie
                for category in ALL_CATEGORIES:
                    if category in cat_stats:
                        stats = cat_stats[category]
                        cat_tokens = stats["main_tokens"] + stats["sub_tokens"]
                        cat_articles = stats["main_articles"] + stats["sub_articles"]
                        stats_row[f"{category} tokens"] = cat_tokens
                        stats_row[f"{category} articles"] = cat_articles
                    else:
                        stats_row[f"{category} tokens"] = 0
                        stats_row[f"{category} articles"] = 0
                
                # Ajouter la ligne au DataFrame global
                global_stats = pd.concat([global_stats, pd.DataFrame([stats_row])], ignore_index=True)
                
                # Sauvegarder immédiatement les statistiques globales
                global_stats.to_csv(global_stats_path, index=False)
                
                # 4. Mettre à jour l'état de reprise
                already_processed_languages.add(language)
                resume_state['processed_languages'] = list(already_processed_languages)
                resume_state['last_update'] = time.time()
                
                with open(resume_file, 'w', encoding='utf-8') as f:
                    json.dump(resume_state, f)
                
                # 5. Libérer la mémoire
                articles = None
                
                print(f"✓ Statistiques sauvegardées pour {language}")
                
            else:
                logging.warning(f"Aucun article collecté pour {language}")
                
                # Ajouter quand même la langue à la liste des langues traitées
                already_processed_languages.add(language)
                resume_state['processed_languages'] = list(already_processed_languages)
                resume_state['last_update'] = time.time()
                
                with open(resume_file, 'w', encoding='utf-8') as f:
                    json.dump(resume_state, f)
            
        except Exception as e:
            logging.error(f"Erreur lors de la collecte pour la langue {language}: {str(e)}", exc_info=True)
            
            # Enregistrer l'erreur dans l'état de reprise
            resume_state['last_error_language'] = language
            resume_state['last_error'] = str(e)
            resume_state['last_error_time'] = time.time()
            
            with open(resume_file, 'w', encoding='utf-8') as f:
                json.dump(resume_state, f)
            
            # Sauvegarder quand même les statistiques globales
            global_stats.to_csv(global_stats_path, index=False)
    
    # Calculer le temps total d'exécution
    total_time = time.time() - start_time
    
    # Afficher un résumé global
    print("\n\n====== RÉSUMÉ DE LA COLLECTE ======")
    print(f"Temps total d'exécution: {total_time/60:.1f} minutes")
    print(f"Nombre total d'articles collectés: {len(all_articles)}")
    print(f"Nombre total de tokens: {global_stats['Total tokens'].sum()}")
    
    # Afficher un tableau résumé des résultats par langue
    print("\nRésultats par langue:")
    for _, row in global_stats.sort_values(by=['Total tokens'], ascending=False).iterrows():
        language = row['Language']
        articles_count = row['Article count']
        tokens = row['Total tokens']

        # Calculer le pourcentage par rapport à l'objectif
        target_for_lang = get_target_for_language(language)
        target_pct = tokens / target_for_lang * 100

        time_min = row['Execution time (min)']
        
        print(f"{language}: {articles_count} articles, {tokens} tokens ({target_pct:.1f}%), {time_min:.1f} min")
    
    print(f"\nStatistiques détaillées sauvegardées dans {global_stats_path}")
    
    # Indiquer que la collecte est terminée dans le fichier d'état
    resume_state['completed'] = True
    resume_state['completion_time'] = time.time()
    
    with open(resume_file, 'w', encoding='utf-8') as f:
        json.dump(resume_state, f)
    
    return global_stats, all_articles


if __name__ == "__main__":
    # Lancer la collecte
    stats, articles = main()


# Avant: 2030 lignes