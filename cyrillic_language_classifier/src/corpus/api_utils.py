# src/corpus/api_utils.py

import requests
import time
import random
import logging


def fetch_subcategories(api_url, category_title, limit=50, sleep_time=(0.5, 1.5)):
    """
    Récupère la liste des sous-catégories d'une catégorie donnée.
    
    Cette fonction encapsule la logique répétitive pour récupérer les sous-catégories
    via l'API Wikipedia. Elle gère automatiquement les erreurs et les délais.
    
    Args:
        api_url: URL de l'API Wikipedia (ex: "https://ru.wikipedia.org/w/api.php" pour le russe)
        category_title: titre complet de la catégorie (ex: "Категория:Культура")
        limit: nombre maximum de sous-catégories à récupérer
        sleep_time: tuple (min, max) pour le délai aléatoire entre requêtes
    
    Returns:
        liste des sous-catégories trouvées, ou liste vide en cas d'erreur
    """
    params = {
        "action": "query",
        "format": "json",
        "list": "categorymembers",
        "cmtitle": category_title,
        "cmlimit": str(limit),
        "cmtype": "subcat"
    }
    
    # Respecter les délais pour éviter de surcharger l'API
    time.sleep(random.uniform(*sleep_time))
    
    try:
        response = requests.get(api_url, params=params).json()
        if 'query' in response and 'categorymembers' in response['query']:
            return response['query']['categorymembers']
        else:
            # Cas où la réponse est valide mais ne contient pas de sous-catégories
            return []
    except Exception as e:
        logging.error(f"Erreur lors de la récupération des sous-catégories de {category_title}: {e}")
        return []


def fetch_category_articles(api_url, category_title, limit=50, sort_method="sortkey", 
                          sort_direction="asc", sleep_time=(0.5, 1.5)):
    """
    Récupère la liste des articles d'une catégorie donnée.
    
    Cette fonction standardise la récupération d'articles d'une catégorie avec
    différentes options de tri et de limite.
    
    Args:
        api_url: URL de l'API Wikipedia
        category_title: titre complet de la catégorie
        limit: nombre max d'articles à récupérer
        sort_method: méthode de tri ("sortkey", "timestamp")
        sort_direction: direction du tri ("asc", "desc")
        sleep_time: tuple (min, max) pour le délai aléatoire
    
    Returns:
        liste des articles trouvés, ou liste vide en cas d'erreur
    """
    params = {
        "action": "query",
        "format": "json",
        "list": "categorymembers",
        "cmtitle": category_title,
        "cmlimit": str(limit),
        "cmtype": "page",
        "cmsort": sort_method,
        "cmdir": sort_direction
    }
    
    time.sleep(random.uniform(*sleep_time))
    
    try:
        response = requests.get(api_url, params=params).json()
        if 'query' in response and 'categorymembers' in response['query']:
            return response['query']['categorymembers']
        else:
            return []
    except Exception as e:
        logging.error(f"Erreur lors de la récupération des articles de {category_title}: {e}")
        return []


def fetch_article_content(api_url, page_id, sleep_time=(0.5, 1.5)):
    """
    Récupère le contenu textuel d'un article Wikipédia.
    
    Cette fonction encapsule la logique pour récupérer le texte d'un article
    spécifique par son ID.
    
    Args:
        api_url: URL de l'API Wikipedia
        page_id: ID de la page Wikipedia
        sleep_time: tuple (min, max) pour le délai aléatoire
    
    Returns:
        Tuple (texte_de_l_article, titre) si succès, (None, None) sinon
    """
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": "1",
        "pageids": str(page_id)
    }
    
    time.sleep(random.uniform(*sleep_time))
    
    try:
        response = requests.get(api_url, params=params).json()
        if 'query' in response and 'pages' in response['query']:
            page_data = response['query']['pages'][str(page_id)]
            if 'extract' in page_data:
                # Récupérer aussi le titre si disponible
                title = page_data.get('title', f'Article_{page_id}')
                return page_data['extract'], title
    except Exception as e:
        logging.error(f"Erreur lors de la récupération du contenu pour l'article {page_id}: {e}")
    
    return None, None


def fetch_random_article(api_url, sleep_time=(0.5, 1.5)):
    """
    Récupère un article aléatoire depuis Wikipédia.
    
    Cette fonction encapsule la logique pour obtenir un article aléatoire
    dans l'espace de noms principal (articles).
    
    Args:
        api_url: URL de l'API Wikipedia
        sleep_time: tuple (min, max) pour le délai aléatoire
    
    Returns:
        Dictionnaire avec 'id' et 'title' si succès, None sinon
    """
    params = {
        "action": "query",
        "format": "json",
        "list": "random",
        "rnnamespace": "0",  # articles principaux uniquement
        "rnlimit": "1"
    }
    
    time.sleep(random.uniform(*sleep_time))
    
    try:
        response = requests.get(api_url, params=params).json()
        if 'query' in response and 'random' in response['query']:
            random_article = response['query']['random'][0]
            return {
                'id': random_article['id'],
                'title': random_article['title']
            }
    except Exception as e:
        logging.error(f"Erreur lors de la récupération d'un article aléatoire: {e}")
    
    return None