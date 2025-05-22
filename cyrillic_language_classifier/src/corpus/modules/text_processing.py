# src/corpus/text_processing.py

import random
from api_utils import fetch_article_content
from cache_manager import too_short_article_ids, mark_as_too_short, is_too_short


def validate_article(page_data, language_code, category_name, min_char_length, 
                    collected_ids, api_url, sleep_time, article_type="article"):
    """
    Valide un article candidat.
    
    Cette fonction vérifie si l'article:
        - n'a pas déjà été collecté
        - est suffisamment long
    
    Args:
        page_data: dictionnaire avec 'pageid' et 'title'
        language_code: code de la langue (ex: 'ru')
        category_name: nom de la catégorie
        min_char_length: longueur minimale en caractères
        collected_ids: ensemble des IDs déjà collectés
        api_url: URL de l'API Wikipedia
        sleep_time: délai pour les requêtes API
        article_type: type d'article ("ordonné", "aléatoire", "article")
    
    Returns:
        dictionnaire de l'article si valide, None sinon
    """
    page_id = page_data['pageid']
    title = page_data['title']
    
    # Vérifier si l'article a déjà été collecté
    if page_id in collected_ids:
        return None
    
    # En récupérer le contenu
    extract, _ = fetch_article_content(api_url, page_id, sleep_time)
    
    if extract and len(extract) >= min_char_length:
        # L'article est valide -> créer le dictionnaire de base
        return {
            "language": language_code,
            "title": title,
            "text": extract,    # texte complet (sans traitement)
            "pageid": page_id,
            "url": f"https://{language_code}.wikipedia.org/?curid={page_id}",
            "category": category_name,
            "type": article_type,
            "token_count": len(extract.split())  # estimation basique du nombre de tokens
        }
    
    return None


def process_article(article, max_token_length):
    """
    Traite un article validé en limitant son texte au nombre maximum de tokens.
    
    Cette fonction prend un article déjà validé et en modifie le texte
    pour le limiter au nombre maximum de tokens spécifié.
    
    Args:
        article: dictionnaire d'article validé
        max_token_length: nombre maximum de tokens à conserver
    
    Returns:
        article modifié avec le texte traité
    """
    if not article or "text" not in article:
        return article
    
    # Traiter le texte
    processed_text, token_count = process_text(article["text"], max_token_length)
    
    # Mettre à jour l'article
    article["text"] = processed_text
    article["token_count"] = token_count
    
    return article


def process_text(text, max_tokens):
    """
    Traite le texte en le limitant à un nombre maximum de tokens,
    avec un point de départ aléatoire pour les longs textes.
    
    Args:
        text: texte à traiter
        max_tokens: nombre maximum de tokens à garder
        
    Returns:
        tuple (texte traité, nombre de tokens)
    """
    # Tokenisation simple par espaces
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
    valid_articles = []
    
    # Filtrer d'abord les candidats déjà connus comme trop courts
    filtered_candidates = [ c for c in candidates if not is_too_short(c['pageid']) ]
    already_collected_candidates = [ c for c in filtered_candidates if c['pageid'] in already_collected_ids ]
    
    print(f"  {len(candidates) - len(filtered_candidates)} article(s) déjà identifié(s) comme trop court(s) ignoré(s)")
    print(f"  {len(already_collected_candidates)} article(s) déjà collecté(s) ignoré(s)")
    print(f"  {len(filtered_candidates) - len(already_collected_candidates)} articles candidats restants à examiner")

    if len(filtered_candidates) - len(already_collected_candidates) == 0:
        print("  Tous les articles disponibles ont déjà été traités, abandon de la tentative.")
        return valid_articles

    # Déterminer l'ordre de parcours
    if article_type == "aléatoire":
        indices = list(range(len(filtered_candidates)))
        random.shuffle(indices)
        candidates_to_check = [filtered_candidates[idx] for idx in indices]
    else:
        candidates_to_check = filtered_candidates
    
    # Parcourir les candidats dans l'ordre déterminé
    for member in candidates_to_check:
        if len(valid_articles) >= num_needed:
            break
            
        # Vérifier si l'article est valide
        validated_article = validate_article(
            member, 
            language_code, 
            category_name,
            min_length, 
            already_collected_ids,
            api_url,
            sleep_time,
            article_type
        )
        
        if validated_article:
            # Article valide -> l'ajouter à la liste
            valid_articles.append(validated_article)
            print(f"  Article {article_type} valide trouvé: {validated_article['title']} ({len(validated_article['text'])} caractères)")
        else:
            # Identifier pourquoi l'article a été rejeté (pour logging et too_short_article_ids)
            page_id = member['pageid']
            title = member['title']
            
            if page_id in already_collected_ids:
                # Ne pas afficher de msg pour les articles déjà collectés pour éviter le spam
                pass
            else:
                # Si l'article n'est pas dans already_collected_ids, vérifier sa longueur
                extract, _ = fetch_article_content(api_url, page_id, sleep_time)
                if extract and len(extract) < min_length:
                    print(f"  Article {article_type} ignoré (trop court): {title}")
                    mark_as_too_short(page_id)
                elif not extract:
                    print(f"  Article {article_type} ignoré (contenu non disponible): {title}")
                else:
                    print(f"  Article {article_type} ignoré (pas valide): {title}")
    
    return valid_articles