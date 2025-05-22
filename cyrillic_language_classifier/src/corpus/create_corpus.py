# Script de collecte avec stratégie adaptative par groupe de langues
import os
import time
import logging
import pandas as pd
from datetime import datetime

# Import des modules refactorisés
from corpus.modules import (
    # Config
    TIME_LIMIT, LANGUAGES, ALL_CATEGORIES, MAX_DEPTHS_BY_GROUP, CATEGORY_TRANSLATIONS,
    get_adaptive_params, get_language_group, get_target_for_language,  
    # Classes
    ArticleCollector, CollectionStats,
    # Fonctions
    process_text, save_articles_to_csv, calculate_token_targets, 
    print_collection_plan, save_global_stats
)


# Définir les chemins de fichiers de base
BASE_DIR = "data"
ARTICLES_DIR = f"{BASE_DIR}/raw"

# Chemins pour les statistiques
METRICS_DIR = "results/metrics/corpus_analysis/collection"
LANGUAGES_STATS_DIR = f"{METRICS_DIR}/languages"
GLOBAL_STATS_DIR = f"{METRICS_DIR}/global"

# Créer les dossiers s'ils n'existent pas
os.makedirs(ARTICLES_DIR, exist_ok=True)
os.makedirs(LANGUAGES_STATS_DIR, exist_ok=True)
os.makedirs(GLOBAL_STATS_DIR, exist_ok=True)

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


def collect_articles(language_code, categories):
    """
        Fonction de collecte avec adaptation par groupe de langue.
    """
    logging.info(f"Démarrage de la collecte pour la langue: {language_code}")

    # Récupérer les informations de base
    group = get_language_group(language_code)
    token_target = get_target_for_language(language_code)
    params = get_adaptive_params(language_code)
    
    # Déterminer les catégories disponibles pour la langue en cours
    available_categories = []
    for category in categories:
        if language_code in CATEGORY_TRANSLATIONS[category]:
            available_categories.append(category)

    print(f"\nCatégories disponibles ({len(available_categories)}/{len(categories)}): {', '.join(available_categories)}")

    if not available_categories:
        logging.warning(f"Attention: aucune catégorie disponible pour {language_code}")
        return []
    
    # Initialiser les stats
    stats = CollectionStats(language_code, token_target, categories)
    
    # Calculer les objectifs
    targets = calculate_token_targets(token_target, params, available_categories)
    main_target, sub_target, random_target, tokens_per_main, tokens_per_sub = targets
    
    # Définir tous les objectifs dans la classe de stats
    stats.set_token_targets(main_target, sub_target, random_target, 
                           tokens_per_main, tokens_per_sub)
    
    # Afficher le plan de collecte
    print_collection_plan(language_code, group, token_target, params, targets)
    
    # Variables de travail
    start_time = time.time()
    articles = []
    collected_article_ids = set()
    random_tokens = 0

    # Créer une instance du collecteur d'articles
    collector = ArticleCollector(
        language_code=language_code,
        categories=categories,
        params=params,
        already_collected_ids=collected_article_ids
    )

    # 1. Collecter des articles des catégories principales
    print(f"\n1. Collecte d'articles des catégories principales (objectif: {stats.main_category_token_target} tokens)")
    
    for category in available_categories:
        # Vérifier si on a dépassé la limite de temps
        if time.time() - start_time > TIME_LIMIT:
            print(f"Limite de temps atteinte pour {language_code}. Passage à la langue suivante.")
            break
            
        translated_category = CATEGORY_TRANSLATIONS[category][language_code]
        print(f"\n  Catégorie: {category} ({translated_category})")
        
        # Adapter le batch_size en fonction des besoins
        category_target = tokens_per_main[category]
        category_tokens = 0
        attempts = 0
        max_attempts = 15    # nombre max de tentatives
        
        while category_tokens < category_target and attempts < max_attempts:
            attempts += 1
            # Calculer le batch_size en fonction des tokens restants
            remaining_tokens = category_target - category_tokens

            if group in ['A', 'B']:
                batch_size = max(5, min(15, remaining_tokens // 200))  # plus petit batch initial
            else:
                batch_size = max(10, min(30, remaining_tokens // 100)) # plus grand pour les groupes C & D
            
            print(f"  Tentative {attempts}: recherche de {batch_size} articles (objectif: {remaining_tokens} tokens manquants)")
            
            try:
                category_articles = collector.collect_by_category(
                    category_name=translated_category,
                    category_target=category_target,
                    num_articles=batch_size,
                    fixed_ratio=params['fixed_selection_ratio'],
                    sleep_time=(1, 2.5)
                )
                
                # Enregistrer le nombre d'articles disponibles
                if category_articles:  # s'assurer qu'il y a des articles
                    stats.set_available_articles(category, len(category_articles))
                
                # Si pas d'articles disponibles, passer à la catégorie suivante
                if not category_articles:
                    print(f"  Aucun article disponible pour {category} ({translated_category})")
                    break
                
                # Ajouter les articles jusqu'à atteindre l'objectif pour cette catégorie
                for article in category_articles:
                    articles.append(article)
                    
                    # Mettre à jour les statistiques
                    stats.update_main_category_stats(category, article, article.get("type"))

                    # Mettre à jour le compteur local pour cette catégorie
                    category_tokens += article.get("token_count", 0)

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
    stats.print_progress_summary("les catégories principales", stats.main_category_tokens, stats.main_category_token_target)
    
    if time.time() - start_time > TIME_LIMIT:
        print(f"Limite de temps atteinte pour {language_code}. Collecte des sous-catégories et articles aléatoires annulée.")
        # Même en cas de dépassement de temps, continuer pour collecter des statistiques partielles
        pass
    else:
        # 2. Collecter des articles des sous-catégories
        print(f"\n2. Collecte d'articles des sous-catégories (objectif: {stats.subcategory_token_target} tokens)")

        # Pour chaque catégorie
        subcategories_cache = {}  # dictionnaire pour stocker les sous-catégories par catégorie

        for category in available_categories:
            if time.time() - start_time > TIME_LIMIT:
                print(f"Limite de temps atteinte pendant la collecte des sous-catégories.")
                break
                
            translated_category = CATEGORY_TRANSLATIONS[category][language_code]
            print(f"\n  Sous-catégories de: {category} ({translated_category})")
            
            # Logique de collecte pour chaque catégorie
            category_target = sub_target
            category_tokens = 0
            attempts = 0
            max_attempts = 15

            # Récupérer le cache des sous-catégories si disponible
            cached_subcats = subcategories_cache.get(category, None)
            
            while category_tokens < category_target and attempts < max_attempts:
                attempts += 1
                remaining_tokens = category_target - category_tokens

                if group in ['A', 'B']:
                    batch_size = max(5, min(15, remaining_tokens // 200))
                else:
                    batch_size = max(10, min(30, remaining_tokens // 100))
                
                print(f"  Tentative {attempts}: recherche de {batch_size} articles")
                
                try:
                    # Récupérer la profondeur adaptée au groupe de langue
                    max_depth = MAX_DEPTHS_BY_GROUP[group]

                    subcategory_articles, updated_cache = collector.collect_from_subcategories(
                        category_name=translated_category,
                        num_articles=batch_size,
                        max_depth=max_depth,      # profondeur adaptée au groupe
                        sleep_time=(0.5, 1.5),
                        cached_subcategories=cached_subcats, # passer le cache
                        attempt_number=attempts
                    )

                    # Mettre à jour le cache pour les tentatives suivantes
                    if updated_cache:
                        subcategories_cache[category] = updated_cache
                        cached_subcats = updated_cache
                    
                    # Ajouter les articles collectés
                    for article in subcategory_articles:
                        # Traiter et limiter le texte
                        article_text, token_count = process_text(article["text"], params['max_token_length'])
                        article["text"] = article_text
                        article["token_count"] = token_count
                        
                        # Ajouter l'article et mettre à jour les compteurs
                        articles.append(article)
                        
                        # Mettre à jour les statistiques
                        stats.update_subcategory_stats(category, article)
                        category_tokens += token_count
                        
                        print(f"  Article ajouté: {article['title']} ({token_count} tokens)")
                        
                        # Si on a atteint l'objectif pour cette catégorie, arrêter
                        if category_tokens >= category_target:
                            break
                    
                    print(f"  → Sous-catégories de {category}: {category_tokens} tokens collectés sur {category_target} ciblés")
                    
                except Exception as e:
                    logging.error(f"  Erreur lors de la collecte des sous-catégories pour {category}: {str(e)}", exc_info=True)
                    break

        # Afficher le résumé de la collecte des sous-catégories
        stats.print_progress_summary("les sous-catégories", stats.subcategory_tokens, stats.subcategory_token_target)
        
        if time.time() - start_time > TIME_LIMIT:
            print(f"Limite de temps atteinte pour {language_code}. Collecte des articles aléatoires annulée.")
        else:
            # 3. Collecter des articles aléatoires
            print(f"\n3. Collecte d'articles aléatoires (objectif: {stats.random_token_target} tokens)")
            
            try:
                random_tokens = 0  # initialiser le compteur de tokens aléatoires

                while random_tokens < stats.random_token_target:
                    if time.time() - start_time > TIME_LIMIT:
                        print(f"Limite de temps atteinte pendant la collecte des articles aléatoires.")
                        break
                    
                    # Calculer combien d'articles collecter pour cette itération
                    remaining_tokens = stats.random_token_target - random_tokens
                    batch_size = max(1, min(5, remaining_tokens // 500))
                    
                    random_articles = collector.collect_random(
                        num_articles=batch_size,
                        sleep_time=(0.5, 2)
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
                        stats.update_random_stats(article)
                        random_tokens += token_count
                        
                        print(f"  Article aléatoire ajouté: {article['title']} ({token_count} tokens)")
                        
                        # Vérifier si on a atteint l'objectif
                        if random_tokens >= stats.random_token_target:
                            break

                        if time.time() - start_time > TIME_LIMIT:
                            print(f"Limite de temps atteinte pendant la collecte des articles aléatoires.")
                            break
                    
                    stats.print_progress_summary("les articles aléatoires", stats.random_tokens, stats.random_token_target)
                    
            except Exception as e:
                logging.error(f"  Erreur lors de la collecte des articles aléatoires: {str(e)}", exc_info=True)
    
    total_tokens = stats.total_tokens

    logging.info(f"Collecte terminée pour {language_code}: {len(articles)} articles, {total_tokens} tokens")

    # Calculer le temps d'exécution total
    execution_time = time.time() - start_time

    # Générer et sauvegarder les statistiques finales pour cette langue
    stats.save_to_file(execution_time, params['max_token_length'], output_dir=LANGUAGES_STATS_DIR)

    logging.info(f"Statistiques finales pour {language_code} sauvegardées dans {LANGUAGES_STATS_DIR}/{language_code}_stats.txt")

    return (
        articles, 
        stats.categories_stats,
        stats.total_tokens,
        execution_time, 
        stats.main_ordered_tokens,
        stats.main_random_tokens,
        stats.limited_articles,
        stats.limited_articles / stats.articles_count * 100 if stats.articles_count > 0 else 0  # calculer le %
    )


def main():
    """
        Fonction principale pour exécuter la collecte adaptative avec sauvegarde progressive et reprise
    """
    logging.info("Démarrage de la fonction principale main")

    output_folder = ARTICLES_DIR
    
    # Variables pour le suivi de la progression
    all_articles = []
    already_processed_languages = set()

    max_languages = len(LANGUAGES)
    
    print(f"====== DÉMARRAGE DE LA COLLECTE SUR {max_languages} LANGUES ======")
    
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
                # Ajouter les articles à la liste globale
                all_articles.extend(articles)
                
                # Sauvegarder les articles dans un fichier CSV
                save_articles_to_csv(language, articles, output_folder)
                
                print(f"✓ {len(articles)} articles collectés pour {language}, {total_tokens} tokens")
            else:
                print(f"! Aucun article collecté pour {language}")
            
            # Marquer la langue comme traitée
            already_processed_languages.add(language)
            
        except Exception as e:
            logging.error(f"Erreur lors de la collecte pour la langue {language}: {str(e)}", exc_info=True)
    
    print(f"\n====== COLLECTE TERMINÉE ======")
    print(f"Total: {len(all_articles)} articles collectés pour {len(already_processed_languages)} langues")

    global_stats_path = f"{GLOBAL_STATS_DIR}/global_stats.csv"
    save_global_stats(already_processed_languages, all_articles, global_stats_path)
    
    return all_articles


if __name__ == "__main__":
    # Lancer la collecte
    articles = main()
    print(f"Collecte terminée : {len(articles)} articles au total")