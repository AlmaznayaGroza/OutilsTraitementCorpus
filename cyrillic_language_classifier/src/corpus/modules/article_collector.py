import random
import logging

from config import (
    CATEGORY_PREFIXES,
    get_adaptive_params
)
from api_utils import (
    fetch_category_articles,
    fetch_subcategories,
    fetch_random_article
)
from text_processing import (
    select_valid_articles,
    process_article,
    validate_article
)
from cache_manager import too_short_article_ids, is_too_short, mark_as_too_short

class ArticleCollector:
    def __init__(self, language_code, categories, params=None, already_collected_ids=None):
        """
        Initialise un collecteur d'articles pour une langue donnée.
        
        Args:
            language_code: code de la langue (ex: 'ru', 'uk')
            categories: liste des catégories à explorer
            params: paramètres adaptatifs (optionnel)
            already_collected_ids: ensemble des IDs d'articles déjà collectés (optionnel)
        """
        self.language_code = language_code
        self.categories = categories
        self.params = params or get_adaptive_params(language_code)
        self.already_collected_ids = already_collected_ids or set()
        self.api_url = f"https://{language_code}.wikipedia.org/w/api.php"
        
    def collect_by_category(self, category_name, category_target, num_articles=20, 
                                fixed_ratio=0.6, sleep_time=(1, 3)):
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
            params: dictionnaire des paramètres adaptatifs (longueur min, max tokens, etc.)
        """
        logging.info(f"Récupération d'articles par catégorie pour {self.language_code}, catégorie: {category_name}")
        
        articles = []
        
        # Obtenir le préfixe adéquat
        prefix = CATEGORY_PREFIXES.get(self.language_code, 'Category:')
        full_category = f"{prefix}{category_name}"
        
        try:
            # === RÉCUPÉRATION D'ARTICLES DANS L'ORDRE ===
            ordered_members = fetch_category_articles(
                self.api_url, 
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
                self.api_url,
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
                self.already_collected_ids,
                self.params['min_char_length'],
                self.language_code,
                category_name,
                sleep_time,
                self.api_url,
                "ordonné"
            )

            # Si aucun article ordonné n'est disponible ET qu'il n'y a pas d'articles aléatoires à collecter
            if not fixed_articles and num_random_selection == 0:
                return [], 0, 0, 0       # retourne des résultats vides

            # Ajouter les articles ordonnés jusqu'à atteindre l'objectif
            articles_to_add = []
            for article in fixed_articles:
                processed_article = process_article(article, self.params['max_token_length'])
                # Ajouter l'article et mettre à jour les compteurs
                articles_to_add.append(processed_article)
                ordered_tokens_collected += processed_article["token_count"]
                category_tokens += processed_article["token_count"]
                
                print(f"  Article ajouté: {processed_article['title']} ({processed_article['token_count']} tokens)")
                
                # Si on a atteint l'objectif pour les articles ordonnés, arrêter
                if ordered_tokens_collected >= ordered_tokens_target:
                    break

            # Mettre à jour les IDs déjà collectés
            for article in articles_to_add:
                self.already_collected_ids.add(article["pageid"])

            used_ids = { article['pageid'] for article in articles_to_add }

            # Si on n'a pas atteint l'objectif avec les articles ordonnés,
            # augmenter l'objectif pour les articles aléatoires
            remaining_tokens = category_target - ordered_tokens_collected
            if remaining_tokens > random_tokens_target:
                random_tokens_target = remaining_tokens

            # === COLLECTE ET TRAITEMENT DES ARTICLES ALÉATOIRES ===
            filtered_random_members = [ m for m in random_members if m['pageid'] not in used_ids ]

            random_articles = select_valid_articles(
                filtered_random_members,
                num_random_selection,
                self.already_collected_ids,
                self.params['min_char_length'],
                self.language_code,
                category_name,
                sleep_time,
                self.api_url,
                "aléatoire"
            )

            # Si aucun article aléatoire n'est disponible ET qu'il n'y a pas d'articles ordonnés collectés
            if not random_articles and not articles_to_add:
                return [], 0, 0, 0

            # Ajouter les articles aléatoires jusqu'à atteindre l'objectif
            for article in random_articles:
                if random_tokens_collected >= random_tokens_target:
                    break
                    
                processed_article = process_article(article, self.params['max_token_length'])
        
                # Ajouter l'article et mettre à jour les compteurs
                articles_to_add.append(processed_article)
                random_tokens_collected += processed_article["token_count"]
                category_tokens += processed_article["token_count"]
                self.already_collected_ids.add(processed_article["pageid"])
                
                print(f"    + Article: {processed_article['title']} ({processed_article['token_count']} tokens)")

            # Résumé des résultats
            print(f"    Détail: {ordered_tokens_collected} tokens ordonnés + {random_tokens_collected} tokens aléatoires")
            print(f"    Batch: {category_tokens}/{category_target} tokens collectés ({len(articles_to_add)} articles)")

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
        
        logging.info(f"Fin de la récupération pour {self.language_code}, catégorie: {category_name}, {len(articles)} articles trouvés")

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
        logging.info(f"Distribution pour {category_name} (langue: {self.language_code}):")
        logging.info(f"- Articles: {ordered_articles_ratio:.2f} ordonnés vs {random_articles_ratio:.2f} aléatoires")
        logging.info(f"- Tokens: {ordered_tokens_ratio:.2f} ordonnés vs {random_tokens_ratio:.2f} aléatoires")
        logging.info(f"- Objectif initial: {fixed_ratio:.2f} ordonnés vs {1-fixed_ratio:.2f} aléatoires")

        return articles_to_add, category_tokens, ordered_tokens_collected, random_tokens_collected

        
    def collect_from_subcategories(self, category_name, num_articles=10, max_depth=5,
                                   sleep_time=(1, 3), cached_subcategories=None,
                                   attempt_number=1, token_target=None):
        """
        Récupère des articles aléatoires dans les sous-catégories d'une catégorie donnée.
        
        Args:
            category_name: nom de la catégorie principale
            num_articles: nombre d'articles à récupérer
            max_depth: profondeur max de recherche dans les sous-catégories
            sleep_time: intervalle d'attente entre les requêtes
            cached_subcategories: ensemble des sous-catégories déjà trouvées
            attempt_number: numéro de tentative actuelle
            token_target: objectif de tokens à atteindre
            
        Returns:
            tuple (liste d'articles, cache des sous-catégories mis à jour, total de tokens collectés)
        """
        logging.info(f"Recherche dans les sous-catégories pour {self.language_code}, catégorie: {category_name}")

        articles = []
        total_tokens = 0     # compteur de tokens pour cette collecte
        
        prefix = CATEGORY_PREFIXES.get(self.language_code, 'Category:')
        full_category = f"{prefix}{category_name}"
        
        # Initialiser le cache des sous-catégories si non fourni
        if cached_subcategories is None:
            # Ensemble pour stocker toutes les sous-catégories trouvées
            all_subcategories = set()
            # File d'attente pour l'exploration en largeur des sous-catégories
            subcategory_queue = [ (full_category, 0) ]  # (catégorie, profondeur)
        
            # Explorer les sous-catégories en largeur d'abord (BFS)
            while subcategory_queue and len(all_subcategories) < 800:
                current_category, depth = subcategory_queue.pop(0)
                
                # Si on a déjà exploré cette catégorie ou si on a atteint la profondeur max
                if current_category in all_subcategories or depth > max_depth:
                    continue
                    
                all_subcategories.add(current_category)
                
                # Si on n'est pas à la profondeur maximale, chercher les sous-catégories
                if depth <= max_depth:
                    subcats = fetch_subcategories(self.api_url, current_category, 50, sleep_time)
                    
                    for subcat in subcats:
                        if 'title' in subcat:
                            subcategory_queue.append((subcat['title'], depth + 1))
            
            print(f"    {len(all_subcategories)} sous-catégories trouvée(s) pour '{category_name}'.")

            # Retourner le cache des sous-catégories pour les tentatives futures
            cached_subcategories = all_subcategories

        else:
            all_subcategories = cached_subcategories
            print(f"    Utilisation de {len(all_subcategories)} sous-catégories déjà explorées pour '{category_name}'.")
        
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
            
            while new_queue and len(new_explored) < 300:  # limiter le nb de nouvelles sous-catégories
                current_cat, depth = new_queue.pop(0)
                
                # Si déjà à la profondeur maximale pour cette exploration secondaire, passer
                if depth > secondary_depth:
                    continue
                
                new_subcats = fetch_subcategories(self.api_url, current_cat, 50, sleep_time)
                        
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
            cached_subcategories = all_subcategories  # mise à jour du cache
        
        # Après l'exploration secondaire et l'enrichissement de all_subcategories
        # Si aucune sous-catégorie n'est trouvée, retourner une liste vide
        if not all_subcategories:
            return [], None, 0
            
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
                
            members = fetch_category_articles(self.api_url, subcat, 50, sleep_time=sleep_time)
                    
            # S'il y a des articles valides
            if members:
                sample_size = min(len(members), 5)
                selected_members = random.sample(members, sample_size)
                
                for member in selected_members:
                    # Valider l'article
                    validated_article = validate_article(
                        member,
                        self.language_code,
                        f"{category_name} (Sous-catégorie)",
                        self.params['min_char_length'],
                        self.already_collected_ids,
                        self.api_url,
                        sleep_time
                    )
                    
                    if validated_article:
                        # Traiter l'article pour limiter les tokens
                        processed_article = process_article(validated_article, self.params['max_token_length'])
                        articles.append(processed_article)
                        self.already_collected_ids.add(processed_article["pageid"])
                        articles_collected_count += 1
                        
                        # Mettre à jour le compteur de tokens
                        total_tokens += processed_article["token_count"]
                        
                        print(f"    + Article: {processed_article['title']} ({processed_article['token_count']} tokens)")

                        # Vérifier si on a atteint l'objectif de tokens (si fourni)
                        if token_target and total_tokens >= token_target:
                            print(f"    Objectif de tokens atteint ({total_tokens}/{token_target}), arrêt de la collecte.")
                            return articles, cached_subcategories, total_tokens

                    else:
                        # Si l'article est invalide, vérifier pourquoi
                        page_id = member['pageid']
                        title = member['title']
                        
                        if page_id not in self.already_collected_ids:
                            print(f"    - Article ignoré (trop court): {title}")
        
        # Vérifier si l'objectif de tokens a été atteint avant d'explorer plus de sous-catégories
        if token_target and total_tokens >= token_target:
            print(f"    Objectif de tokens déjà atteint ({total_tokens}/{token_target}), pas d'exploration supplémentaire.")
            return articles, cached_subcategories, total_tokens

        # Si on n'a pas atteint l'objectif, essayer plus de sous-catégories
        if articles_collected_count < num_articles and subcats_explored < len(selected_subcats):
            print(f"  Objectif non atteint, exploration de sous-catégories supplémentaires...")
            additional_subcats = selected_subcats[subcats_explored:]

            empty_subcats_count = 0
            max_empty_subcats = 10     # arrêter après 10 sous-catégories vides consécutives
            
            for subcat in additional_subcats[:20]:  # essayer jusqu'à 20 sous-catégories supplémentaires
                if articles_collected_count >= num_articles:
                    break
                    
                subcats_explored += 1
                
                members = fetch_category_articles(self.api_url, subcat, 50, sleep_time=sleep_time)
                        
                # S'il y a des articles valides
                if members:
                    # Sélectionner aléatoirement quelques articles
                    sample_size = min(len(members), 5)
                    selected_members = random.sample(members, sample_size)
                    
                    for member in selected_members:
                        # Valider l'article
                        validated_article = validate_article(
                            member,
                            self.language_code,
                            f"{category_name} (Sous-catégorie)",
                            self.params['min_char_length'],
                            self.already_collected_ids,
                            self.api_url,
                            sleep_time
                        )
                        
                        if validated_article:
                            # Traiter l'article pour limiter les tokens
                            processed_article = process_article(validated_article, self.params['max_token_length'])
                            articles.append(processed_article)
                            self.already_collected_ids.add(processed_article["pageid"])
                            articles_collected_count += 1
                            
                            # Mettre à jour le compteur de tokens
                            total_tokens += processed_article["token_count"]
                            
                            print(f"    + Article: {processed_article['title']} ({processed_article['token_count']} tokens)")

                            if token_target and total_tokens >= token_target:
                                print(f"    Objectif de tokens atteint ({total_tokens}/{token_target}) - arrêt de la collecte.")
                                return articles, cached_subcategories, total_tokens

                        else:
                            # Si l'article est invalide, vérifier pourquoi
                            page_id = member['pageid']
                            title = member['title']
                            
                            if page_id not in self.already_collected_ids:
                                print(f"    - Article ignoré (trop court): {title}")
                
                if not members:  # si la sous-catégorie est vide
                    empty_subcats_count += 1
                    if empty_subcats_count >= max_empty_subcats:
                        print(f"  Trop de sous-catégories vides consécutives, arrêt de la recherche supplémentaire.")
                        break
                else:
                    empty_subcats_count = 0  # réinitialiser le compteur
            
            if articles_collected_count < num_articles:
                print(f"  Terminé avec {articles_collected_count} articles sur {num_articles} objectif")

        print(f"    Batch: {articles_collected_count} articles collectés, {total_tokens} tokens")
        logging.info(f"Récupération terminée, {len(articles)} articles trouvés dans les sous-catégories")

        return articles, cached_subcategories, total_tokens

        
    def collect_random(self, num_articles=20, sleep_time=(1, 3)):
        """
        Récupère aléatoirement des articles de Wikipédia dans la langue spécifiée.
        
        Args:
            num_articles: nb d'articles à récupérer
            sleep_time: tuple (min, max) pour le temps d'attente entre les requêtes
            
        Returns:
            tuple (liste d'articles, total des tokens)
        """
        logging.info(f"Récupération d'articles aléatoires pour {self.language_code}, objectif: {num_articles} articles")
  
        articles = []
        total_tokens = 0     # compteur pour le total des tokens
        
        # Compteur pour éviter les boucles infinies
        articles_collected = 0
        attempts = 0
        max_attempts = num_articles * 15  # limite pour éviter les boucles infinies
        
        # Continuer jusqu'à obtenir le nombre d'articles demandé ou atteindre le max de tentatives
        while articles_collected < num_articles and attempts < max_attempts:
            attempts += 1
            
            # 1. Obtenir un article aléatoire
            random_article = fetch_random_article(self.api_url, sleep_time)

            if random_article:
                # Valider l'article
                validated_article = validate_article(
                    random_article,
                    self.language_code,
                    "Random",
                    self.params['min_char_length'],
                    self.already_collected_ids,
                    self.api_url,
                    sleep_time
                )
                
                if validated_article:
                    # Traiter l'article pour limiter les tokens
                    processed_article = process_article(validated_article, self.params['max_token_length'])
                    articles.append(processed_article)
                    self.already_collected_ids.add(processed_article["pageid"])
                    
                    # Mettre à jour les compteurs
                    articles_collected += 1
                    total_tokens += processed_article["token_count"]

                    print(f"    + Article aléatoire: {processed_article['title']} ({processed_article['token_count']} tokens)")
        
        if articles_collected < num_articles:
            logging.warning(f"Attention: seulement {articles_collected}/{num_articles} articles aléatoires trouvés pour {self.language_code}")
        
        print(f"    Batch: {articles_collected} articles aléatoires collectés, {total_tokens} tokens")
        logging.info(f"Récupération d'articles aléatoires terminée pour {self.language_code}: {articles_collected} articles trouvés")

        return articles, total_tokens