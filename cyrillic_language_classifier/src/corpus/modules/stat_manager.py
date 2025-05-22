# src/corpus/stat_manager.py

import os
import logging
import pandas as pd
from datetime import datetime


class CollectionStats:
    """
    Gestionnaire des statistiques de collecte d'articles.
    
    Cette classe encapsule toute la logique liée à la collecte et
    la génération de statistiques sur les articles collectés.
    """
    
    def __init__(self, language_code, token_target, categories):
        """
        Initialise le gestionnaire de statistiques.
        
        Args:
            language_code: code de la langue traitée
            token_target: objectif de tokens pour cette langue
            categories: liste des catégories disponibles
        """
        self.language_code = language_code
        self.token_target = token_target

        # Objectifs par type de collecte
        self.main_category_token_target = 0
        self.subcategory_token_target = 0 
        self.random_token_target = 0
        
        # Objectifs par catégorie
        self.tokens_per_main_category = {}
        self.tokens_per_subcategory = {}
        
        # Compteurs globaux
        self.total_tokens = 0
        self.main_category_tokens = 0
        self.subcategory_tokens = 0
        self.random_tokens = 0
        self.main_ordered_tokens = 0
        self.main_random_tokens = 0
        
        # Initialiser les statistiques par catégorie
        self.categories_stats = {}
        for category in categories:
            self.categories_stats[category] = {
                "main_articles": 0,
                "main_tokens": 0,
                "sub_articles": 0,
                "sub_tokens": 0,
                "available_articles": 0
            }
        
        # Informations sur les articles
        self.articles_count = 0
        self.limited_articles = 0
        self.start_time = datetime.now()

    def set_token_targets(self, main_target, sub_target, random_target, 
                         tokens_per_main, tokens_per_sub):
        """
        Définit tous les objectifs de tokens en 1 fois.
        
        Args:
            main_target: objectif pour les catégories principales
            sub_target: objectif pour les sous-catégories
            random_target: objectif pour les articles aléatoires
            tokens_per_main: tokens par catégorie principale
            tokens_per_sub: tokens par sous-catégorie
        """
        self.main_category_token_target = main_target
        self.subcategory_token_target = sub_target
        self.random_token_target = random_target
        
        # Objectifs par catégorie
        for category in self.categories_stats:
            self.tokens_per_main_category[category] = tokens_per_main
            self.tokens_per_subcategory[category] = tokens_per_sub
    
    def update_main_category_stats(self, category, article, article_type):
        """
        Met à jour les statistiques pour un article de catégorie principale.
        
        Args:
            category: nom de la catégorie
            article: dictionnaire de l'article
            article_type: "ordonné" ou "aléatoire"
        """
        token_count = article.get("token_count", 0)
        
        # Mettre à jour les compteurs globaux
        self.main_category_tokens += token_count
        self.total_tokens += token_count
        
        # Mettre à jour les statistiques par catégorie
        if category in self.categories_stats:
            self.categories_stats[category]["main_articles"] += 1
            self.categories_stats[category]["main_tokens"] += token_count
        
        # Mettre à jour les compteurs par type
        if article_type == "ordonné":
            self.main_ordered_tokens += token_count
        elif article_type == "aléatoire":
            self.main_random_tokens += token_count
        
        self.articles_count += 1
    
    def update_subcategory_stats(self, category, article):
        """
        Met à jour les statistiques pour un article de sous-catégorie.
        
        Args:
            category: nom de la catégorie
            article: dictionnaire de l'article
        """
        token_count = article.get("token_count", 0)
        
        # Mettre à jour les compteurs globaux
        self.subcategory_tokens += token_count
        self.total_tokens += token_count
        
        # Mettre à jour les statistiques par catégorie
        if category in self.categories_stats:
            self.categories_stats[category]["sub_articles"] += 1
            self.categories_stats[category]["sub_tokens"] += token_count
        
        self.articles_count += 1
    
    def update_random_stats(self, article):
        """
        Met à jour les statistiques pour un article aléatoire.
        
        Args:
            article: dictionnaire de l'article
        """
        token_count = article.get("token_count", 0)
        
        # Mettre à jour les compteurs globaux
        self.random_tokens += token_count
        self.total_tokens += token_count
        
        self.articles_count += 1
    
    def check_limited_article(self, article, token_limit):
        """
        Vérifie si un article est limité en tokens.
        
        Args:
            article: dictionnaire de l'article
            token_limit: limite de tokens
        """
        if article.get("token_count", 0) >= token_limit:
            self.limited_articles += 1
    
    def print_progress_summary(self, phase, current, target):
        """Affiche un résumé de progression pour une phase de collecte."""
        percent = (current / target * 100) if target > 0 else 0
        print(f"\nTotal collecté pour {phase}: {current}/{target} tokens ({percent:.1f}%)")
    
    def set_available_articles(self, category, count):
        """
        Définit le nombre d'articles disponibles pour une catégorie.
        
        Args:
            category: nom de la catégorie
            count: nombre d'articles disponibles
        """
        if category in self.categories_stats:
            self.categories_stats[category]["available_articles"] = count

    def generate_summary(self, execution_time, token_limit):
        """
        Génère un résumé des statistiques de collecte.
        
        Args:
            execution_time: temps d'exécution en secondes
            token_limit: limite de tokens pour les articles
        
        Returns:
            Texte formaté avec les statistiques
        """
        # Calculer les pourcentages
        total_percent = (self.total_tokens / self.token_target * 100) if self.token_target > 0 else 0
        main_percent = (self.main_category_tokens / self.total_tokens * 100) if self.total_tokens > 0 else 0
        sub_percent = (self.subcategory_tokens / self.total_tokens * 100) if self.total_tokens > 0 else 0
        random_percent = (self.random_tokens / self.total_tokens * 100) if self.total_tokens > 0 else 0
        
        ordered_percent = (self.main_ordered_tokens / self.main_category_tokens * 100) if self.main_category_tokens > 0 else 0
        main_random_percent = (self.main_random_tokens / self.main_category_tokens * 100) if self.main_category_tokens > 0 else 0
        
        # Calculer le pourcentage d'articles limités
        limited_percentage = (self.limited_articles / self.articles_count * 100) if self.articles_count > 0 else 0
        
        # Formater le résumé
        summary = f"""=== STATISTIQUES FINALES POUR {self.language_code} ===
Temps d'exécution: {execution_time/60:.1f} minutes
Total d'articles: {self.articles_count}
Total de tokens: {self.total_tokens}/{self.token_target} ({total_percent:.1f}%)
Répartition réelle des tokens:
- Catégories principales: {self.main_category_tokens} tokens ({main_percent:.1f}%)
- Sous-catégories: {self.subcategory_tokens} tokens ({sub_percent:.1f}%)
- Articles aléatoires: {self.random_tokens} tokens ({random_percent:.1f}%)

Distribution dans les catégories principales:
- Articles ordonnés: {self.main_ordered_tokens} tokens ({ordered_percent:.1f}%)
- Articles aléatoires: {self.main_random_tokens} tokens ({main_random_percent:.1f}%)

Statistiques par catégorie:
"""
        
        # Ajouter les statistiques par catégorie
        for category, stats in self.categories_stats.items():
            total_cat_tokens = stats["main_tokens"] + stats["sub_tokens"]
            cat_percent = (total_cat_tokens / self.total_tokens * 100) if self.total_tokens > 0 else 0
            
            summary += f"- {category}: {total_cat_tokens} tokens ({cat_percent:.1f}%)\n"
            summary += f"  * {stats['main_articles']} articles principaux ({stats['main_tokens']} tokens)\n"
            summary += f"  * {stats['sub_articles']} articles de sous-catégories ({stats['sub_tokens']} tokens)\n"
            summary += f"  * {stats['available_articles']} articles disponibles au total\n"
        
        # Ajouter les statistiques sur la longueur des articles
        if self.articles_count > 0:
            avg_tokens = self.total_tokens / self.articles_count
            summary += f"\nLongueur moyenne des articles: {avg_tokens:.1f} tokens/article\n"
            summary += f"Articles limités à {token_limit} tokens: {self.limited_articles} ({limited_percentage:.1f}%)\n"
        
        return summary
    
    def save_to_file(self, execution_time, token_limit):
        """
        Sauvegarde les statistiques dans un fichier texte.
        
        Args:
            execution_time: temps d'exécution en secondes
            token_limit: limite de tokens pour les articles
            
        Returns:
            chemin du fichier de statistiques
        """
        # Créer le répertoire pour les statistiques
        stats_dir = "results/metrics/collection/language"
        os.makedirs(stats_dir, exist_ok=True)
        
        # Générer le résumé
        summary = self.generate_summary(execution_time, token_limit)
        
        # Chemin du fichier
        file_path = f"{stats_dir}/{self.language_code}_stats.txt"
        
        # Sauvegarder dans un fichier
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(summary)
        
        logging.info(f"Statistiques finales pour {self.language_code} sauvegardées dans {file_path}")
        
        return file_path
    
    def log_progress(self, phase, current, target):
        """
        Affiche la progression d'une phase de collecte.
        
        Args:
            phase: nom de l'étape de collecte (ex: "Catégories principales")
            current: nombre de tokens collectés
            target: objectif de tokens
        """
        if target > 0:
            percent = (current / target * 100)
            print(f"  Progression {phase}: {current}/{target} tokens ({percent:.1f}%)")


# Fonctions utilitaires
def calculate_token_targets(token_target, params, available_categories):
    """
    Calcule les objectifs de tokens pour chaque méthode de collecte.
    
    Args:
        token_target: objectif total de tokens
        params: paramètres adaptatifs
        available_categories: nombre de catégories disponibles
    
    Returns:
        Tuple (main_target, sub_target, random_target, tokens_per_main, tokens_per_sub)
    """
    # Calculer la répartition globale
    main_category_token_target = int(token_target * params['main_category_ratio'])
    subcategory_token_target = int(token_target * params['subcategory_ratio'])
    random_token_target = token_target - main_category_token_target - subcategory_token_target
    
    # Répartir entre les catégories disponibles
    if available_categories:
        tokens_per_main_category = main_category_token_target // len(available_categories)
        tokens_per_subcategory = subcategory_token_target // len(available_categories)
    else:
        tokens_per_main_category = 0
        tokens_per_subcategory = 0
    
    return (
        main_category_token_target,
        subcategory_token_target,
        random_token_target,
        tokens_per_main_category,
        tokens_per_subcategory
    )


def print_collection_plan(language_code, group, token_target, params, targets):
    """
    Affiche le plan de collecte adaptatif.
    
    Args:
        language_code: code de la langue
        group: groupe de la langue
        token_target: objectif de tokens
        params: paramètres adaptatifs
        targets: tuple des objectifs calculés
    """
    main_target, sub_target, random_target, per_main, per_sub = targets
    
    print(f"\n\n===== COLLECTE POUR LA LANGUE: {language_code} =====")
    print(f"Groupe de langue: {group}")
    print(f"Objectif de tokens adaptatif: {token_target}")
    
    print(f"Paramètres adaptatifs:")
    print(f"- Longueur minimale: {params['min_char_length']} caractères")
    print(f"- Longueur maximale: {params['max_token_length']} tokens")
    print(f"- Ratio catégories principales: {params['main_category_ratio']*100}%")
    print(f"- Ratio sous-catégories: {params['subcategory_ratio']*100}%")
    print(f"- Ratio articles aléatoires: {params['random_ratio']*100}%")
    print(f"- Proportion de sélection ordonnée: {params['fixed_selection_ratio']*100}%")
    
    print(f"Plan de collecte adaptatif:")
    print(f"- Tokens catégories principales: {main_target} ({per_main} par catégorie)")
    print(f"- Tokens sous-catégories: {sub_target} ({per_sub} par catégorie)")
    print(f"- Tokens articles aléatoires: {random_target}")


def save_global_stats(processed_languages, all_articles, output_path):
    """
    Sauvegarde les statistiques globales pour toutes les langues.
    
    Args:
        processed_languages: ensemble des langues traitées
        all_articles: liste de tous les articles collectés
        output_path: chemin du fichier de sortie
    """
    # Calculer des statistiques globales
    total_articles = len(all_articles)
    languages_count = len(processed_languages)
    
    # Compter les articles par langue
    articles_by_language = {}
    for article in all_articles:
        lang = article.get('language', 'unknown')
        articles_by_language[lang] = articles_by_language.get(lang, 0) + 1
    
    # Créer un DataFrame pour les statistiques
    stats_data = []
    for lang, count in articles_by_language.items():
        stats_data.append({
            'language': lang,
            'articles_count': count,
            'percentage': (count / total_articles * 100) if total_articles > 0 else 0
        })
    
    global_stats = pd.DataFrame(stats_data)
    
    # Ajouter une ligne avec le total
    total_row = {
        'language': 'TOTAL',
        'articles_count': total_articles,
        'percentage': 100.0
    }
    global_stats = pd.concat([global_stats, pd.DataFrame([total_row])], ignore_index=True)
    
    # Sauvegarder les statistiques
    global_stats.to_csv(output_path, index=False)
    
    logging.info(f"Statistiques globales sauvegardées dans {output_path}")
    print(f"Statistiques globales sauvegardées: {languages_count} langues, {total_articles} articles")