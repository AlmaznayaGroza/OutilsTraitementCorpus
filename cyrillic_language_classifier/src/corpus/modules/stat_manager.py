"""Gestionnaire de statistiques pour la collecte de corpus Wikipédia

Ce module fournit des outils complets pour la collecte, le calcul et la génération
de statistiques détaillées sur la collecte d'articles Wikipédia. Il encapsule
la logique liée au suivi des performances, à l'analyse des métriques et à la
génération de rapports de collecte.

Fonctionnalités principales :
- Suivi en temps réel des statistiques de collecte par langue
- Calcul automatique des objectifs de tokens et distribution adaptative
- Génération de rapports détaillés avec métriques de performance
- Analyse comparative entre objectifs et résultats obtenus
- Export de statistiques globales et par langue
- Validation des métriques et détection d'anomalies

Le module utilise une approche orientée objet avec la classe CollectionStats
qui maintient l'état des statistiques pour une langue donnée, et des fonctions
utilitaires pour les calculs globaux et la planification de collecte.
"""

import logging
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Set, Any, Union
from pathlib import Path
import json


# === CONSTANTES DE CONFIGURATION ===

# Formats de fichiers supportés
STATS_FILE_EXTENSION = ".txt"
JSON_EXTENSION = ".json"
CSV_EXTENSION = ".csv"

# Types de collecte
COLLECTION_TYPES = {
    "MAIN_ORDERED": "catégories principales (ordonné)",
    "MAIN_RANDOM": "catégories principales (aléatoire)",
    "SUBCATEGORY": "sous-catégories",
    "RANDOM": "articles aléatoires",
}

# Limites de validation
MIN_TOKEN_COUNT = 1
MAX_TOKEN_COUNT = 100000
MIN_EXECUTION_TIME = 0.1    # secondes
MAX_EXECUTION_TIME = 21600  # 6 h

# Métriques par défaut
DEFAULT_METRICS = {
    "total_tokens": 0,
    "total_articles": 0,
    "avg_tokens_per_article": 0.0,
    "collection_efficiency": 0.0,
    "time_per_article": 0.0,
}

# Messages de log standardisés
LOG_MESSAGES = {
    "STATS_INIT": "CollectionStats initialisé pour {lang} (objectif: {target} tokens)",
    "STATS_UPDATE": "Statistiques mises à jour pour {lang}: +{tokens} tokens ({articles} articles)",
    "STATS_EXPORTED": "Statistiques exportées: {path}",
    "STATS_ERROR": "Erreur dans les statistiques pour {lang}: {error}",
    "TARGET_REACHED": "Objectif atteint pour {lang}: {current}/{target} tokens ({percent:.1f}%)",
    "INVALID_METRIC": "Métrique invalide détectée: {metric} = {value}",
    "STATS_VALIDATED": "Validation des statistiques pour {lang}: {status}",
}


class CollectionStats:
    """
    Gestionnaire complet des statistiques de collecte d'articles

    Cette classe encapsule toute la logique liée à la collecte et à la génération
    de statistiques sur les articles collectés pour une langue donnée. Elle maintient
    le suivi en temps réel des métriques de performance et génère des rapports détaillés.

    Attributes:
        language_code: code de la langue traitée (ex: 'ru', 'uk')
        token_target: objectif total de tokens pour cette langue
        categories: liste des catégories disponibles pour cette langue
        start_time: timestamp de début de la collecte
        end_time: timestamp de fin de la collecte (None si en cours)
    """

    def __init__(
        self, language_code: str, token_target: int, categories: List[str]
    ) -> None:
        """
        Initialise le gestionnaire de statistiques pour une langue donnée

        Args:
            language_code: code de la langue traitée (ex: 'ru', 'uk')
            token_target: objectif total de tokens pour cette langue
            categories: liste des catégories disponibles

        Raises:
            ValueError: si les paramètres sont invalides
        """
        # Validation des paramètres d'entrée
        if not isinstance(language_code, str) or not language_code.strip():
            raise ValueError("Le code de langue doit être une chaîne non vide")

        if not isinstance(token_target, int) or token_target <= 0:
            raise ValueError("L'objectif de tokens doit être un entier positif")

        if not isinstance(categories, list) or not categories:
            raise ValueError("Les catégories doivent être une liste non vide")

        if not all(isinstance(cat, str) and cat.strip() for cat in categories):
            raise ValueError("Toutes les catégories doivent être des chaînes non vides")

        # Attributs principaux
        self.language_code = language_code.strip()
        self.token_target = token_target
        self.categories = [cat.strip() for cat in categories]

        # Horodatage
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None

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
        for category in self.categories:
            self.categories_stats[category] = {
                "main_articles": 0,
                "main_tokens": 0,
                "sub_articles": 0,
                "sub_tokens": 0,
                "available_articles": 0,
                "collection_attempts": 0,
                "success_rate": 0.0,
            }

        # Compteurs d'articles
        self.articles_count = 0
        self.limited_articles = 0
        self.redirected_articles = 0  # redirections ignorées


        # Métriques de performance
        self.performance_metrics = {
            "articles_per_minute": 0.0,
            "tokens_per_minute": 0.0,
            "avg_processing_time": 0.0,
            "success_rate": 0.0,
            "efficiency_score": 0.0,
        }

        # Historique des collectes (pour analyse temporelle)
        self.collection_history = []

        logging.info(
            LOG_MESSAGES["STATS_INIT"].format(
                lang=self.language_code, target=self.token_target
            )
        )

    def set_token_targets(
        self,
        main_target: int,
        sub_target: int,
        random_target: int,
        tokens_per_main: int,
        tokens_per_sub: int,
    ) -> None:
        """
        Définit tous les objectifs de tokens en 1 seule opération.

        Args:
            main_target: objectif pour les catégories principales
            sub_target: objectif pour les sous-catégories
            random_target: objectif pour les articles aléatoires
            tokens_per_main: tokens par catégorie principale
            tokens_per_sub: tokens par sous-catégorie

        Raises:
            ValueError: si les objectifs sont invalides
        """
        # Validation des paramètres
        targets = [
            main_target,
            sub_target,
            random_target,
            tokens_per_main,
            tokens_per_sub,
        ]
        if not all(isinstance(t, int) and t >= 0 for t in targets):
            raise ValueError(
                "Tous les objectifs doivent être des entiers positifs ou zéro"
            )

        total_target = main_target + sub_target + random_target
        if total_target != self.token_target:
            logging.warning(
                f"Somme des objectifs ({total_target}) "
                f"différente de l'objectif total ({self.token_target})"
            )

        # Définir les objectifs
        self.main_category_token_target = main_target
        self.subcategory_token_target = sub_target
        self.random_token_target = random_target

        # Objectifs par catégorie
        for category in self.categories_stats:
            self.tokens_per_main_category[category] = tokens_per_main
            self.tokens_per_subcategory[category] = tokens_per_sub

        logging.debug(
            f"Objectifs définis pour {self.language_code}: "
            f"main={main_target}, sub={sub_target}, random={random_target}"
        )

    def update_main_category_stats(
        self,
        category: str,
        article: Dict[str, Any],
        article_type: str
    ) -> None:
        """
        Met à jour les statistiques pour un article de catégorie principale.

        Args:
            category: nom de la catégorie
            article: dictionnaire de l'article avec au minimum 'token_count'
            article_type: type d'article ("ordonné" ou "aléatoire")

        Raises:
            ValueError: si les paramètres sont invalides
        """
        self._validate_article_update_params(category, article, article_type)

        token_count = article.get("token_count", 0)
        if not isinstance(token_count, int) or token_count < 0:
            raise ValueError(f"token_count invalide: {token_count}")

        # Mettre à jour les compteurs globaux
        self.main_category_tokens += token_count
        self.total_tokens += token_count

        # Mettre à jour les statistiques par catégorie
        if category in self.categories_stats:
            self.categories_stats[category]["main_articles"] += 1
            self.categories_stats[category]["main_tokens"] += token_count
            self.categories_stats[category]["collection_attempts"] += 1

        # Mettre à jour les compteurs par type
        if article_type == "ordonné":
            self.main_ordered_tokens += token_count
        elif article_type == "aléatoire":
            self.main_random_tokens += token_count

        self.articles_count += 1

        # Enregistrer dans l'historique
        self._record_collection_event(
            category, token_count, article_type, "main_category"
        )

        # Calculer les métriques de performance
        self._update_performance_metrics()

        logging.debug(
            LOG_MESSAGES["STATS_UPDATE"].format(
                lang=self.language_code, tokens=token_count, articles=1
            )
        )

    def update_subcategory_stats(
            self,
            category: str,
            article: Dict[str, Any]
    ) -> None:
        """
        Met à jour les statistiques pour un article de sous-catégorie.

        Args:
            category: n de la catégorie parent
            article: dictionnaire de l'article avec au minimum 'token_count'

        Raises:
            ValueError: si les paramètres sont invalides
        """
        self._validate_article_update_params(
            category, article, "sous-catégorie"
        )

        token_count = article.get("token_count", 0)
        if not isinstance(token_count, int) or token_count < 0:
            raise ValueError(f"token_count invalide: {token_count}")

        # Mettre à jour les compteurs globaux
        self.subcategory_tokens += token_count
        self.total_tokens += token_count

        # Mettre à jour les statistiques par catégorie
        if category in self.categories_stats:
            self.categories_stats[category]["sub_articles"] += 1
            self.categories_stats[category]["sub_tokens"] += token_count
            self.categories_stats[category]["collection_attempts"] += 1

        self.articles_count += 1

        # Enregistrer dans l'historique
        self._record_collection_event(
            category, token_count, "sous-catégorie", "subcategory"
        )

        # Calculer les métriques de performance
        self._update_performance_metrics()

        logging.debug(
            LOG_MESSAGES["STATS_UPDATE"].format(
                lang=self.language_code, tokens=token_count, articles=1
            )
        )

    def update_random_stats(self, article: Dict[str, Any]) -> None:
        """
        Met à jour les statistiques pour un article aléatoire.

        Args:
            article: dictionnaire de l'article avec au minimum 'token_count'

        Raises:
            ValueError: si l'article est invalide
        """
        if not isinstance(article, dict):
            raise ValueError("L'article doit être un dictionnaire")

        token_count = article.get("token_count", 0)
        if not isinstance(token_count, int) or token_count < 0:
            raise ValueError(f"token_count invalide: {token_count}")

        # Mettre à jour les compteurs globaux
        self.random_tokens += token_count
        self.total_tokens += token_count
        self.articles_count += 1

        # Enregistrer dans l'historique
        self._record_collection_event(
            "Random", token_count, "aléatoire", "random"
        )

        # Calculer les métriques de performance
        self._update_performance_metrics()

        logging.debug(
            LOG_MESSAGES["STATS_UPDATE"].format(
                lang=self.language_code,
                tokens=token_count,
                articles=1
            )
        )

    def check_limited_article(
            self,
            article: Dict[str, Any],
            token_limit: int
    ) -> None:
        """
        Vérifie si un article est limité en tokens et met à jour les compteurs.

        Args:
            article: dictionnaire de l'article
            token_limit: limite de tokens appliquée
        """
        if not isinstance(article, dict):
            raise ValueError("L'article doit être un dictionnaire")

        if not isinstance(token_limit, int) or token_limit <= 0:
            raise ValueError("La limite de tokens doit être un entier positif")

        token_count = article.get("token_count", 0)
        if isinstance(token_count, int) and token_count >= token_limit:
            self.limited_articles += 1
            logging.debug(
                f"Article limité détecté: {token_count} >= {token_limit} tokens"
            )
    
    def check_redirected_article(self, is_redirect: bool) -> None:
        """
        Incrémente le compteur d'articles redirigés si applicable.

        Args:
            is_redirect: True si l'article est une redirection
        """
        if is_redirect:
            self.redirected_articles += 1


    def set_available_articles(self, category: str, count: int) -> None:
        """
        Définit le nombre d'articles disponibles pour une catégorie.

        Args:
            category: nom de la catégorie
            count: nb d'articles disponibles

        Raises:
            ValueError: si les paramètres sont invalides
        """
        if not isinstance(category, str) or not category.strip():
            raise ValueError("Le nom de catégorie doit être une chaîne non vide")

        if not isinstance(count, int) or count < 0:
            raise ValueError("Le nombre d'articles doit être un entier positif ou zéro")

        category = category.strip()
        if category in self.categories_stats:
            self.categories_stats[category]["available_articles"] = count

            # Calculer le taux de succès si des tentatives ont été faites
            attempts = self.categories_stats[category]["collection_attempts"]
            articles_collected = (
                self.categories_stats[category]["main_articles"]
                + self.categories_stats[category]["sub_articles"]
            )

            if attempts > 0:
                self.categories_stats[category]["success_rate"] = (
                    articles_collected / attempts
                )

            logging.debug(f"Articles disponibles pour {category}: {count}")
        else:
            logging.warning(f"Catégorie inconnue: {category}")

    def print_progress_summary(
            self,
            phase: str,
            current: int,
            target: int
    ) -> None:
        """
        Affiche un résumé de progression pour une phase de collecte

        Args:
            phase: nom de l'étape de collecte
            current: nb de tokens collectés
            target: objectif de tokens
        """
        if not isinstance(phase, str) or not phase.strip():
            logging.warning("Phase de progression invalide")
            return

        if not isinstance(current, int) or not isinstance(target, int):
            logging.warning("Valeurs de progression invalides")
            return

        if target > 0:
            percent = current / target * 100
            print(
                f"  Progression {phase}: {current}/{target} tokens ({percent:.1f}%)"
            )

            # Log si l'objectif est atteint
            if current >= target:
                logging.info(
                    LOG_MESSAGES["TARGET_REACHED"].format(
                        lang=self.language_code,
                        current=current,
                        target=target,
                        percent=percent,
                    )
                )
        else:
            print(f"  {phase}: {current} tokens collectés")

    def log_progress(self, phase: str, current: int, target: int) -> None:
        """
        Enregistre la progression d'une phase de collecte dans les logs

        Args:
            phase: nom de l'étape de collecte
            current: nb de tokens collectés
            target: objectif de tokens
        """
        self.print_progress_summary(phase, current, target)

    def finalize_collection(self) -> None:
        """
        Finalise la collecte en enregistrant l'heure de fin
        et en calculant les métriques finales
        """
        self.end_time = datetime.now()
        self._update_performance_metrics()
        self._validate_final_metrics()

        execution_time = self.get_execution_time()
        logging.info(
            f"Collecte finalisée pour {self.language_code} "
            f"(durée: {execution_time/60:.1f} minutes, "
            f"articles: {self.articles_count}, tokens: {self.total_tokens})"
        )

    def get_execution_time(self) -> float:
        """
        Calcule le temps d'exécution total en secondes.

        Returns:
            temps d'exécution en secondes
        """
        end_time = self.end_time or datetime.now()
        delta = end_time - self.start_time
        return delta.total_seconds()

    def get_completion_percentage(self) -> float:
        """
        Calcule le pourcentage de completion de l'objectif

        Returns:
            pourcentage de completion (0-100)
        """
        if self.token_target <= 0:
            return 0.0

        return min((self.total_tokens / self.token_target) * 100, 100.0)

    def _validate_article_update_params(
        self,
        category: str,
        article: Dict[str, Any],
        article_type: str
    ) -> None:
        """Valide les paramètres pour les mises à jour d'articles."""
        if not isinstance(category, str) or not category.strip():
            raise ValueError("Le nom de catégorie doit être une chaîne non vide")

        if not isinstance(article, dict):
            raise ValueError("L'article doit être un dictionnaire")

        if "token_count" not in article:
            raise ValueError("L'article doit contenir un champ 'token_count'")

        if not isinstance(article_type, str) or not article_type.strip():
            raise ValueError("Le type d'article doit être une chaîne non vide")

    def _record_collection_event(
        self,
        category: str,
        tokens: int,
        article_type: str,
        collection_method: str
    ) -> None:
        """Enregistre un événement de collecte dans l'historique."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "category": category,
            "tokens": tokens,
            "article_type": article_type,
            "collection_method": collection_method,
            "total_tokens_at_time": self.total_tokens,
        }

        self.collection_history.append(event)

        # Limiter l'historique pour éviter une croissance excessive
        if len(self.collection_history) > 1000:
            self.collection_history = self.collection_history[
                -800:
            ]  # garder les 800 plus récents

    def _update_performance_metrics(self) -> None:
        """Met à jour les métriques de performance en temps réel."""
        execution_time = self.get_execution_time()

        if execution_time > 0 and self.articles_count > 0:
            # Métriques de vitesse
            self.performance_metrics["articles_per_minute"] = (
                self.articles_count / execution_time
            ) * 60
            self.performance_metrics["tokens_per_minute"] = (
                self.total_tokens / execution_time
            ) * 60
            self.performance_metrics["avg_processing_time"] = (
                execution_time / self.articles_count
            )

            # Taux de réussite global
            total_attempts = sum(
                cat["collection_attempts"] for cat in self.categories_stats.values()
            )
            if total_attempts > 0:
                self.performance_metrics["success_rate"] = (
                    self.articles_count / total_attempts
                )

            # Score d'efficacité (completion × vitesse)
            completion_rate = self.get_completion_percentage() / 100
            speed_factor = min(
                self.performance_metrics["articles_per_minute"] / 10, 1.0
            )  # Normalisé sur 10 art/min
            self.performance_metrics["efficiency_score"] = (
                completion_rate * speed_factor
            )

    def _validate_final_metrics(self) -> None:
        """Valide les métriques finales et signale les anomalies."""
        issues = []

        # Vérifier la cohérence des totaux
        calculated_total = (
            self.main_category_tokens + self.subcategory_tokens + self.random_tokens
        )
        if abs(calculated_total - self.total_tokens) > 1:  # tolérance de 1 token
            issues.append(
                f"Incohérence des totaux: calculé={calculated_total}, "
                f"enregistré={self.total_tokens}"
            )

        # Vérifier les métriques de performance
        if self.performance_metrics["articles_per_minute"] > 1000:  # seuil irréaliste
            issues.append(
                "Vitesse de collecte irréaliste: "
                f"{self.performance_metrics['articles_per_minute']:.1f} art/min"
            )

        # Vérifier les ratios
        if self.articles_count > 0:
            avg_tokens = self.total_tokens / self.articles_count
            if avg_tokens < 1 or avg_tokens > 10000:
                issues.append(
                    f"Moyenne de tokens par article suspecte: {avg_tokens:.1f}"
                )

        # Logger les problèmes détectés
        if issues:
            status = "ANOMALIES DÉTECTÉES"
            for issue in issues:
                logging.warning(
                    LOG_MESSAGES["INVALID_METRIC"].format(
                        metric="validation", value=issue
                    )
                )
        else:
            status = "VALIDE"

        logging.info(
            LOG_MESSAGES["STATS_VALIDATED"].format(
                lang=self.language_code, status=status
            )
        )

    def generate_summary(self, execution_time: float, token_limit: int) -> str:
        """
        Génère un résumé complet des statistiques de collecte

        Args:
            execution_time: emps d'exécution en secondes
            token_limit: limite de tokens pour les articles

        Returns:
            texte formaté avec les statistiques complètes
        """
        if not isinstance(execution_time, (int, float)) or execution_time < 0:
            raise ValueError("Le temps d'exécution doit être un nombre positif")

        if not isinstance(token_limit, int) or token_limit <= 0:
            raise ValueError("La limite de tokens doit être un entier positif")

        # Calculer les pourcentages
        total_percent = (
            (self.total_tokens / self.token_target * 100)
            if self.token_target > 0
            else 0
        )
        main_percent = (
            (self.main_category_tokens / self.total_tokens * 100)
            if self.total_tokens > 0
            else 0
        )
        sub_percent = (
            (self.subcategory_tokens / self.total_tokens * 100)
            if self.total_tokens > 0
            else 0
        )
        random_percent = (
            (self.random_tokens / self.total_tokens * 100)
            if self.total_tokens > 0
            else 0
        )

        ordered_percent = (
            (self.main_ordered_tokens / self.main_category_tokens * 100)
            if self.main_category_tokens > 0
            else 0
        )
        main_random_percent = (
            (self.main_random_tokens / self.main_category_tokens * 100)
            if self.main_category_tokens > 0
            else 0
        )

        # Calculer le pourcentage d'articles limités
        limited_percentage = (
            (self.limited_articles / self.articles_count * 100)
            if self.articles_count > 0
            else 0
        )

        # Calculer les métriques de performance
        avg_tokens = (
            self.total_tokens / self.articles_count if self.articles_count > 0 else 0
        )

        # Formater le résumé
        summary = f"""=== STATISTIQUES FINALES POUR {self.language_code.upper()} ===
Période de collecte: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')} - {(self.end_time or datetime.now()).strftime('%Y-%m-%d %H:%M:%S')}
Temps d'exécution: {execution_time/60:.1f} minutes ({execution_time:.1f} secondes)
Total d'articles: {self.articles_count:,}
Total de tokens: {self.total_tokens:,}/{self.token_target:,} ({total_percent:.1f}%)

=== RÉPARTITION RÉELLE DES TOKENS ===
- Catégories principales: {self.main_category_tokens:,} tokens ({main_percent:.1f}%)
- Sous-catégories: {self.subcategory_tokens:,} tokens ({sub_percent:.1f}%)
- Articles aléatoires: {self.random_tokens:,} tokens ({random_percent:.1f}%)

=== DISTRIBUTION DANS LES CATÉGORIES PRINCIPALES ===
- Articles ordonnés: {self.main_ordered_tokens:,} tokens ({ordered_percent:.1f}%)
- Articles aléatoires: {self.main_random_tokens:,} tokens ({main_random_percent:.1f}%)

=== MÉTRIQUES DE PERFORMANCE ===
- Vitesse de collecte: {self.performance_metrics['articles_per_minute']:.1f} articles/minute
- Débit de tokens: {self.performance_metrics['tokens_per_minute']:.1f} tokens/minute
- Temps moyen par article: {self.performance_metrics['avg_processing_time']:.2f} secondes
- Taux de réussite: {self.performance_metrics['success_rate']*100:.1f}%
- Score d'efficacité: {self.performance_metrics['efficiency_score']*100:.1f}%

=== STATISTIQUES PAR CATÉGORIE ==="""

        # Ajouter les statistiques par catégorie
        for category, stats in self.categories_stats.items():
            total_cat_tokens = stats["main_tokens"] + stats["sub_tokens"]
            cat_percent = (
                (total_cat_tokens / self.total_tokens * 100)
                if self.total_tokens > 0
                else 0
            )

            summary += f"""
{category}:
  • Total: {total_cat_tokens:,} tokens ({cat_percent:.1f}% du corpus)
  • Articles principaux: {stats['main_articles']} articles ({stats['main_tokens']:,} tokens)
  • Articles de sous-catégories: {stats['sub_articles']} articles ({stats['sub_tokens']:,} tokens)
  • Articles disponibles: {stats['available_articles']:,}
  • Tentatives de collecte: {stats['collection_attempts']}
  • Taux de succès: {stats['success_rate']*100:.1f}%"""

        # Ajouter les statistiques sur la longueur des articles
        summary += f"""

=== ANALYSE DES ARTICLES ===
Longueur moyenne: {avg_tokens:.1f} tokens/article
Articles limités à {token_limit} tokens: {self.limited_articles} ({limited_percentage:.1f}%)
Articles redirigés ignorés: {self.redirected_articles}
"""

        # Ajouter l'analyse temporelle si on a des données d'historique
        if len(self.collection_history) > 1:
            summary += self._generate_temporal_analysis()

        return summary

    def _generate_temporal_analysis(self) -> str:
        """Génère une analyse temporelle de la collecte."""
        if len(self.collection_history) < 2:
            return ""

        # Analyser la progression dans le temps
        first_event = self.collection_history[0]
        last_event = self.collection_history[-1]

        first_time = datetime.fromisoformat(first_event["timestamp"])
        last_time = datetime.fromisoformat(last_event["timestamp"])

        duration = (last_time - first_time).total_seconds()

        if duration > 0:
            # Calculer la vitesse moyenne
            tokens_collected = last_event["total_tokens_at_time"]
            avg_rate = tokens_collected / duration * 60  # tokens par minute

            # Analyser les phases de collecte
            phases = {}
            for event in self.collection_history:
                method = event["collection_method"]
                if method not in phases:
                    phases[method] = {"tokens": 0, "articles": 0}
                phases[method]["tokens"] += event["tokens"]
                phases[method]["articles"] += 1

            analysis = f"""
=== ANALYSE TEMPORELLE ===
Durée effective de collecte: {duration/60:.1f} minutes
Vitesse moyenne globale: {avg_rate:.1f} tokens/minute
Phases de collecte identifiées: {len(phases)}"""

            for phase, data in phases.items():
                analysis += f"""
  • {phase.replace('_', ' ').title()}: {data['articles']} articles, {data['tokens']:,} tokens"""

            return analysis

        return ""

    def save_to_file(
        self,
        execution_time: float,
        token_limit: int,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> str:
        """
        Sauvegarde les statistiques dans un fichier texte

        Args:
            execution_time: temps d'exécution en secondes
            token_limit: limite de tokens pour les articles
            output_dir: dossier de sortie (optionnel)

        Returns:
            chemin du fichier de statistiques créé

        Raises:
            IOError: si la sauvegarde échoue
        """
        # Déterminer le dossier de sortie
        if output_dir is None:
            output_dir = Path("results/metrics/collection/language")
        else:
            output_dir = Path(output_dir)

        try:
            # Créer le répertoire
            output_dir.mkdir(parents=True, exist_ok=True)

            # Générer le résumé
            summary = self.generate_summary(execution_time, token_limit)

            # Chemin du fichier avec timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"{self.language_code}_stats_{timestamp}{STATS_FILE_EXTENSION}"
            file_path = output_dir / file_name

            # Sauvegarder dans le fichier
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(summary)

            logging.info(LOG_MESSAGES["STATS_EXPORTED"].format(path=file_path))
            print(f"📊 Statistiques sauvegardées: {file_path}")

            return str(file_path)

        except Exception as e:
            error_msg = f"Erreur lors de la sauvegarde des statistiques: {e}"
            logging.error(
                LOG_MESSAGES["STATS_ERROR"].format(
                    lang=self.language_code, error=str(e)
                )
            )
            raise IOError(error_msg)

    def export_to_json(
            self,
            output_dir: Optional[Union[str, Path]] = None
    ) -> str:
        """
        Exporte les statistiques au format JSON pour analyse programmatique

        Args:
            output_dir: dossier de sortie (optionnel)

        Returns:
            chemin du fichier JSON créé
        """
        # Déterminer le dossier de sortie
        if output_dir is None:
            output_dir = Path("results/metrics/collection/language")
        else:
            output_dir = Path(output_dir)

        try:
            output_dir.mkdir(parents=True, exist_ok=True)

            # Préparer les données pour JSON
            export_data = {
                "language_code": self.language_code,
                "collection_metadata": {
                    "start_time": self.start_time.isoformat(),
                    "end_time": self.end_time.isoformat() if self.end_time else None,
                    "execution_time_seconds": self.get_execution_time(),
                    "token_target": self.token_target,
                    "completion_percentage": self.get_completion_percentage(),
                },
                "targets": {
                    "main_category_target": self.main_category_token_target,
                    "subcategory_target": self.subcategory_token_target,
                    "random_target": self.random_token_target,
                    "tokens_per_main_category": dict(self.tokens_per_main_category),
                    "tokens_per_subcategory": dict(self.tokens_per_subcategory),
                },
                "results": {
                    "total_tokens": self.total_tokens,
                    "total_articles": self.articles_count,
                    "main_category_tokens": self.main_category_tokens,
                    "subcategory_tokens": self.subcategory_tokens,
                    "random_tokens": self.random_tokens,
                    "main_ordered_tokens": self.main_ordered_tokens,
                    "main_random_tokens": self.main_random_tokens,
                    "limited_articles": self.limited_articles,
                },
                "performance_metrics": dict(self.performance_metrics),
                "categories_stats": dict(self.categories_stats),
                "collection_history": self.collection_history[
                    -100:
                ],  # Derniers 100 événements
            }

            # Chemin du fichier
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"{self.language_code}_stats_{timestamp}{JSON_EXTENSION}"
            file_path = output_dir / file_name

            # Sauvegarder en JSON
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)

            logging.info(f"Statistiques JSON exportées: {file_path}")
            return str(file_path)

        except Exception as e:
            error_msg = f"Erreur lors de l'export JSON: {e}"
            logging.error(error_msg)
            raise IOError(error_msg)

    def get_summary_dict(self) -> Dict[str, Any]:
        """
        Retourne un dictionnaire avec les statistiques principales

        Returns:
            dictionnaire des statistiques essentielles
        """
        return {
            "language_code": self.language_code,
            "total_tokens": self.total_tokens,
            "total_articles": self.articles_count,
            "token_target": self.token_target,
            "completion_percentage": self.get_completion_percentage(),
            "execution_time_seconds": self.get_execution_time(),
            "performance_metrics": dict(self.performance_metrics),
            "categories_count": len(self.categories),
            "main_category_tokens": self.main_category_tokens,
            "subcategory_tokens": self.subcategory_tokens,
            "random_tokens": self.random_tokens,
        }


# === FONCTIONS UTILITAIRES ===


def calculate_token_targets(
    token_target: int,
    params: Dict[str, float],
    available_categories: int
) -> Tuple[int, int, int, int, int]:
    """
    Calcule les objectifs de tokens pour chaque méthode de collecte

    Args:
        token_target: objectif total de tokens
        params: paramètres adaptatifs avec les ratios
        available_categories: nb de catégories disponibles

    Returns:
        tuple (main_target, sub_target, random_target, tokens_per_main, tokens_per_sub)

    Raises:
        ValueError: si les paramètres sont invalides
    """
    # Validation des paramètres
    if not isinstance(token_target, int) or token_target <= 0:
        raise ValueError("L'objectif de tokens doit être un entier positif")

    if not isinstance(params, dict):
        raise ValueError("Les paramètres doivent être un dictionnaire")

    required_params = [
        "main_category_ratio", "subcategory_ratio", "random_ratio"
    ]
    for param in required_params:
        if param not in params:
            raise ValueError(f"Paramètre manquant: {param}")
        if (
            not isinstance(params[param], (int, float))
            or not (0 <= params[param] <= 1)
        ):
            raise ValueError(f"Le paramètre {param} doit être un nombre entre 0 et 1")

    if not isinstance(available_categories, int) or available_categories < 0:
        raise ValueError("Le nombre de catégories doit être un entier positif ou zéro")

    # Vérifier que les ratios sont cohérents
    total_ratio = (
        params["main_category_ratio"]
        + params["subcategory_ratio"]
        + params["random_ratio"]
    )
    if abs(total_ratio - 1.0) > 0.01:  # tolérance de 1%
        logging.warning(f"Somme des ratios ({total_ratio:.3f}) différente de 1.0")

    # Calculer la répartition globale
    main_category_token_target = int(token_target * params["main_category_ratio"])
    subcategory_token_target = int(token_target * params["subcategory_ratio"])
    random_token_target = (
        token_target - main_category_token_target - subcategory_token_target
    )

    # Répartir entre les catégories disponibles
    if available_categories > 0:
        tokens_per_main_category = main_category_token_target // available_categories
        tokens_per_subcategory = subcategory_token_target // available_categories
    else:
        tokens_per_main_category = 0
        tokens_per_subcategory = 0
        logging.warning(
            "Aucune catégorie disponible, objectifs par catégorie définis à 0"
        )

    # Logger la distribution calculée
    logging.debug(
        f"Distribution calculée pour {token_target} tokens: "
        f"main={main_category_token_target}, sub={subcategory_token_target}, "
        f"random={random_token_target}"
    )

    return (
        main_category_token_target,
        subcategory_token_target,
        random_token_target,
        tokens_per_main_category,
        tokens_per_subcategory,
    )


def print_collection_plan(
    language_code: str,
    group: str,
    token_target: int,
    params: Dict[str, Any],
    targets: Tuple[int, int, int, int, int],
) -> None:
    """
    Affiche le plan de collecte adaptatif de manière formatée

    Args:
        language_code: code de la langue
        group: groupe de la langue (A, B, C, D)
        token_target: objectif de tokens
        params: paramètres adaptatifs
        targets: tuple des objectifs calculés
    """
    if not isinstance(language_code, str) or not language_code.strip():
        logging.error("Code de langue invalide pour l'affichage du plan")
        return

    if not isinstance(targets, tuple) or len(targets) != 5:
        logging.error("Tuple des objectifs invalide pour l'affichage du plan")
        return

    main_target, sub_target, random_target, per_main, per_sub = targets

    print(f"\n\n===== COLLECTE POUR LA LANGUE: {language_code.upper()} =====")
    print(f"Groupe de langue: {group}")
    print(f"Objectif de tokens adaptatif: {token_target:,}")

    print("\nParamètres adaptatifs:")
    print(f"- Longueur minimale: {params.get('min_char_length', 'N/A')} caractères")
    print(f"- Longueur maximale: {params.get('max_token_length', 'N/A')} tokens")
    print(
        f"- Ratio catégories principales: {params.get('main_category_ratio', 0)*100:.1f}%"
    )
    print(f"- Ratio sous-catégories: {params.get('subcategory_ratio', 0)*100:.1f}%")
    print(f"- Ratio articles aléatoires: {params.get('random_ratio', 0)*100:.1f}%")
    print(
        f"- Proportion de sélection ordonnée: {params.get('fixed_selection_ratio', 0)*100:.1f}%"
    )

    print("\nPlan de collecte adaptatif:")
    print(
        f"- Tokens catégories principales: {main_target:,} ({per_main:,} par catégorie)"
    )
    print(f"- Tokens sous-catégories: {sub_target:,} ({per_sub:,} par catégorie)")
    print(f"- Tokens articles aléatoires: {random_target:,}")

    # Afficher des métriques estimées
    estimated_articles = token_target // 150  # estimation de 150 tokens par article
    estimated_time = estimated_articles * 2  # estimation de 2 secondes par article

    print("\nEstimations:")
    print(f"- Articles estimés: ~{estimated_articles:,}")
    print(f"- Temps estimé: ~{estimated_time/60:.1f} minutes")


def save_global_stats(
    processed_languages: Set[str],
    all_articles: List[Dict[str, Any]],
    output_path: Union[str, Path],
) -> str:
    """
    Sauvegarde les statistiques globales pour toutes les langues

    Args:
        processed_languages: ensemble des langues traitées
        all_articles: liste de tous les articles collectés
        output_path: chemin du fichier de sortie

    Returns:
        chemin du fichier de statistiques créé

    Raises:
        ValueError: si les paramètres sont invalides
        IOError: si la sauvegarde échoue
    """
    # Validation des paramètres
    if not isinstance(processed_languages, set):
        raise ValueError("processed_languages doit être un ensemble")

    if not isinstance(all_articles, list):
        raise ValueError("all_articles doit être une liste")

    if not processed_languages:
        logging.warning("Aucune langue traitée, statistiques globales vides")

    output_path = Path(output_path)

    try:
        # Créer le dossier parent si nécessaire
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Calculer des statistiques globales
        total_articles = len(all_articles)
        languages_count = len(processed_languages)

        # Compter les articles par langue
        articles_by_language = {}
        tokens_by_language = {}

        for article in all_articles:
            lang = article.get("language", "unknown")
            articles_by_language[lang] = articles_by_language.get(lang, 0) + 1

            # Compter les tokens si disponible
            tokens = article.get("token_count", 0)
            if isinstance(tokens, int):
                tokens_by_language[lang] = tokens_by_language.get(lang, 0) + tokens

        # Créer un DataFrame pour les statistiques
        stats_data = []
        for lang in sorted(processed_languages):
            article_count = articles_by_language.get(lang, 0)
            token_count = tokens_by_language.get(lang, 0)
            avg_tokens = token_count / article_count if article_count > 0 else 0

            stats_data.append(
                {
                    "language": lang,
                    "articles_count": article_count,
                    "tokens_total": token_count,
                    "tokens_avg_per_article": round(avg_tokens, 1),
                    "percentage_of_corpus": (
                        round((article_count / total_articles * 100), 2)
                        if total_articles > 0
                        else 0
                    ),
                }
            )

        global_stats = pd.DataFrame(stats_data)

        # Ajouter une ligne avec le total
        total_tokens = sum(tokens_by_language.values())
        total_row = {
            "language": "TOTAL",
            "articles_count": total_articles,
            "tokens_total": total_tokens,
            "tokens_avg_per_article": (
                round(total_tokens / total_articles, 1) if total_articles > 0 else 0
            ),
            "percentage_of_corpus": 100.0,
        }
        global_stats = pd.concat(
            [global_stats, pd.DataFrame([total_row])], ignore_index=True
        )

        # Sauvegarder les statistiques
        global_stats.to_csv(output_path, index=False, encoding="utf-8")

        # Créer aussi un résumé textuel
        summary_path = output_path.with_suffix(".txt")
        summary_text = f"""=== STATISTIQUES GLOBALES DU CORPUS ===
Généré le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Résumé général:
- Langues traitées: {languages_count}
- Articles totaux: {total_articles:,}
- Tokens totaux: {total_tokens:,}
- Moyenne tokens/article: {total_tokens/total_articles:.1f} (si total_articles > 0 else 0)

Répartition par langue:
"""

        for _, row in global_stats.iterrows():
            if row["language"] != "TOTAL":
                summary_text += f"- {row['language']}: {row['articles_count']:,} articles ({row['percentage_of_corpus']:.1f}%), {row['tokens_total']:,} tokens\n"

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_text)

        logging.info(LOG_MESSAGES["STATS_EXPORTED"].format(path=output_path))
        logging.info(f"Résumé textuel sauvegardé: {summary_path}")
        print(
            f"📊 Statistiques globales sauvegardées: {languages_count} langues, {total_articles:,} articles"
        )

        return str(output_path)

    except Exception as e:
        error_msg = f"Erreur lors de la sauvegarde des statistiques globales: {e}"
        logging.error(error_msg)
        raise IOError(error_msg)


def compare_language_performance(
        stats_list: List[CollectionStats]
) -> pd.DataFrame:
    """
    Compare les performances de collecte entre plusieurs langues

    Args:
        stats_list: liste des objets CollectionStats à comparer

    Returns:
        DataFrame avec la comparaison des performances
    """
    if not isinstance(stats_list, list) or not stats_list:
        raise ValueError(
            "stats_list doit être une liste non vide d'objets CollectionStats"
        )

    comparison_data = []

    for stats in stats_list:
        if not isinstance(stats, CollectionStats):
            logging.warning(f"Objet non-CollectionStats ignoré: {type(stats)}")
            continue

        summary = stats.get_summary_dict()
        comparison_data.append(summary)

    if not comparison_data:
        return pd.DataFrame()

    comparison_df = pd.DataFrame(comparison_data)

    # Ajouter des colonnes de classement
    if len(comparison_data) > 1:
        comparison_df["completion_rank"] = comparison_df["completion_percentage"].rank(
            ascending=False
        )
        comparison_df["efficiency_rank"] = (
            comparison_df["performance_metrics"]
            .apply(lambda x: x.get("efficiency_score", 0))
            .rank(ascending=False)
        )
        comparison_df["speed_rank"] = (
            comparison_df["performance_metrics"]
            .apply(lambda x: x.get("articles_per_minute", 0))
            .rank(ascending=False)
        )

    return comparison_df


def generate_performance_report(
    stats_list: List[CollectionStats],
    output_dir: Union[str, Path]
) -> str:
    """
    Génère un rapport de performance comparatif entre langues

    Args:
        stats_list: liste des statistiques de collecte
        output_dir: dossier de sortie

    Returns:
        chemin du fichier de rapport créé
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Créer la comparaison
    comparison = compare_language_performance(stats_list)

    if comparison.empty:
        raise ValueError("Impossible de générer le rapport: aucune donnée valide")

    # Générer le rapport
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"performance_report_{timestamp}.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== RAPPORT DE PERFORMANCE COMPARATIF ===\n")
        f.write(f"Généré le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Résumé global
        f.write(f"Langues analysées: {len(comparison)}\n")
        f.write(f"Tokens totaux collectés: {comparison['total_tokens'].sum():,}\n")
        f.write(
            f"Articles totaux collectés: {comparison['total_articles'].sum():,}\n\n"
        )

        # Classements
        if "completion_rank" in comparison.columns:
            f.write("=== CLASSEMENTS ===\n")
            f.write("Meilleur taux de completion:\n")
            top_completion = comparison.nsmallest(3, "completion_rank")
            for _, row in top_completion.iterrows():
                f.write(
                    f"  {row['language_code']}: {row['completion_percentage']:.1f}%\n"
                )

            f.write("\nMeilleure efficacité:\n")
            top_efficiency = comparison.nsmallest(3, "efficiency_rank")
            for _, row in top_efficiency.iterrows():
                efficiency = row["performance_metrics"].get("efficiency_score", 0) * 100
                f.write(f"  {row['language_code']}: {efficiency:.1f}%\n")

    logging.info(f"Rapport de performance généré: {report_path}")
    return str(report_path)
