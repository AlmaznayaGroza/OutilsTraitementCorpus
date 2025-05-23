"""Gestionnaire de données pour la sauvegarde et fusion de corpus Wikipédia

Ce module fournit des fonctions spécialisées pour la gestion des données
collectées depuis Wikipédia. Il encapsule la logique de sauvegarde, de fusion
et de manipulation des fichiers CSV contenant les articles du corpus.

Fonctionnalités principales :
- Sauvegarde d'articles en format CSV avec validation des données
- Fusion intelligente de nouvelles données avec des données existantes
- Gestion des doublons avec stratégies de déduplication configurables
- Validation des structures de données et intégrité des fichiers
- Gestion robuste des erreurs avec récupération automatique
- Prise en charge de différents encodages et formats de fichiers

Le module respecte les bonnes pratiques de gestion de fichiers et assure
l'intégrité des données lors des opérations de fusion et sauvegarde.
"""

import logging
import pandas as pd
import shutil
from typing import List, Dict, Optional, Tuple, Any, Union
from pathlib import Path
import csv
from datetime import datetime


# === CONSTANTES DE CONFIGURATION ===

# Encodages supportés
DEFAULT_ENCODING = "utf-8"
FALLBACK_ENCODINGS = ["utf-8", "utf-8-sig", "cp1251", "koi8-r"]

# Extensions de fichiers
CSV_EXTENSION = ".csv"
BACKUP_SUFFIX = "_backup"
TEMP_SUFFIX = "_temp"

# Colonnes requises pour les articles
REQUIRED_COLUMNS = {
    "language",
    "title",
    "text",
    "pageid",
    "url",
    "category",
    "type",
    "token_count",
    "char_count",
}

# Colonnes optionnelles
OPTIONAL_COLUMNS = {"is_truncated", "truncation_ratio"}

# Limites de validation
MIN_ARTICLE_LENGTH = 10
MAX_TITLE_LENGTH = 500
MIN_TOKEN_COUNT = 1
MAX_TOKEN_COUNT = 10000

# Messages de log standardisés
LOG_MESSAGES = {
    "SAVE_SUCCESS": "Sauvegarde réussie: {count} articles dans {path}",
    "SAVE_ERROR": "Erreur lors de la sauvegarde dans {path}: {error}",
    "MERGE_START": "Début de la fusion des données pour {lang}",
    "MERGE_SUCCESS": "Fusion réussie pour {lang}: {total} articles au total",
    "MERGE_REPLACE": "Remplacement du fichier existant pour {lang} suite à une erreur de fusion",
    "VALIDATION_ERROR": "Erreur de validation des données: {error}",
    "BACKUP_CREATED": "Sauvegarde créée: {backup_path}",
    "FILE_NOT_FOUND": "Fichier non trouvé: {path}",
    "DUPLICATE_REMOVED": "Doublons supprimés: {count} articles en double",
}


def validate_article_data(
        articles: List[Dict[str, Any]]
) -> Tuple[bool, List[str]]:
    """
    Valide la structure et le contenu des données d'articles.

    Args:
        articles: liste des articles à valider

    Returns:
        tuple (validation_réussie, liste_des_erreurs)
    """
    if not isinstance(articles, list):
        return False, ["Les articles doivent être fournis sous forme de liste"]

    if not articles:
        return False, ["La liste d'articles ne peut pas être vide"]

    errors = []

    for i, article in enumerate(articles):
        if not isinstance(article, dict):
            errors.append(f"Article {i}: doit être un dictionnaire")
            continue

        # Vérifier les colonnes requises
        missing_columns = REQUIRED_COLUMNS - set(article.keys())
        if missing_columns:
            errors.append(f"Article {i}: colonnes manquantes: {missing_columns}")

        # Valider les types et valeurs
        validation_errors = _validate_single_article(article, i)
        errors.extend(validation_errors)

    return len(errors) == 0, errors


def _validate_single_article(article: Dict[str, Any], index: int) -> List[str]:
    """
    Valide un seul article.

    Args:
        article: dictionnaire de l'article
        index: index de l'article pour les messages d'erreur

    Returns:
        liste des erreurs trouvées
    """
    errors = []

    # Validation du titre
    title = article.get("title")
    if not isinstance(title, str) or not title.strip():
        errors.append(f"Article {index}: titre invalide ou vide")
    elif len(title) > MAX_TITLE_LENGTH:
        errors.append(
            f"Article {index}: titre trop long "
            f"({len(title)} > {MAX_TITLE_LENGTH})"
        )

    # Validation du texte
    text = article.get("text")
    if not isinstance(text, str) or len(text) < MIN_ARTICLE_LENGTH:
        errors.append(
            f"Article {index}: texte trop court "
            f"(minimum {MIN_ARTICLE_LENGTH} caractères)"
        )

    # Validation de l'ID de page
    pageid = article.get("pageid")
    if not isinstance(pageid, int) or pageid <= 0:
        errors.append(f"Article {index}: pageid doit être un entier positif")

    # Validation du nombre de tokens
    token_count = article.get("token_count")
    if not isinstance(token_count, int) or not (
        MIN_TOKEN_COUNT <= token_count <= MAX_TOKEN_COUNT
    ):
        errors.append(f"Article {index}: token_count invalide ({token_count})")

    # Validation du nombre de caractères
    char_count = article.get("char_count")
    if not isinstance(char_count, int) or char_count <= 0:
        errors.append(
            f"Article {index}: char_count doit être un entier positif"
        )

    # Validation de la langue
    language = article.get("language")
    if not isinstance(language, str) or not language.strip():
        errors.append(f"Article {index}: code de langue invalide")

    # Validation de l'URL
    url = article.get("url")
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        errors.append(f"Article {index}: URL invalide")

    return errors


def create_backup_if_exists(file_path: Union[str, Path]) -> Optional[str]:
    """
    Crée une sauvegarde du fichier s'il existe.

    Args:
        file_path: chemin vers le fichier à sauvegarder

    Returns:
        chemin de la sauvegarde créée ou None si pas de fichier existant

    Raises:
        IOError: si la création de la sauvegarde échoue
    """
    file_path = Path(file_path)

    if not file_path.exists():
        return None

    # Générer un nom de sauvegarde unique
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{file_path.stem}{BACKUP_SUFFIX}_{timestamp}{file_path.suffix}"
    backup_path = file_path.parent / backup_name

    try:
        shutil.copy2(file_path, backup_path)
        logging.info(
            LOG_MESSAGES["BACKUP_CREATED"].format(backup_path=backup_path)
        )
        return str(backup_path)
    except (IOError, OSError) as e:
        raise IOError(f"Impossible de créer la sauvegarde {backup_path}: {e}")


def save_articles_to_csv(
    language_code: str,
    articles: List[Dict[str, Any]],
    output_folder: Union[str, Path],
    create_backup: bool = True,
    encoding: str = DEFAULT_ENCODING,
    validate_data: bool = True,
) -> str:
    """
    Sauvegarde les articles dans un fichier CSV avec validation optionnelle.

    Args:
        language_code: code de la langue des articles
        articles: liste des articles à sauvegarder
        output_folder: dossier de destination
        create_backup: si True, crée une sauvegarde du fichier existant
        encoding: encodage du fichier CSV
        validate_data: si True, valide les données avant sauvegarde

    Returns:
        chemin du fichier CSV créé

    Raises:
        ValueError: si les données sont invalides
        IOError: si la sauvegarde échoue
    """
    # Validation des paramètres d'entrée
    if not isinstance(language_code, str) or not language_code.strip():
        raise ValueError("Le code de langue doit être une chaîne non vide")

    if not isinstance(articles, list):
        raise ValueError("Les articles doivent être fournis sous forme de liste")

    if not articles:
        logging.warning("Liste d'articles vide, aucun fichier créé")
        return ""

    # Validation des données si demandée
    if validate_data:
        is_valid, validation_errors = validate_article_data(articles)
        if not is_valid:
            error_msg = f"Données invalides: {'; '.join(validation_errors)}"
            logging.error(
                LOG_MESSAGES["VALIDATION_ERROR"].format(error=error_msg)
            )
            raise ValueError(error_msg)

    # Préparer les chemins
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    csv_path = output_folder / f"{language_code}_articles{CSV_EXTENSION}"

    # Créer une sauvegarde si demandée
    backup_path = None
    if create_backup:
        backup_path = create_backup_if_exists(csv_path)

    try:
        # Créer le DataFrame et sauvegarder
        df = pd.DataFrame(articles)

        # Réorganiser les colonnes avec les colonnes requises en premier
        all_columns = list(df.columns)
        required_first = [
            col for col in REQUIRED_COLUMNS if col in all_columns
        ]
        other_columns = [
            col for col in all_columns if col not in REQUIRED_COLUMNS
        ]
        ordered_columns = required_first + sorted(other_columns)

        df = df[ordered_columns]

        # Sauvegarder avec gestion d'erreurs d'encodage
        try:
            df.to_csv(
                csv_path,
                index=False,
                encoding=encoding,
                quoting=csv.QUOTE_NONNUMERIC,
                escapechar="\\",
            )
        except UnicodeEncodeError:
            # Fallback vers UTF-8 avec BOM si l'encodage spécifié échoue
            logging.warning(
                f"Échec d'encodage {encoding}, fallback vers utf-8-sig"
            )
            df.to_csv(
                csv_path,
                index=False,
                encoding="utf-8-sig",
                quoting=csv.QUOTE_NONNUMERIC,
                escapechar="\\",
            )

        # Vérifier que le fichier a été créé correctement
        if not csv_path.exists() or csv_path.stat().st_size == 0:
            raise IOError("Le fichier CSV créé est vide ou n'existe pas")

        logging.info(
            LOG_MESSAGES["SAVE_SUCCESS"].format(
                count=len(articles), path=csv_path
            )
        )
        print(f"✓ {len(articles)} articles sauvegardés dans {csv_path}")

        return str(csv_path)

    except Exception as e:
        # En cas d'erreur, restaurer la sauvegarde si elle existe
        if backup_path and Path(backup_path).exists():
            try:
                shutil.copy2(backup_path, csv_path)
                logging.info(
                    f"Fichier restauré depuis la sauvegarde: {backup_path}"
                )
            except Exception as restore_error:
                logging.error(f"Échec de la restauration: {restore_error}")

        error_msg = f"Erreur lors de la sauvegarde: {e}"
        logging.error(
            LOG_MESSAGES["SAVE_ERROR"].format(path=csv_path, error=str(e))
        )
        raise IOError(error_msg)


def load_csv_with_fallback_encoding(
        file_path: Union[str, Path]
) -> pd.DataFrame:
    """
    Charge un fichier CSV en essayant différents encodages.

    Args:
        file_path: chemin vers le fichier CSV

    Returns:
        DataFrame pandas chargé

    Raises:
        IOError: si le fichier ne peut pas être lu avec aucun encodage
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise IOError(f"Fichier non trouvé: {file_path}")

    for encoding in FALLBACK_ENCODINGS:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            logging.debug(f"Fichier {file_path} lu avec l'encodage {encoding}")
            return df
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as e:
            logging.error(
                f"Erreur lors du chargement de {file_path} avec {encoding}: {e}"
            )
            continue

    raise IOError(
        f"Impossible de lire {file_path} avec les encodages disponibles: {FALLBACK_ENCODINGS}"
    )


def remove_duplicates(
    df: pd.DataFrame,
    deduplication_columns: Optional[List[str]] = None,
    keep_strategy: str = "first",
) -> Tuple[pd.DataFrame, int]:
    """
    Supprime les doublons d'un DataFrame avec stratégie configurable.

    Args:
        df: DataFrame à dédupliquer
        deduplication_columns: colonnes sur lesquelles baser la déduplication
        keep_strategy: stratégie de conservation
            ("first", "last", False pour supprimer tous)

    Returns:
        tuple (DataFrame_dédupliqué, nombre_de_doublons_supprimés)
    """
    if df.empty:
        return df, 0

    initial_count = len(df)

    # Colonnes de déduplication par défaut
    if deduplication_columns is None:
        available_columns = set(df.columns)
        if "text" in available_columns:
            deduplication_columns = ["text"]
        elif "pageid" in available_columns:
            deduplication_columns = ["pageid"]
        else:
            # Utiliser toutes les colonnes si aucune colonne standard n'est trouvée
            deduplication_columns = list(df.columns)

    # Filtrer les colonnes qui existent réellement
    existing_columns = [
        col for col in deduplication_columns if col in df.columns
    ]

    if not existing_columns:
        logging.warning(
            "Aucune colonne de déduplication trouvée, "
            "pas de déduplication effectuée"
        )
        return df, 0

    # Supprimer les doublons
    try:
        deduplicated_df = df.drop_duplicates(
            subset=existing_columns, keep=keep_strategy
        )
        duplicates_removed = initial_count - len(deduplicated_df)

        if duplicates_removed > 0:
            logging.info(
                LOG_MESSAGES["DUPLICATE_REMOVED"].format(
                    count=duplicates_removed
                )
            )

        return deduplicated_df, duplicates_removed

    except Exception as e:
        logging.error(f"Erreur lors de la déduplication: {e}")
        return df, 0


def merge_with_existing_data(
    missing_languages: List[str],
    temp_dir: Union[str, Path],
    target_dir: Union[str, Path],
    deduplication_strategy: str = "text",
    create_backups: bool = True,
) -> Dict[str, Any]:
    """
    Fusionne les données collectées avec les données existantes.

    Cette fonction combine intelligemment les nouvelles données avec les fichiers
    existants, en gérant les doublons et en créant des sauvegardes de sécurité.

    Args:
        missing_languages: liste des codes de langues à fusionner
        temp_dir: dossier contenant les nouveaux fichiers temporaires
        target_dir: dossier de destination final
        deduplication_strategy: stratégie de déduplication
            ("text", "pageid", "all")
        create_backups: si True, crée des sauvegardes avant fusion

    Returns:
        dictionnaire avec les statistiques de fusion

    Raises:
        ValueError: si les paramètres sont invalides
        IOError: si les opérations de fichiers échouent
    """
    # Validation des paramètres
    if not isinstance(missing_languages, list) or not missing_languages:
        raise ValueError("missing_languages doit être une liste non vide")

    temp_dir = Path(temp_dir)
    target_dir = Path(target_dir)

    if not temp_dir.exists():
        raise IOError(f"Dossier temporaire non trouvé: {temp_dir}")

    # Créer le dossier de destination
    target_dir.mkdir(parents=True, exist_ok=True)

    # Statistiques de fusion
    stats = {
        "processed_languages": 0,
        "merged_files": 0,
        "created_files": 0,
        "error_count": 0,
        "total_articles_before": 0,
        "total_articles_after": 0,
        "duplicates_removed": 0,
        "backups_created": [],
    }

    logging.info("Fusion des nouvelles données avec les données existantes...")
    print("Fusion des données en cours...")

    # Mapper les stratégies de déduplication
    dedup_columns_map = {
        "text": ["text"],
        "pageid": ["pageid"],
        "all": None,  # utilise toutes les colonnes
        "title": ["title"],
    }

    dedup_columns = dedup_columns_map.get(deduplication_strategy, ["text"])

    for language in missing_languages:
        try:
            result = _merge_single_language(
                language,
                temp_dir,
                target_dir,
                dedup_columns,
                create_backups,
                stats
            )

            if result["success"]:
                if result["action"] == "merged":
                    stats["merged_files"] += 1
                elif result["action"] == "created":
                    stats["created_files"] += 1

                stats["total_articles_before"] += result.get(
                    "articles_before", 0
                )
                stats["total_articles_after"] += result.get(
                    "articles_after", 0
                )
                stats["duplicates_removed"] += result.get(
                    "duplicates_removed", 0
                )

                if result.get("backup_path"):
                    stats["backups_created"].append(result["backup_path"])
            else:
                stats["error_count"] += 1

            stats["processed_languages"] += 1

        except Exception as e:
            logging.error(
                f"Erreur lors du traitement de {language}: {e}", exc_info=True
            )
            stats["error_count"] += 1

    # Afficher le résumé final
    _print_merge_summary(stats)

    logging.info(
        f"Fusion terminée: {stats['processed_languages']} langues traitées"
    )

    return stats


def _merge_single_language(
    language: str,
    temp_dir: Path,
    target_dir: Path,
    dedup_columns: Optional[List[str]],
    create_backups: bool,
    stats: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Fusionne les données pour une seule langue.

    Args:
        language: code de la langue
        temp_dir: dossier temporaire
        target_dir: dossier de destination
        dedup_columns: colonnes pour la déduplication
        create_backups: si True, crée des sauvegardes
        stats: dictionnaire de statistiques (modifié en place)

    Returns:
        dictionnaire avec les résultats de la fusion
    """
    temp_file = temp_dir / f"{language}_articles{CSV_EXTENSION}"
    target_file = target_dir / f"{language}_articles{CSV_EXTENSION}"

    result = {
        "success": False,
        "action": None,
        "articles_before": 0,
        "articles_after": 0,
        "duplicates_removed": 0,
        "backup_path": None,
    }

    if not temp_file.exists():
        logging.warning(LOG_MESSAGES["FILE_NOT_FOUND"].format(path=temp_file))
        # Mettre à jour les stats globales même en cas d'échec
        stats["languages_not_found"] = stats.get("languages_not_found", 0) + 1
        return result

    try:
        # Charger les nouvelles données
        new_df = load_csv_with_fallback_encoding(temp_file)
        result["articles_before"] = len(new_df)

        # Mettre à jour les stats globales
        stats["total_new_articles"] = stats.get(
            "total_new_articles", 0
        ) + len(new_df)

        logging.info(LOG_MESSAGES["MERGE_START"].format(lang=language))
        logging.info(
            f"Nouvelles données pour {language}: {len(new_df)} articles"
        )

        # Cas 1: Fusion avec données existantes
        if target_file.exists():
            # Créer une sauvegarde si demandée
            if create_backups:
                backup_path = create_backup_if_exists(target_file)
                result["backup_path"] = backup_path
                # Mettre à jour les stats des sauvegardes
                if backup_path:
                    stats["backups_created"].append(backup_path)

            try:
                existing_df = load_csv_with_fallback_encoding(target_file)
                existing_count = len(existing_df)
                logging.info(
                    f"Données existantes pour {language}: {existing_count} articles"
                )

                # Mettre à jour les stats globales
                stats["total_existing_articles"] = stats.get(
                    "total_existing_articles", 0
                ) + existing_count

                # Combiner les DataFrames
                combined_df = pd.concat(
                    [existing_df, new_df], ignore_index=True
                )

                # Supprimer les doublons
                deduplicated_df, duplicates_count = remove_duplicates(
                    combined_df, dedup_columns, "first"
                )

                result["duplicates_removed"] = duplicates_count
                result["articles_after"] = len(deduplicated_df)

                # Mettre à jour les stats globales avec les résultats de déduplication
                stats["total_duplicates_removed"] = stats.get(
                    "total_duplicates_removed", 0
                ) + duplicates_count
                stats["total_final_articles"] = stats.get(
                    "total_final_articles", 0
                ) + len(deduplicated_df)

                # Sauvegarder le résultat fusionné
                deduplicated_df.to_csv(
                    target_file,
                    index=False,
                    encoding=DEFAULT_ENCODING
                )

                logging.info(
                    LOG_MESSAGES["MERGE_SUCCESS"].format(
                        lang=language,
                        total=len(deduplicated_df)
                    )
                )
                print(
                    f"✓ Fusion réussie pour {language}: "
                    f"{len(deduplicated_df)} articles au total"
                )

                result["success"] = True
                result["action"] = "merged"

            except Exception as merge_error:
                logging.error(
                    f"Erreur lors de la fusion pour {language}: {merge_error}"
                )

                # Mettre à jour les stats d'erreur
                stats["merge_errors"] = stats.get("merge_errors", 0) + 1

                # En cas d'erreur, remplacer par le nouveau fichier
                shutil.copy2(temp_file, target_file)
                result["articles_after"] = len(new_df)

                # Mettre à jour les stats globales pour le remplacement
                stats["total_final_articles"] = stats.get(
                    "total_final_articles", 0
                ) + len(new_df)

                logging.info(
                    LOG_MESSAGES["MERGE_REPLACE"].format(lang=language)
                )
                print(f"⚠️ Erreur lors de la fusion pour {language} - fichier remplacé")

                result["success"] = True
                result["action"] = "replaced"

                # Mettre à jour les stats de remplacement
                stats["files_replaced"] = stats.get("files_replaced", 0) + 1

        # Cas 2: Nouveau fichier (pas de données existantes)
        else:
            # Supprimer les doublons dans les nouvelles données
            deduplicated_df, duplicates_count = remove_duplicates(
                new_df, dedup_columns, "first"
            )

            result["duplicates_removed"] = duplicates_count
            result["articles_after"] = len(deduplicated_df)

            # Mettre à jour les stats globales
            stats["total_duplicates_removed"] = stats.get(
                "total_duplicates_removed", 0
            ) + duplicates_count
            stats["total_final_articles"] = stats.get(
                "total_final_articles", 0
            ) + len(deduplicated_df)

            # Copier le fichier dédupliqué
            deduplicated_df.to_csv(
                target_file,
                index=False,
                encoding=DEFAULT_ENCODING
            )

            logging.info(f"Nouveau fichier créé pour {language}")
            print(
                f"✓ Nouveau fichier créé pour {language}: "
                f"{len(deduplicated_df)} articles"
            )

            result["success"] = True
            result["action"] = "created"

            # Mettre à jour les stats de création
            stats["new_files_created"] = stats.get("new_files_created", 0) + 1

    except Exception as e:
        logging.error(
            f"Erreur lors du traitement pour {language}: {e}", exc_info=True
        )
        print(f"❌ Erreur lors du traitement pour {language}")
        result["success"] = False

        # Mettre à jour les stats d'erreur globales
        stats["processing_errors"] = stats.get("processing_errors", 0) + 1

    return result


def _print_merge_summary(stats: Dict[str, Any]) -> None:
    """Affiche un résumé des statistiques de fusion."""
    print(f"\n=== RÉSUMÉ DE LA FUSION ===")
    print(f"Langues traitées: {stats['processed_languages']}")
    print(f"Fichiers fusionnés: {stats['merged_files']}")
    print(f"Nouveaux fichiers: {stats['created_files']}")
    print(f"Erreurs: {stats['error_count']}")
    print(f"Articles avant fusion: {stats['total_articles_before']}")
    print(f"Articles après fusion: {stats['total_articles_after']}")
    print(f"Doublons supprimés: {stats['duplicates_removed']}")
    
    if stats['backups_created']:
        print(f"Sauvegardes créées: {len(stats['backups_created'])}")


def cleanup_temp_files(
    temp_dir: Union[str, Path], file_patterns: Optional[List[str]] = None
) -> int:
    """
    Nettoie les fichiers temporaires.

    Args:
        temp_dir: dossier contenant les fichiers temporaires
        file_patterns: motifs de fichiers à supprimer (optionnel)

    Returns:
        nombre de fichiers supprimés
    """
    temp_dir = Path(temp_dir)

    if not temp_dir.exists():
        return 0

    if file_patterns is None:
        file_patterns = [f"*{TEMP_SUFFIX}*", f"*{CSV_EXTENSION}"]

    deleted_count = 0

    for pattern in file_patterns:
        for file_path in temp_dir.glob(pattern):
            try:
                if file_path.is_file():
                    file_path.unlink()
                    deleted_count += 1
                    logging.debug(f"Fichier temporaire supprimé: {file_path}")
            except OSError as e:
                logging.warning(f"Impossible de supprimer {file_path}: {e}")

    if deleted_count > 0:
        logging.info(
            f"Nettoyage terminé: {deleted_count} fichiers temporaires supprimés"
        )

    return deleted_count


def get_corpus_statistics(corpus_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Calcule des statistiques sur un corpus existant.

    Args:
        corpus_dir: dossier contenant les fichiers du corpus

    Returns:
        dictionnaire avec les statistiques du corpus
    """
    corpus_dir = Path(corpus_dir)

    stats = {
        "total_files": 0,
        "total_articles": 0,
        "total_tokens": 0,
        "total_characters": 0,
        "languages": [],
        "files_by_language": {},
        "avg_tokens_per_article": 0,
        "avg_chars_per_article": 0,
    }

    if not corpus_dir.exists():
        return stats

    # Parcourir tous les fichiers CSV
    csv_files = list(corpus_dir.glob(f"*{CSV_EXTENSION}"))
    stats["total_files"] = len(csv_files)

    for csv_file in csv_files:
        try:
            # Extraire le code de langue du nom de fichier
            language = csv_file.stem.replace("_articles", "")
            stats["languages"].append(language)

            # Charger et analyser le fichier
            df = load_csv_with_fallback_encoding(csv_file)
            article_count = len(df)

            stats["files_by_language"][language] = {
                "articles": article_count,
                "file_size": csv_file.stat().st_size,
            }

            stats["total_articles"] += article_count

            # Calculer les tokens et caractères si les colonnes existent
            if "token_count" in df.columns:
                tokens = df["token_count"].sum()
                stats["total_tokens"] += tokens
                stats["files_by_language"][language]["tokens"] = tokens

            if "char_count" in df.columns:
                chars = df["char_count"].sum()
                stats["total_characters"] += chars
                stats["files_by_language"][language]["characters"] = chars

        except Exception as e:
            logging.error(f"Erreur lors de l'analyse de {csv_file}: {e}")
            continue

    # Calculer les moyennes
    if stats["total_articles"] > 0:
        stats["avg_tokens_per_article"] = (
            stats["total_tokens"] / stats["total_articles"]
        )
        stats["avg_chars_per_article"] = (
            stats["total_characters"] / stats["total_articles"]
        )

    return stats


def validate_corpus_integrity(
    corpus_dir: Union[str, Path],
    repair_errors: bool = False
) -> Dict[str, Any]:
    """
    Valide l'intégrité d'un corpus existant.

    Args:
        corpus_dir: dossier contenant le corpus
        repair_errors: si True, tente de réparer les erreurs détectées

    Returns:
        dictionnaire avec les résultats de validation
    """
    corpus_dir = Path(corpus_dir)

    validation_results = {
        "is_valid": True,
        "total_files_checked": 0,
        "valid_files": 0,
        "invalid_files": 0,
        "errors": [],
        "warnings": [],
        "repaired_files": [],
    }

    if not corpus_dir.exists():
        validation_results["is_valid"] = False
        validation_results["errors"].append(
            f"Dossier du corpus non trouvé: {corpus_dir}"
        )
        return validation_results

    csv_files = list(corpus_dir.glob(f"*{CSV_EXTENSION}"))
    validation_results["total_files_checked"] = len(csv_files)

    for csv_file in csv_files:
        try:
            # Charger le fichier
            df = load_csv_with_fallback_encoding(csv_file)

            # Vérifier la structure
            file_errors = []

            # Vérifier les colonnes requises
            missing_columns = REQUIRED_COLUMNS - set(df.columns)
            if missing_columns:
                file_errors.append(f"Colonnes manquantes: {missing_columns}")

            # Vérifier les données
            if not df.empty:
                # Convertir en liste de dictionnaires pour la validation
                articles = df.to_dict("records")
                is_valid, data_errors = validate_article_data(articles)

                if not is_valid:
                    file_errors.extend(
                        data_errors[:5]
                    )  # limiter à 5 erreurs par fichier

            # Traiter les résultats
            if file_errors:
                validation_results["invalid_files"] += 1
                validation_results["is_valid"] = False
                validation_results["errors"].append(
                    f"{csv_file.name}: {'; '.join(file_errors)}"
                )

                # Tentative de réparation si demandée
                if repair_errors:
                    repaired = _attempt_file_repair(csv_file, df)
                    if repaired:
                        validation_results["repaired_files"].append(
                            str(csv_file)
                        )
            else:
                validation_results["valid_files"] += 1

        except Exception as e:
            validation_results["invalid_files"] += 1
            validation_results["is_valid"] = False
            validation_results["errors"].append(
                f"{csv_file.name}: Erreur de lecture - {e}"
            )

    return validation_results


def _attempt_file_repair(
        file_path: Path,
        df: pd.DataFrame
) -> bool:
    """
    Tente de réparer un fichier avec des erreurs basiques.
    
    Se concentre sur les réparations simples et communes.
    """
    try:
        backup_path = create_backup_if_exists(file_path)
        repaired_df = df.copy()
        repairs_applied = []
        
        # Réparation 1: Nettoyer les données critiques
        if "title" in repaired_df.columns:
            before = len(repaired_df)
            repaired_df = repaired_df.dropna(subset=["title"])
            repaired_df = repaired_df[repaired_df["title"].str.strip() != ""]
            if len(repaired_df) != before:
                repairs_applied.append(
                    f"Suppression {before - len(repaired_df)} titres invalides"
                )
        
        # Réparation 2: Corriger les pageids
        if "pageid" in repaired_df.columns:
            before = len(repaired_df)
            repaired_df["pageid"] = pd.to_numeric(
                repaired_df["pageid"], errors="coerce"
            )
            repaired_df = repaired_df.dropna(subset=["pageid"])
            repaired_df["pageid"] = repaired_df["pageid"].astype(int)
            if len(repaired_df) != before:
                repairs_applied.append(
                    f"Correction {before - len(repaired_df)} pageids invalides"
                )
        
        # Réparation 3: Ajouter les colonnes basiques manquantes
        if "language" not in repaired_df.columns:
            lang_code = file_path.stem.replace("_articles", "")
            repaired_df["language"] = lang_code
            repairs_applied.append("Ajout colonne language")
        
        if (
            "token_count" not in repaired_df.columns
            and "text" in repaired_df.columns
        ):
            repaired_df["token_count"] = repaired_df["text"].str.split().str.len()
            repairs_applied.append("Calcul token_count")
        
        # Vérifier si on a fait des réparations utiles
        if not repairs_applied or repaired_df.empty:
            return False
        
        # Sauvegarder
        repaired_df.to_csv(file_path, index=False, encoding=DEFAULT_ENCODING)
        
        logging.info(
            f"Fichier réparé: {file_path} "
            f"(sauvegarde: {backup_path}) - {'; '.join(repairs_applied)}"
        )
        return True
        
    except Exception as e:
        logging.error(f"Échec de la réparation de {file_path}: {e}")
        return False


def export_corpus_summary(
    corpus_dir: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None
) -> str:
    """
    Exporte un résumé détaillé du corpus.

    Args:
        corpus_dir: dossier contenant le corpus
        output_file: fichier de sortie (optionnel)

    Returns:
        chemin du fichier de résumé créé
    """
    corpus_dir = Path(corpus_dir)

    if output_file is None:
        output_file = corpus_dir / "corpus_summary.txt"
    else:
        output_file = Path(output_file)

    # Collecter les statistiques
    stats = get_corpus_statistics(corpus_dir)
    validation = validate_corpus_integrity(corpus_dir)

    # Générer le résumé
    summary_lines = [
        "=== RÉSUMÉ DU CORPUS WIKIPEDIA ===",
        f"Généré le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Dossier du corpus: {corpus_dir}",
        "",
        "=== STATISTIQUES GÉNÉRALES ===",
        f"Nombre total de fichiers: {stats['total_files']}",
        f"Nombre total d'articles: {stats['total_articles']:,}",
        f"Nombre total de tokens: {stats['total_tokens']:,}",
        f"Nombre total de caractères: {stats['total_characters']:,}",
        f"Moyenne tokens/article: {stats['avg_tokens_per_article']:.1f}",
        f"Moyenne caractères/article: {stats['avg_chars_per_article']:.1f}",
        "",
        "=== LANGUES DISPONIBLES ===",
        f"Nombre de langues: {len(stats['languages'])}",
        f"Codes de langue: {', '.join(sorted(stats['languages']))}",
        "",
        "=== DÉTAIL PAR LANGUE ===",
    ]

    # Ajouter les détails par langue
    for language in sorted(stats["languages"]):
        lang_info = stats["files_by_language"][language]
        summary_lines.extend(
            [
                f"{language}:",
                f"  Articles: {lang_info['articles']:,}",
                (
                    f"  Tokens: {lang_info.get('tokens', 'N/A'):,}"
                    if isinstance(lang_info.get("tokens"), int)
                    else "  Tokens: N/A"
                ),
                (
                    f"  Caractères: {lang_info.get('characters', 'N/A'):,}"
                    if isinstance(lang_info.get("characters"), int)
                    else "  Caractères: N/A"
                ),
                f"  Taille fichier: {lang_info['file_size']:,} octets",
                "",
            ]
        )

    # Ajouter les résultats de validation
    summary_lines.extend(
        [
            "=== VALIDATION D'INTÉGRITÉ ===",
            f"Statut global: {'✓ VALIDE' if validation['is_valid'] else '❌ ERREURS DÉTECTÉES'}",
            f"Fichiers vérifiés: {validation['total_files_checked']}",
            f"Fichiers valides: {validation['valid_files']}",
            f"Fichiers avec erreurs: {validation['invalid_files']}",
            "",
        ]
    )

    if validation["errors"]:
        summary_lines.append("=== ERREURS DÉTECTÉES ===")
        for error in validation["errors"][:10]:  # limiter à 10 erreurs
            summary_lines.append(f"- {error}")
        if len(validation["errors"]) > 10:
            summary_lines.append(
                f"... et {len(validation['errors']) - 10} autres erreurs"
            )
        summary_lines.append("")

    if validation["warnings"]:
        summary_lines.append("=== AVERTISSEMENTS ===")
        for warning in validation["warnings"][:5]:
            summary_lines.append(f"- {warning}")
        summary_lines.append("")

    # Sauvegarder le résumé
    summary_text = "\n".join(summary_lines)

    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(summary_text)

        logging.info(f"Résumé du corpus exporté: {output_file}")
        return str(output_file)

    except Exception as e:
        error_msg = f"Erreur lors de l'export du résumé: {e}"
        logging.error(error_msg)
        raise IOError(error_msg)
