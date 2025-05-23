"""Script de fusion des variantes orthographiques du bélarussien

Ce script implémente la fusion méthodologiquement motivée des 2 principales
variantes (principalement) orthographiques du bélarussien: narkamoŭka
(variante officielle) et taraškevica (variante traditionnelle).
Cette unification linguistique optimise la cohérence du corpus
tout en préservant la diversité textuelle de cette langue.

Justification linguistique:
    Les 2 variantes représentent le même système linguistique avec des
    conventions orthographiques et parfois lexicales distinctes,
    issues de perspectives politiques et culturels différentes.
    Leur fusion respecte la réalité linguistique sous-jacente
    tout en créant un corpus plus robuste pour application en TAL.

Méthodologie de fusion:
    Le processus harmonise les codes de langue (be-tarask → be) tout en
    préservant l'intégrité des métadonnées et des contenus textuels.
    Cette approche permet de maintenir la traçabilité des sources tout
    en optimisant l'utilité computationnelle du corpus résultant.

Impact sur le corpus:
    Cette fusion double la taille des échantillons bélarussiens
    disponibles pour l'entraînement et l'évaluation, améliorant ainsi
    la significativité statistique des analyses et la robustesse
    des modèles développés sur cette langue.
"""

import pandas as pd
import os
import shutil
import logging
from typing import Tuple


# =============================================================================
# CONSTANTES DE CONFIGURATION POUR LA FUSION BÉLARUSSIENNE
# =============================================================================

# Codes de langue et leur harmonisation
BELARUSIAN_VARIANTS = {
    "traditional": "be-tarask",  # traditionnelle (taraškevica)
    "official": "be",  # officielle (narkamoŭka)
    "unified": "be",  # code unifié après fusion
}

# Configuration des répertoires par défaut
DEFAULT_PATHS = {
    "input_dir": "data/processed/cleaned",
    "output_dir": "data/processed/merged",
}

# Noms des fichiers attendus et générés
FILE_PATTERNS = {
    "official_input": "be_cleaned_articles.csv",
    "traditional_input": "be-tarask_cleaned_articles.csv",
    "merged_output": "be_merged_cleaned_articles.csv",
}

# Configuration du logging
LOGGING_CONFIG = {
    "level": logging.INFO,
    "format": "%(asctime)s - %(levelname)s - %(message)s",
}


# =============================================================================
# FONCTIONS UTILITAIRES POUR LA VALIDATION ET LA GESTION D'ERREURS
# =============================================================================


def validate_input_directory(input_dir: str) -> None:
    """Valide l'existence et l'accessibilité du répertoire d'entrée

    Cette fonction vérifie que le dossier contenant les fichiers de
    variantes bélarussiennes existe et est accessible en lecture.

    Args:
        input_dir (str): chemin vers le dossier d'entrée

    Raises:
        FileNotFoundError: si le dossier n'existe pas
        PermissionError: si le dossier n'est pas accessible en lecture
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(
            f"Le dossier d'entrée '{input_dir}' "
            f"n'existe pas"
        )

    if not os.access(input_dir, os.R_OK):
        raise PermissionError(
            f"Pas d'autorisation de lecture "
            f"sur '{input_dir}'"
        )


def validate_output_directory(output_dir: str) -> None:
    """Valide et crée si nécessaire le répertoire de sortie

    Cette fonction s'assure que le dossier de destination pour les
    fichiers fusionnés existe et est accessible en écriture.

    Args:
        output_dir (str): chemin vers le dossier de sortie

    Raises:
        PermissionError: si impossible de créer ou écrire dans le dossier
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        # Tester les permissions d'écriture
        test_file = os.path.join(output_dir, ".test_write")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
    except PermissionError:
        raise PermissionError(
            f"Impossible d'écrire dans le dossier "
            f"'{output_dir}'"
        )
    except Exception as e:
        raise RuntimeError(
            f"Erreur lors de la validation du "
            f"dossier de sortie: {e}"
        )


def load_belarusian_variant(file_path: str, variant_name: str) -> pd.DataFrame:
    """Charge un fichier de variante bélarussienne avec validation

    Cette fonction charge et valide un fichier CSV contenant des articles
    dans une variante spécifique du bélarussien, en s'assurant de la
    cohérence des données et de la présence des colonnes essentielles.

    Args:
        file_path (str): chemin vers le fichier à charger
        variant_name (str): nom de la variante pour les messages informatifs

    Returns:
        pd.DataFrame: DataFrame contenant les articles
            (ou DataFrame vide si le fichier n'existe pas ou est invalide)
    """
    if not os.path.exists(file_path):
        print(f"Fichier {variant_name} non trouvé: {file_path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(file_path)

        # Validation des colonnes essentielles
        required_columns = ["language", "text", "title"]
        missing_columns = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:
            print(
                f"Colonnes manquantes dans {variant_name}: "
                f"{missing_columns}"
            )
            return pd.DataFrame()

        # Filtrage des lignes valides (avec texte non vide)
        original_count = len(df)
        df = df[df["text"].notna() & (df["text"] != "")]

        if len(df) < original_count:
            print(
                f"Filtré {original_count - len(df)} "
                f"articles vides pour {variant_name}"
            )

        print(f"Chargé {len(df)} articles valides pour {variant_name}")
        return df

    except Exception as e:
        print(f"Erreur lors du chargement de {variant_name}: {e}")
        return pd.DataFrame()


def copy_other_language_files(input_dir: str, output_dir: str) -> int:
    """Copie tous les autres fichiers de langues vers le dossier de sortie

    Cette fonction préserve tous les fichiers de langues autres que les
    variantes bélarussiennes dans le dossier de sortie, maintenant
    ainsi l'intégrité complète du corpus multilingue.

    Args:
        input_dir (str): dossier source contenant tous les fichiers
        output_dir (str): dossier de destination

    Returns:
        int: nb de fichiers copiés avec succès

    Note:
        La fonction exclut automatiquement les fichiers des variantes
        bélarussiennes pour éviter les doublons avec le fichier fusionné.
    """
    excluded_files = {
        FILE_PATTERNS["official_input"],
        FILE_PATTERNS["traditional_input"],
    }

    copied_count = 0

    for filename in os.listdir(input_dir):
        if filename.endswith(".csv") and filename not in excluded_files:
            src_path = os.path.join(input_dir, filename)
            dst_path = os.path.join(output_dir, filename)

            try:
                shutil.copy2(src_path, dst_path)
                copied_count += 1
            except Exception as e:
                print(f"Erreur lors de la copie de {filename}: {e}")

    return copied_count


# ===============================
# FONCTION PRINCIPALE DE FUSION
# ===============================


def merge_belarusian_variants(
    input_dir: str = DEFAULT_PATHS["input_dir"],
    output_dir: str = DEFAULT_PATHS["output_dir"],
) -> Tuple[pd.DataFrame, dict]:
    """
    Fusionne les variantes du bélarussien
    avec validation complète et statistiques

    Cette fonction implémente le processus complet de fusion des variantes
    du bélarussien, depuis la validation des données d'entrée
    jusqu'à la génération du corpus unifié et des statistiques associées.

    Processus de fusion:
        1. Validation des dossiers d'entrée et de sortie
        2. Chargement et validation des fichiers de variantes
        3. Harmonisation des codes de langue (be-tarask → be)
        4. Fusion des DataFrames avec préservation des métadonnées
        5. Sauvegarde du corpus unifié et copie des autres langues
        6. Génération de statistiques détaillées sur la fusion

    Args:
        input_dir (str): dossier contenant les fichiers nettoyés par langue
            (par éfaut: 'data/processed/cleaned')
        output_dir (str): Répertoire où sauvegarder le corpus fusionné
            (par défaut: 'data/processed/merged')

    Returns:
        Tuple[pd.DataFrame, dict]: tuple contenant le DataFrame fusionné
            et un dictionnaire de statistiques détaillées sur l'opération

    Raises:
        FileNotFoundError: si les dossiers d'entrée n'existent pas
        PermissionError: si impossible d'accéder aux dossiers requis
        RuntimeError: si la fusion échoue pour des raisons techniques

    Example:
        >>> merged_df, stats = merge_belarusian_variants()
        >>> print(f"Fusion réussie: {stats['total_articles']} articles")
        >>> print(
        >>>    f"Gain: +{stats['articles_gain']} articles par rapport à 'be' seul"
        >>> )

    Note linguistique :
        Cette fusion respecte la réalité sociolinguistique du bélarussien
        où les 2 variantes coexistent, permettant de créer un corpus
        plus représentatif de la diversité textuelle réelle de cette langue.
    """

    # Configuration du logging pour cette opération
    logging.basicConfig(**LOGGING_CONFIG)
    logger = logging.getLogger(__name__)

    logger.info("Début de la fusion des variantes bélarussiennes")

    try:
        # 1. Validation des dossiers
        validate_input_directory(input_dir)
        validate_output_directory(output_dir)

        # 2. Construction des chemins de fichiers
        be_path = os.path.join(input_dir, FILE_PATTERNS["official_input"])
        be_tarask_path = os.path.join(input_dir, FILE_PATTERNS["traditional_input"])
        output_path = os.path.join(output_dir, FILE_PATTERNS["merged_output"])

        # 3. Chargement des variantes avec validation
        print("Chargement des variantes bélarussiennes...")
        be_df = load_belarusian_variant(be_path, "bélarussien officiel (be)")
        be_tarask_df = load_belarusian_variant(
            be_tarask_path, "bélarussien traditionnel (be-tarask)"
        )

        # 4. Calcul des statistiques pré-fusion
        stats = {
            "official_articles": len(be_df),
            "traditional_articles": len(be_tarask_df),
            "had_traditional_variant": not be_tarask_df.empty,
            "input_directory": input_dir,
            "output_directory": output_dir,
        }

        print("\nStatistiques avant fusion:")
        print(
            f"Articles en bélarussien officiel (be): "
            f"{stats['official_articles']}"
        )
        print(
            f"Articles en bélarussien traditionnel (be-tarask): "
            f"{stats['traditional_articles']}"
        )

        # 5. Processus de fusion
        if not be_tarask_df.empty:
            # Harmonisation du code de langue pour la variante traditionnelle
            be_tarask_df = be_tarask_df.copy()
            be_tarask_df["language"] = BELARUSIAN_VARIANTS["unified"]

            # Fusion des DataFrames
            merged_be_df = pd.concat([be_df, be_tarask_df], ignore_index=True)

            # Calcul des statistiques post-fusion
            stats.update(
                {
                    "total_articles": len(merged_be_df),
                    "articles_gain": len(merged_be_df) - len(be_df),
                    "fusion_performed": True,
                }
            )

            print("\nFusion réalisée avec succès:")
            print(
                f"Total des articles bélarussiens: "
                f"{stats['total_articles']}"
            )
            print(
                f"Gain par rapport à la variante officielle seule: "
                f"+{stats['articles_gain']} articles"
            )

        else:
            # Cas où seule la variante officielle existe
            merged_be_df = be_df.copy() if not be_df.empty else pd.DataFrame()

            stats.update(
                {
                    "total_articles": len(merged_be_df),
                    "articles_gain": 0,
                    "fusion_performed": False,
                }
            )

            print("\nAucune variante traditionnelle trouvée")
            if not be_df.empty:
                print(
                    f"Conservation de la variante officielle: "
                    f"{stats['total_articles']} articles"
                )
            else:
                print("Aucun article bélarussien disponible")

        # 6. Sauvegarde du résultat
        if not merged_be_df.empty:
            merged_be_df.to_csv(output_path, index=False)
            logger.info(
                f"Corpus bélarussien fusionné sauvegardé: "
                f"{output_path}"
            )
        else:
            logger.warning("Aucun article bélarussien à sauvegarder")

        # 7. Copie des fichiers des autres langues
        print("\nCopie des fichiers des autres langues...")
        copied_files = copy_other_language_files(input_dir, output_dir)
        stats["other_files_copied"] = copied_files

        print(f"Copié {copied_files} fichiers d'autres langues")

        # 8. Finalisation et logging
        logger.info("Fusion des variantes bélarussiennes terminée avec succès")

        print("\n✅ Processus de fusion terminé!")
        print(f"📁 Fichiers fusionnés sauvegardés dans: {output_dir}")

        return merged_be_df, stats

    except Exception as e:
        logger.error(
            f"Erreur lors de la fusion "
            f"des variantes bélarussiennes: {e}"
        )
        raise RuntimeError(f"Échec de la fusion: {e}") from e


# =========================
# POINT D'ENTRÉE PRINCIPAL
# =========================

if __name__ == "__main__":
    """Point d'entrée principal avec gestion d'erreurs robuste

    Exécute la fusion des variantes bélarussiennes avec gestion
    des erreurs et affichage des résultats pour utilisation en standalone.

    Usage:
        python merge_belarusian.py
    """
    try:
        print("Lancement de la fusion des variantes bélarussiennes...")

        # Exécution de la fusion avec paramètres par défaut
        merged_df, statistics = merge_belarusian_variants()

        # Affichage du résumé final
        print("\n📊 Résumé de l'opération:")
        print(f"   • Articles officiels (be): {statistics['official_articles']:,}")
        print(
            f"   • Articles traditionnels (be-tarask): "
            f"{statistics['traditional_articles']:,}"
        )
        print(f"   • Total après fusion: {statistics['total_articles']:,}")

        if statistics["fusion_performed"]:
            print(
                f"   • Gain de diversité: "
                f"+{statistics['articles_gain']:,} articles"
            )
        else:
            print("   • Aucune fusion nécessaire (variante unique)")

        print(
            f"   • Autres langues préservées: "
            f"{statistics['other_files_copied']} fichiers"
        )

    except KeyboardInterrupt:
        print("\n❌ Processus interrompu par l'utilisateur")
        exit(1)
    except (FileNotFoundError, PermissionError) as e:
        print(f"\n❌ Erreur d'accès aux fichiers: {e}")
        print("💡 Vérifiez les chemins et permissions des répertoires")
        exit(1)
    except RuntimeError as e:
        print(f"\n❌ Erreur lors de la fusion: {e}")
        print("💡 Consultez les logs pour plus de détails")
        exit(1)
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        print("💡 Contactez le support technique si le problème persiste")
        exit(1)
