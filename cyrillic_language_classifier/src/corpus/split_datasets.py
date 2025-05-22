"""Module de division stratifiée de corpus multilingues pour l'apprentissage automatique

Ce module fournit des outils spécialisés pour diviser intelligemment un corpus
multilingue en ensembles d'entraînement, validation et test. Il prend en compte
certaines spécificités des langues cyrilliques et les déséquilibres naturels des corpus.

Le module implémente une stratégie de division adaptative qui:
    * préserve la représentativité linguistique dans chaque ensemble
    * gère les langues avec peu d'exemples par des stratégies spéciales
    * maintient l'équilibre entre données originales et augmentées
    * calcule des métriques de qualité de la division (entropie, statistiques)

Architecture de division:
    La division respecte une hiérarchie langue > source > échantillonnage aléatoire
    pour garantir que chaque ensemble contient une représentation équitable de
    toutes les langues et sources de données disponibles.
    Les langues minoritaires bénéficient de stratégies de division adaptées
    pour maximiser leur utilité tout en maintenant la validité statistique
    des ensembles créés.

Gestion des cas particuliers :
    Le module traite les situations où certaines langues ont très peu d'exemples,
    en adaptant les ratios de division et en appliquant des seuils minimaux
    pour maintenir la cohérence méthodologique.

Sortie standardisée :
    Les ensembles générés respectent les conventions d'organisation des données
    pour l'apprentissage automatique, avec des répertoires séparés et des
    statistiques détaillées pour faciliter l'évaluation de la qualité de la division.
"""


import os
import pandas as pd
import numpy as np


# ========================================================
# CONSTANTES DE CONFIGURATION POUR LA DIVISION DU CORPUS
# ========================================================

# Ratios par défaut pour la division train/validation/test
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VAL_RATIO = 0.1
DEFAULT_TEST_RATIO = 0.1

# Seuils pour la gestion des petits datasets
MIN_EXAMPLES_FOR_NORMAL_SPLIT = 10  # assez d'exs pour une division significative
MIN_EXAMPLES_FOR_ANY_SPLIT = 3  # minimum pour diviser en plusieurs ensembles
MIN_EXAMPLES_FOR_DUAL_SPLIT = 2  # minimum pour diviser en 2 ensembles

# Paramètres de validation
RATIO_SUM_TOLERANCE = 1e-10  # tolérance pour vérifier que les ratios somment à 1

# Mapping des variants de langues pour harmonisation
LANGUAGE_HARMONIZATION = {"be-tarask": "be"}  # harmoniser les variantes du bélarussien

# Colonnes essentielles du corpus (dans l'ordre de priorité)
ESSENTIAL_COLUMNS = ["language", "title", "text"]

# Paramètres par défaut des répertoires
DEFAULT_MERGED_DIR = "data/processed/merged"
DEFAULT_AUGMENTED_DIR = "data/processed/augmented"
DEFAULT_OUTPUT_DIR = "data/final"

# Noms des sous-répertoires de sortie
OUTPUT_SUBDIRS = {"train": "train", "validation": "validation", "test": "test"}

# Noms des fichiers de sortie
OUTPUT_FILENAMES = {
    "train": "train_corpus.csv",
    "validation": "validation_corpus.csv",
    "test": "test_corpus.csv",
}

# Paramètres pour les statistiques
STATS_PRECISION = 1  # nb de décimales pour les pourcentages
ENTROPY_PRECISION = 4  # nb de décimales pour l'entropie

# Messages d'erreur standardisés
ERROR_MESSAGES = {
    "ratio_sum": "Les ratios doivent sommer à 1",
    "no_data": "Aucune donnée disponible",
    "empty_output": "Ensemble vide",
}


# =======================
# FONCTIONS UTILITAIRES
# =======================

def validate_split_ratios(train_ratio, val_ratio, test_ratio):
    """Valide que les ratios de division sont cohérents

    Args:
        train_ratio (float): proportion de l'ensemble d'entraînement
        val_ratio (float): proportion de l'ensemble de validation
        test_ratio (float): proportion de l'ensemble de test

    Raises:
        ValueError: si les ratios ne somment pas à 1 ou sont invalides
    """
    # Vérifier que tous les ratios sont des nombres positifs
    ratios = [train_ratio, val_ratio, test_ratio]
    ratio_names = ["train_ratio", "val_ratio", "test_ratio"]

    for ratio, name in zip(ratios, ratio_names):
        if not isinstance(ratio, (int, float)) or ratio < 0:
            raise ValueError(f"{name} doit être un nombre positif, ici: {ratio}")
        if ratio > 1:
            raise ValueError(f"{name} ne peut pas être supérieur à 1, ici: {ratio}")

    # Vérifier que les ratios somment à 1
    ratio_sum = sum(ratios)
    if abs(ratio_sum - 1.0) > RATIO_SUM_TOLERANCE:
        raise ValueError(f"{ERROR_MESSAGES['ratio_sum']}. Somme actuelle: {ratio_sum}")


def validate_directories(merged_dir, augmented_dir, output_dir):
    """Valide l'existence des répertoires d'entrée et crée le répertoire de sortie

    Args:
        merged_dir (str): dossier des données fusionnées
        augmented_dir (str): dossier des données augmentées
        output_dir (str): dossier de sortie

    Raises:
        FileNotFoundError: si un dossier d'entrée n'existe pas
        PermissionError: si impossible de créer le dossier de sortie
    """
    # Vérifier l'existence des dossiers d'entrée
    for directory, name in [
        (merged_dir, "merged_dir"),
        (augmented_dir, "augmented_dir"),
    ]:
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Le répertoire {name} '{directory}' n'existe pas")
        if not os.access(directory, os.R_OK):
            raise PermissionError(
                f"Pas d'autorisation de lecture sur {name} '{directory}'"
            )

    # Créer le dossier de sortie et ses sous-dossiers
    try:
        for subdir_name in OUTPUT_SUBDIRS.values():
            subdir_path = os.path.join(output_dir, subdir_name)
            os.makedirs(subdir_path, exist_ok=True)
    except PermissionError:
        raise PermissionError(
            f"Impossible de créer le répertoire de sortie '{output_dir}'"
        )


def validate_dataframe_compatibility(dataframes_list, source_name):
    """Valide qu'une liste de DataFrames peut être fusionnée

    Args:
        dataframes_list (list): liste de DataFrames à valider
        source_name (str): nom de la source pour les messages d'erreur

    Raises:
        ValueError: si les DataFrames ne peuvent pas être fusionnés

    Returns:
        bool: True si la validation réussit
    """
    if not dataframes_list:
        raise ValueError(f"Aucun DataFrame trouvé pour {source_name}")

    # Vérifier que tous les éléments sont des DataFrames
    for i, df in enumerate(dataframes_list):
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"L'élément {i} de {source_name} n'est pas un DataFrame")

        if df.empty:
            print(
                f"Avertissement: DataFrame vide trouvé dans {source_name} (index {i})"
            )

    return True


def harmonize_language_codes(dataframe):
    """Harmonise les codes de langue selon les règles prédéfinies

    Args:
        dataframe (pd.DataFrame): DataFrame contenant une colonne 'language'

    Returns:
        pd.DataFrame: DataFrame avec codes de langue harmonisés
    """
    if "language" not in dataframe.columns:
        return dataframe

    # Appliquer les harmonisations
    df_copy = dataframe.copy()
    for old_code, new_code in LANGUAGE_HARMONIZATION.items():
        df_copy["language"] = df_copy["language"].replace(old_code, new_code)


def calculate_entropy(df, col="language"):
    """
    Calcule l'entropie de la distribution des valeurs dans une colonne.

    L'entropie mesure l'équilibre d'une distribution : une entropie plus élevée
    indique une distribution plus équilibrée entre les différentes valeurs.

    Args:
        df (pd.DataFrame): DataFrame à analyser
        col (str): nom de la colonne à analyser (défaut: 'language')

    Returns:
        float: entropie en bits (0.0 si DataFrame vide)

    Note:
        Une entropie de 0 indique qu'une seule valeur est présente.
        L'entropie maximale dépend du nombre de valeurs uniques.
    """
    if len(df) == 0:
        return 0.0

    counts = df[col].value_counts()
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy


# ====================
# FONCTION PRINCIPALE
# ====================


def split_datasets(
    merged_dir=DEFAULT_MERGED_DIR,
    augmented_dir=DEFAULT_AUGMENTED_DIR,
    output_dir=DEFAULT_OUTPUT_DIR,
    train_ratio=DEFAULT_TRAIN_RATIO,
    val_ratio=DEFAULT_VAL_RATIO,
    test_ratio=DEFAULT_TEST_RATIO,
):
    """
    Divise le corpus final en ensembles d'entraînement, validation et test.

    Args:
        merged_dir (str): dossier contenant les fichiers originaux nettoyés
        augmented_dir (str): dossier contenant les fichiers augmentés
        output_dir (str): dossier où sauvegarder les ensembles
        train_ratio (float): proportion de l'ensemble d'entraînement
        val_ratio (float): proportion de l'ensemble de validation
        test_ratio (float): proportion de l'ensemble de test

    Returns:
        tuple: (train_df, val_df, test_df) - les 3 DataFrames créés

    Raises:
        ValueError: si les paramètres sont invalides
        FileNotFoundError: si les dossiers d'entrée n'existent pas
        PermissionError: si impossible d'écrire dans le dossier de sortie
    """
    try:
        # 1. Validation des paramètres d'entrée
        print("Validation des paramètres...")
        validate_split_ratios(train_ratio, val_ratio, test_ratio)
        validate_directories(merged_dir, augmented_dir, output_dir)

        print("=== Début de la division du corpus ===")

        # 2. Charger les articles originaux
        print("Chargement des articles originaux...")
        original_articles = []

        try:
            for file in os.listdir(merged_dir):
                if file.endswith(".csv"):
                    file_path = os.path.join(merged_dir, file)
                    df = pd.read_csv(file_path)

                    # S'assurer que toutes les colonnes nécessaires existent
                    # et sont dans le bon ordre pour les articles originaux
                    if "language" not in df.columns and "page_id" in df.columns:
                        # Renommer page_id en pageid pour cohérence si nécessaire
                        if "pageid" not in df.columns and "page_id" in df.columns:
                            df = df.rename(columns={"page_id": "pageid"})

                    # Ajouter une colonne source si elle n'existe pas
                    if "source" not in df.columns:
                        df["source"] = "original"  # marquer comme original

                    original_articles.append(df)

        except Exception as e:
            raise FileNotFoundError(
                f"Erreur lors du chargement depuis {merged_dir}: {e}"
            )

        # 3. Charger les articles augmentés
        print("Chargement des articles augmentés...")
        augmented_articles = []

        try:
            for file in os.listdir(augmented_dir):
                if file.endswith(".csv"):
                    file_path = os.path.join(augmented_dir, file)
                    df = pd.read_csv(file_path)

                    # S'assurer que toutes les colonnes nécessaires existent
                    if "pageid" not in df.columns:
                        df["pageid"] = None  # ajouter une colonne pageid vide

                    augmented_articles.append(df)

        except Exception as e:
            print(
                f"Avertissement: erreur lors du chargement depuis {augmented_dir} - {e}"
            )

        # 4. Fusionner tous les articles
        all_dfs = original_articles + augmented_articles
        if all_dfs:
            # Identifier toutes les colonnes uniques dans tous les DataFrames
            all_columns = set()
            for df in all_dfs:
                all_columns.update(df.columns)

            # S'assurer que chaque DataFrame a toutes les colonnes
            standardized_dfs = []
            for df in all_dfs:
                # Ajouter les colonnes manquantes avec des valeurs None
                for col in all_columns:
                    if col not in df.columns:
                        df[col] = None
                standardized_dfs.append(df)

            # Fusionner les DataFrames standardisés
            full_corpus = pd.concat(standardized_dfs, ignore_index=True)

            # Harmoniser les variantes du bélarussien
            full_corpus = harmonize_language_codes(full_corpus)

            # Réorganiser les colonnes
            # pour mettre 'language', 'title', 'text' en premier
            other_columns = [
                col for col in full_corpus.columns
                if col not in ESSENTIAL_COLUMNS
            ]
            full_corpus = full_corpus[ESSENTIAL_COLUMNS + other_columns]
        else:
            full_corpus = pd.DataFrame()

        # 5. Diviser directement par langue
        # pour obtenir une distribution équilibrée
        train_dfs = []
        val_dfs = []
        test_dfs = []

        for language in full_corpus["language"].unique():
            lang_df = full_corpus[full_corpus["language"] == language]

            # Pour chaque langue, stratifier par source si possible
            if "source" in lang_df.columns and len(lang_df["source"].unique()) > 1:
                # Grouper par source pour assurer une distribution équilibrée
                for source in lang_df["source"].unique():
                    source_df = lang_df[lang_df["source"] == source]

                    # Si on a assez d'exemples, diviser normalement
                    if len(source_df) >= MIN_EXAMPLES_FOR_NORMAL_SPLIT:
                        train_size = int(len(source_df) * train_ratio)
                        val_size = int(len(source_df) * val_ratio)

                        # Mélanger les données
                        shuffled_df = source_df.sample(
                            frac=1, random_state=42
                        ).reset_index(drop=True)

                        # Diviser
                        train_part = shuffled_df.iloc[:train_size]
                        val_part = shuffled_df.iloc[train_size:train_size+val_size]
                        test_part = shuffled_df.iloc[train_size+val_size:]

                        train_dfs.append(train_part)
                        val_dfs.append(val_part)
                        test_dfs.append(test_part)

                    elif len(source_df) >= MIN_EXAMPLES_FOR_ANY_SPLIT:
                        train_size = max(1, int(len(source_df) * train_ratio))
                        val_size = max(1, int(len(source_df) * val_ratio))

                        # Mélanger les données
                        shuffled_df = source_df.sample(
                            frac=1, random_state=42
                        ).reset_index(drop=True)

                        # Diviser
                        train_part = shuffled_df.iloc[:train_size]
                        val_part = shuffled_df.iloc[train_size:train_size+val_size]
                        test_part = shuffled_df.iloc[train_size+val_size:]

                        train_dfs.append(train_part)
                        if len(val_part) > 0:
                            val_dfs.append(val_part)
                        if len(test_part) > 0:
                            test_dfs.append(test_part)

                    else:  # très peu d'exemples, allouer à l'entraînement principalement
                        if len(source_df) == MIN_EXAMPLES_FOR_DUAL_SPLIT:
                            train_dfs.append(source_df.iloc[:1])
                            val_dfs.append(source_df.iloc[1:])
                        else:  # 1 seul exemple
                            train_dfs.append(source_df)
            else:
                # Si pas de colonne source ou une seule source,
                # diviser directement la langue
                if len(lang_df) >= MIN_EXAMPLES_FOR_NORMAL_SPLIT:
                    train_size = int(len(lang_df) * train_ratio)
                    val_size = int(len(lang_df) * val_ratio)

                    # Mélanger les données
                    shuffled_df = lang_df.sample(
                        frac=1, random_state=42
                        ).reset_index(drop=True)

                    # Diviser
                    train_part = shuffled_df.iloc[:train_size]
                    val_part = shuffled_df.iloc[train_size:train_size+val_size]
                    test_part = shuffled_df.iloc[train_size+val_size:]

                    train_dfs.append(train_part)
                    val_dfs.append(val_part)
                    test_dfs.append(test_part)

                elif len(lang_df) >= MIN_EXAMPLES_FOR_ANY_SPLIT:
                    train_size = max(1, int(len(lang_df) * train_ratio))
                    val_size = max(1, int(len(lang_df) * val_ratio))

                    # Mélanger les données
                    shuffled_df = lang_df.sample(
                        frac=1, random_state=42
                        ).reset_index(drop=True)

                    # Diviser
                    train_part = shuffled_df.iloc[:train_size]
                    val_part = shuffled_df.iloc[train_size:train_size+val_size]
                    test_part = shuffled_df.iloc[train_size+val_size:]

                    train_dfs.append(train_part)
                    if len(val_part) > 0:
                        val_dfs.append(val_part)
                    if len(test_part) > 0:
                        test_dfs.append(test_part)

                else:
                    if len(lang_df) == MIN_EXAMPLES_FOR_DUAL_SPLIT:
                        train_dfs.append(lang_df.iloc[:1])
                        val_dfs.append(lang_df.iloc[1:])
                    else:
                        train_dfs.append(lang_df)

        # 6. Combiner les ensembles
        train_df = (
            pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame()
        )
        val_df = pd.concat(val_dfs, ignore_index=True) if val_dfs else pd.DataFrame()
        test_df = pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame()

        # 7. Sauvegarder les ensembles
        train_df.to_csv(
            os.path.join(
                output_dir, OUTPUT_SUBDIRS["train"], OUTPUT_FILENAMES["train"]
            ),
            index=False,
        )
        val_df.to_csv(
            os.path.join(
                output_dir,
                OUTPUT_SUBDIRS["validation"],
                OUTPUT_FILENAMES["validation"]
            ),
            index=False,
        )
        test_df.to_csv(
            os.path.join(
                output_dir,
                OUTPUT_SUBDIRS["test"],
                OUTPUT_FILENAMES["test"]
            ),
            index=False,
        )

        # 8. Afficher les statistiques
        print("\nDivision du corpus en ensembles:")
        print(f"Corpus complet: {len(full_corpus)} articles")
        print(
            f"Ensemble d'entraînement: {len(train_df)} articles "
            f"({len(train_df)/len(full_corpus)*100:.{STATS_PRECISION}f}%)"
        )
        print(
            f"Ensemble de validation: {len(val_df)} articles "
            f"({len(val_df)/len(full_corpus)*100:.{STATS_PRECISION}f}%)"
        )
        print(
            f"Ensemble de test: {len(test_df)} articles "
            f"({len(test_df)/len(full_corpus)*100:.{STATS_PRECISION}f}%)"
        )

        # Statistiques par langue
        print("\nDistribution des langues dans chaque ensemble:")
        print("\nEntraînement:")
        print(train_df["language"].value_counts())
        print("\nValidation:")
        print(val_df["language"].value_counts() if len(val_df) > 0 else "Ensemble vide")
        print("\nTest:")
        print(
            test_df["language"].value_counts() if len(test_df) > 0 else "Ensemble vide"
        )

        # Calculer les statistiques de longueur
        # pour vérifier la représentativité
        if "token_count" in train_df.columns and len(train_df) > 0:
            train_length_stats = train_df["token_count"].describe()

            print("\nStatistiques de longueur des textes (en tokens) par ensemble:")
            print(
                f"Entraînement: moyenne={train_length_stats['mean']:.1f}, "
                f"médiane={train_length_stats['50%']:.1f}, "
                f"min={train_length_stats['min']:.1f}, "
                f"max={train_length_stats['max']:.1f}"
            )

            if "token_count" in val_df.columns and len(val_df) > 0:
                val_length_stats = val_df["token_count"].describe()
                print(
                    f"Validation: moyenne={val_length_stats['mean']:.1f}, "
                    f"médiane={val_length_stats['50%']:.1f}, "
                    f"min={val_length_stats['min']:.1f}, "
                    f"max={val_length_stats['max']:.1f}"
                )
            else:
                print("Validation: pas de statistiques disponibles")

            if "token_count" in test_df.columns and len(test_df) > 0:
                test_length_stats = test_df["token_count"].describe()
                print(
                    f"Test: moyenne={test_length_stats['mean']:.1f}, "
                    f"médiane={test_length_stats['50%']:.1f}, "
                    f"min={test_length_stats['min']:.1f}, "
                    f"max={test_length_stats['max']:.1f}"
                )
            else:
                print("Test: pas de statistiques disponibles")

        # Vérifier la distribution des méthodes d'augmentation si applicable
        if "source" in train_df.columns:
            print("\nDistribution des méthodes d'augmentation:")
            print("Entraînement:")
            print(train_df["source"].value_counts())

            print("\nValidation:")
            if "source" in val_df.columns and len(val_df) > 0:
                print(val_df["source"].value_counts())
            else:
                print("Pas de données disponibles")

            print("\nTest:")
            if "source" in test_df.columns and len(test_df) > 0:
                print(test_df["source"].value_counts())
            else:
                print("Pas de données disponibles")

            print("=== Division du corpus terminée avec succès ===")
            return train_df, val_df, test_df

        # Calcul de l'entropie (utilise notre fonction utilitaire)
        train_entropy = calculate_entropy(train_df) if len(train_df) > 0 else 0.0
        val_entropy = calculate_entropy(val_df) if len(val_df) > 0 else 0.0
        test_entropy = calculate_entropy(test_df) if len(test_df) > 0 else 0.0

        print("\nEntropie de la distribution des langues (mesure d'équilibre):")
        print(f"Entraînement: {train_entropy:.{ENTROPY_PRECISION}f} bits")
        print(f"Validation: {val_entropy:.{ENTROPY_PRECISION}f} bits")
        print(f"Test: {test_entropy:.{ENTROPY_PRECISION}f} bits")

        print("=== Division du corpus terminée avec succès ===")
        return train_df, val_df, test_df

    except Exception as e:
        print(f"Erreur lors de la division du corpus: {e}")
        raise


# Point d'entrée principal du script
if __name__ == "__main__":
    """Exécute la division avec les paramètres par défaut"""
    try:
        print("Lancement de la division du corpus...")

        train_df, val_df, test_df = split_datasets()

        print("\n✅ Division terminée avec succès!")
        print(
            f"Train: {len(train_df)} | Validation: {len(val_df)} | Test: {len(test_df)}"
        )
        print(f"📁 Résultats sauvegardés dans: {DEFAULT_OUTPUT_DIR}")

    except (FileNotFoundError, PermissionError) as e:
        print(f"\n❌ Erreur d'accès aux fichiers: {e}")
        exit(1)
    except ValueError as e:
        print(f"\n❌ Erreur de paramètres: {e}")
        exit(1)
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        exit(1)
