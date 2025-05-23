"""Module de visualisation et d'analyse statistique pour corpus multilingues cyrilliques

Ce module fournit un ensemble d'outils pour analyser et visualiser
les caractéristiques d'un corpus de textes en langues cyrilliques.
Il implémente plusieurs analyses linguistiques et statistiques.

Analyses principales:
    * Distribution des longueurs de textes et statistiques descriptives
    * Classification et analyse par groupes linguistiques (familles de langues)
    * Analyse de la loi de Zipf pour étudier les distributions de fréquence des mots
    * Identification des caractères cyrilliques distinctifs par langue
    * Visualisations comparatives multi-langues

Exemples d'utilisation :
    Analyse complète d'un corpus:
        >>> from visualize_corpus import main
        >>> main()  # Génère toutes les visualisations
    Analyse ciblée:
        >>> from visualize_corpus import explore_corpus_stats, load_all_articles
        >>> articles = load_all_articles()
        >>> explore_corpus_stats(articles, output_dir="mes_resultats")

Organisation des données:
    Le module attend des fichiers CSV dans le format suivant:
    - fichiers nommés "{code_langue}_articles.csv"
    - colonnes requises: 'language', 'text', 'token_count'
    - colonnes optionnelles: 'title', 'category', 'source'

Sortie:
    Toutes les visualisations sont sauvegardées dans le dossier spécifié
    avec une résolution haute (300 DPI) adaptée.

Note méthodologique :
    Les analyses prennent en compte les différences
    de richesse des corpus selon les langues.
    La classification par groupes suit une approche linguistique basée sur
    les familles de langues et la disponibilité des ressources numériques.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import re
from collections import Counter
from matplotlib.ticker import FuncFormatter

# Import du module de configuration
from corpus.modules.config import get_language_group


# ============================
# CONSTANTES DE CONFIGURATION
# ============================

# Langues pour analyse approfondie
SAMPLE_LANGUAGES = [
    "ru",
    "uk",
    "be",
    "bg",
    "mk",
    "rue",
    "sr",
    "kk",
    "mn",
    "tt",
    "tg",
    "bxr",
    "ce",
    "koi",
    "mhr",
]

# Paramètres de visualisation - tailles de figures
FIGURE_SIZES = {
    "standard": (12, 8),  # taille par défaut pour la plupart des graphiques
    "wide": (14, 8),  # pour les graphiques nécessitant plus de largeur
    "extra_wide": (16, 10),  # pour les graphiques avec beaucoup de données
    "tall": (12, 10),  # pour les graphiques nécessitant plus de hauteur
    "large": (20, 20),  # pour les heatmaps complexes
    "wordcloud": (16, 8),  # spécifique aux nuages de mots
}

# Paramètres de qualité des images
IMAGE_SETTINGS = {
    "dpi": 300,  # résolution pour les images sauvegardées
    "format": "png",  # format par défaut des images
    "bbox_inches": "tight",  # ajustement automatique des marges
}

# Paramètres d'analyse statistique
ANALYSIS_PARAMS = {
    "histogram_bins": 50,  # nb de bins pour les histogrammes
    "top_languages": 10,  # nb de langues à afficher dans les tops
    "top_categories": 10,  # nb de catégories à afficher dans les tops
    "max_words_zipf": 50,  # nb de mots pour l'analyse de Zipf
    "max_words_cloud": 100,  # nb de mots dans les nuages
    "max_chars_distinctive": 26,  # nb max de caractères distinctifs à analyser
    "min_articles_for_analysis": 5,  # minimum d'articles pour inclure une langue
    "min_articles_for_zipf": 10,  # minimum d'articles pour l'analyse de Zipf
}

# Configuration des couleurs et palettes
COLOR_SCHEMES = {
    "primary": "viridis",  # palette principale
    "secondary": "Set2",  # palette secondaire pour catégories
    "heatmap": "viridis",  # palette pour heatmaps
    "comparative": "deep",  # palette pour comparaisons
}

# Paramètres de style matplotlib
MATPLOTLIB_STYLE = {
    "style": "seaborn-v0_8-whitegrid",  # style de base
    "font_family": "DejaVu Sans",  # police pour caractères cyrilliques
    "default_font_size": 12,  # taille de police par défaut
}

# Répertoires par défaut
DEFAULT_PATHS = {
    "input_pattern": "data/processed/merged/*_articles.csv",
    "output_base": "results/figures/distribution",
}

# Messages et titres standardisés
PLOT_TITLES = {
    "token_distribution": "Distribution des longueurs d'articles (tokens)",
    "top_languages": "Top {} des langues par nombre d'articles",
    "top_categories": "Top {} des catégories thématiques",
    "tokens_per_language": "Distribution des tokens par article pour chaque langue",
    "zipf_analysis": "Loi de Zipf pour la langue {}",
    "zipf_comparison": "Comparaison de la loi de Zipf entre les langues",
    "distinctive_chars": "Caractères cyrilliques distinctifs par langue",
}


# ======================
# FONCTIONS UTILITAIRES
# ======================

def setup_matplotlib_config():
    """Configure matplotlib

    Cette fonction centralise toute la configuration matplotlib pour assurer
    une cohérence visuelle dans toutes les visualisations du projet.
    Elle configure la police DejaVu Sans, qui prend correctement en charge
    les caractères cyrilliques.

    Configurations appliquées:
        - style seaborn pour les graphiques
        - police DejaVu Sans pour la prise en charge du cyrillique
        - taille de figure par défaut
        - taille de police lisible pour les visualisations

    Note:
        Cette fonction doit être appelée avant toute création de graphique
        pour garantir un rendu correct des textes en langues cyrilliques.
    """
    plt.style.use(MATPLOTLIB_STYLE["style"])
    plt.rcParams["font.family"] = MATPLOTLIB_STYLE["font_family"]
    plt.rcParams["figure.figsize"] = FIGURE_SIZES["standard"]
    plt.rcParams["font.size"] = MATPLOTLIB_STYLE["default_font_size"]


def save_plot(filename, output_dir, size_key="standard"):
    """Sauvegarde un graphique avec les paramètres standardisés

    Cette fonction encapsule la logique de sauvegarde pour éviter la répétition
    de code et assurer une qualité uniforme des images produites.

    Args:
        filename (str): nom du fichier (sans extension)
        output_dir (str): dossier de destination
        size_key (str): clé du dictionnaire FIGURE_SIZES à utiliser
            (par défaut: 'standard')

    Note sur la qualité:
        Les images sont sauvegardées en 300 DPI, ce qui assure une qualité optimale
        pour l'intégration dans des documents.

    Gestion mémoire:
        La fonction ferme automatiquement les graphiques après sauvegarde
        pour éviter l'accumulation en mémoire lors de la génération
        de nombreuses visualisations.
    """
    filepath = f"{output_dir}/{filename}.{IMAGE_SETTINGS['format']}"
    plt.savefig(
        filepath, dpi=IMAGE_SETTINGS["dpi"], bbox_inches=IMAGE_SETTINGS["bbox_inches"]
    )
    plt.close()  # libérer la mémoire


def validate_output_directory(output_dir):
    """Valide et crée le dossier de sortie si nécessaire

    Args:
        output_dir (str): chemin du dossier de sortie

    Raises:
        PermissionError: si impossible de créer le dossier
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
    except PermissionError:
        raise PermissionError(
            f"Impossible de créer le dossier de sortie '{output_dir}'"
        )


def filter_languages_for_analysis(articles_df, min_articles=None):
    """Implémente un seuil de qualité pour les analyses statistiques.

    Cette fonction élimine les langues avec trop peu d'articles
    pour éviter des conclusions statistiquement non significatives,
    tout en préservant la diversité linguistique du corpus.

    Rationale méthodologique:
        Les analyses statistiques (loi de Zipf, distribution des caractères, etc.)
        nécessitent un volume minimal de données pour être significatives. Cette
        fonction applique ce principe en filtrant intelligemment les langues.

    Args:
        articles_df (pd.DataFrame): DataFrame contenant les articles du corpus
        min_articles (int, optional): seuil minimum d'articles requis
            (si None, utilise la valeur par défaut de ANALYSIS_PARAMS)

    Returns:
        list: liste des codes de langues validés pour l'analyse

    Note sur les seuils:
        Le seuil par défaut (5 articles) représente un compromis entre
        significativité statistique et préservation de la diversité linguistique.
        Pour l'analyse de Zipf, un seuil plus élevé (10 articles) est recommandé.
    """
    if min_articles is None:
        min_articles = ANALYSIS_PARAMS["min_articles_for_analysis"]

    language_counts = articles_df["language"].value_counts()
    return language_counts[language_counts >= min_articles].index.tolist()


# ==================================
# FONCTION PRINCIPALE DE CHARGEMENT
# ==================================

def load_all_articles():
    """Charge tous les articles depuis les fichiers CSV avec validation robuste

    Cette fonction constitue le point d'entrée principal pour l'accès aux données.
    Elle implémente une logique robuste de chargement qui gère les erreurs de
    format, valide la cohérence des données, et assure la compatibilité entre
    différents formats de fichiers.

    Processus de validation:
        1. Détection automatique des fichiers selon le pattern de nommage
        2. Extraction des codes de langue depuis les noms de fichiers
        3. Validation de la présence des colonnes essentielles
        4. Nettoyage des articles vides ou corrompus
        5. Harmonisation des schémas de données

    Format attendu des fichiers:
        - nom: "{code_langue}_articles.csv"
        - colonnes requises: 'text', plus optionnellement 'language', 'token_count'
        - encodage: UTF-8 (crucial pour les caractères cyrilliques)

    Returns:
        pd.DataFrame: DataFrame unifié contenant tous les articles valides
            avec colonnes standardisées - 'language', 'text', 'token_count', etc.

    Raises:
        FileNotFoundError: si aucun fichier correspondant au pattern n'est trouvé
        ValueError: si aucun fichier CSV valide n'a pu être chargé

    Note:
        La fonction continue le traitement même si certains fichiers sont
        corrompus, en affichant des avertissements informatifs. Cette approche
        permet de travailler avec des corpus partiellement incomplets.
    """
    pattern = DEFAULT_PATHS["input_pattern"]
    all_files = glob.glob(pattern)

    if not all_files:
        raise FileNotFoundError(f"Aucun fichier trouvé avec le pattern: {pattern}")

    dataframes = []
    for file in all_files:
        try:
            # Extraire le code de langue du nom de fichier
            lang_code = os.path.basename(file).split("_")[0]

            df = pd.read_csv(file)

            # Valider les colonnes essentielles
            if "text" not in df.columns:
                print(f"Attention: Colonne 'text' manquante dans {file}")
                continue

            # Ajouter le code de langue si manquant
            if "language" not in df.columns:
                df["language"] = lang_code

            # Filtrer les articles vides
            df = df[df["text"].notna() & (df["text"] != "")]

            if len(df) > 0:
                dataframes.append(df)
                print(f"Chargé {len(df)} articles pour {lang_code}")
            else:
                print(f"Aucun article valide dans {file}")

        except Exception as e:
            print(f"Erreur lors du chargement de {file}: {e}")
            continue

    if not dataframes:
        raise ValueError("Aucun fichier CSV valide n'a pu être chargé")

    combined_df = pd.concat(dataframes, ignore_index=True)
    print(
        f"Total: {len(combined_df)} articles, {combined_df['language'].nunique()} langues"
    )

    return combined_df


# =====================
# FONCTIONS D'ANALYSE
# =====================

def explore_corpus_stats(articles_df, output_dir=DEFAULT_PATHS["output_base"]):
    """Analyse statistique générale du corpus avec paramètres standardisés"""
    validate_output_directory(output_dir)

    print("=== Statistiques globales du corpus ===")
    print(f"Nombre total d'articles: {len(articles_df)}")
    print(f"Nombre total de tokens: {articles_df['token_count'].sum():,}")
    print(f"Nombre de langues: {articles_df['language'].nunique()}")

    # Distribution des tokens par article
    plt.figure(figsize=FIGURE_SIZES["standard"])
    sns.histplot(
        articles_df["token_count"], bins=ANALYSIS_PARAMS["histogram_bins"], kde=True
    )
    plt.title(PLOT_TITLES["token_distribution"])
    plt.xlabel("Nombre de tokens")
    plt.ylabel("Nombre d'articles")
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    save_plot("token_distribution", output_dir)

    # Top des langues
    plt.figure(figsize=FIGURE_SIZES["wide"])
    top_langs = (
        articles_df["language"].value_counts().head(ANALYSIS_PARAMS["top_languages"])
    )
    sns.barplot(
        x=top_langs.index,
        y=top_langs.values,
        hue=top_langs.index,
        palette=COLOR_SCHEMES["primary"],
        legend=False,
    )
    plt.title(PLOT_TITLES["top_languages"].format(ANALYSIS_PARAMS["top_languages"]))
    plt.xlabel("Langue")
    plt.ylabel("Nombre d'articles")
    save_plot("top10_languages_by_articles", output_dir)

    # Distribution des catégories
    articles_df["main_category"] = articles_df["category"].apply(
        lambda x: str(x).split(" (")[0] if isinstance(x, str) and " (" in x else x
    )
    top_categories = (
        articles_df["main_category"]
        .value_counts()
        .head(ANALYSIS_PARAMS["top_categories"])
    )

    plt.figure(figsize=FIGURE_SIZES["wide"])
    sns.barplot(
        x=top_categories.index,
        y=top_categories.values,
        hue=top_categories.index,
        palette=COLOR_SCHEMES["secondary"],
        legend=False,
    )
    plt.title(PLOT_TITLES["top_categories"].format(ANALYSIS_PARAMS["top_categories"]))
    plt.xlabel("Catégorie")
    plt.ylabel("Nombre d'articles")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    save_plot("top10_categories", output_dir)

    # Distribution des tokens par langue
    plt.figure(figsize=FIGURE_SIZES["extra_wide"])
    sns.boxplot(
        x="language", y="token_count", data=articles_df.sort_values(by="language")
    )
    plt.title(PLOT_TITLES["tokens_per_language"])
    plt.xlabel("Langue")
    plt.ylabel("Nombre de tokens")
    plt.xticks(rotation=90)
    plt.tight_layout()
    save_plot("tokens_per_language_boxplot", output_dir)

    return "Exploration des statistiques du corpus terminée!"


def analyze_text_characteristics(articles_df, output_dir=DEFAULT_PATHS["output_base"]):
    """Analyse des caractéristiques textuelles avec groupes de langues"""
    validate_output_directory(output_dir)

    # Ajouter une colonne pour le groupe de langue
    articles_df["language_group"] = articles_df["language"].apply(get_language_group)

    # Créer un boxplot par groupe
    plt.figure(figsize=FIGURE_SIZES["wide"])
    sns.boxplot(
        x="language_group",
        y="token_count",
        data=articles_df,
        order=["A", "B", "C", "D"],
    )
    plt.title("Distribution des longueurs d'articles par groupe de langue")
    plt.xlabel("Groupe de langue")
    plt.ylabel("Nombre de tokens")
    save_plot("token_distribution_by_group", output_dir)

    return "Analyse des caractéristiques textuelles terminée!"


def analyze_zipf_law(articles_df, output_dir=DEFAULT_PATHS["output_base"]):
    """Analyse de la loi de Zipf pour les langues principales"""
    validate_output_directory(output_dir)

    # Filtrer les langues avec suffisamment d'articles
    valid_languages = filter_languages_for_analysis(
        articles_df, ANALYSIS_PARAMS["min_articles_for_zipf"]
    )

    # Intersection avec les langues d'échantillon
    languages_to_analyze = [
        lang for lang in SAMPLE_LANGUAGES if lang in valid_languages
    ]

    print(f"Analyse de la loi de Zipf pour {len(languages_to_analyze)} langues")

    for lang in languages_to_analyze:
        try:
            # Filtrer les articles pour cette langue
            lang_df = articles_df[articles_df["language"] == lang]

            # Concaténer tous les textes
            all_text = " ".join(lang_df["text"].fillna(""))

            # Tokeniser simplement par espaces
            words = re.findall(r"\b\w+\b", all_text.lower())

            # Compter les fréquences
            word_counts = Counter(words)
            most_common = word_counts.most_common(ANALYSIS_PARAMS["max_words_zipf"])

            # Créer un DataFrame pour le graphique
            zipf_df = pd.DataFrame(most_common, columns=["word", "frequency"])
            zipf_df["rank"] = range(1, len(zipf_df) + 1)

            # Log-log plot détaillé
            plt.figure(figsize=FIGURE_SIZES["wide"])

            # Données observées
            plt.loglog(
                zipf_df["rank"],
                zipf_df["frequency"],
                "o",
                markersize=5,
                label="Données observées",
                alpha=0.7,
            )

            # Loi de Zipf théorique
            ideal_zipf = zipf_df["frequency"].iloc[0] / zipf_df["rank"]
            plt.loglog(
                zipf_df["rank"],
                ideal_zipf,
                "r-",
                linewidth=2,
                label="Loi de Zipf idéale (1/rang)",
            )

            plt.title(PLOT_TITLES["zipf_analysis"].format(lang))
            plt.xlabel("Rang (log)")
            plt.ylabel("Fréquence (log)")
            plt.legend()
            plt.grid(True, which="both", ls="-", alpha=0.3)

            # Annotations pour les mots les plus fréquents
            for i in range(min(5, len(zipf_df))):
                plt.annotate(
                    zipf_df["word"][i],
                    (zipf_df["rank"][i], zipf_df["frequency"][i]),
                    xytext=(5, 5),
                    textcoords="offset points",
                )

            save_plot(f"zipf_detailed_{lang}", output_dir)
            print(f"Graphique créé pour {lang}")

        except Exception as e:
            print(f"Erreur lors de l'analyse de Zipf pour {lang}: {e}")
            continue

    return "Analyse approfondie de la loi de Zipf terminée!"


# ==============
# FONCTION MAIN
# ==============

def main():
    """Fonction principale pour l'analyse et la visualisation"""
    try:
        # Configuration initiale
        setup_matplotlib_config()

        output_dir = DEFAULT_PATHS["output_base"]
        validate_output_directory(output_dir)

        print("Chargement des articles...")
        articles_df = load_all_articles()

        print("\n1. Exploration des statistiques générales...")
        explore_corpus_stats(articles_df, output_dir=output_dir)

        print("\n2. Analyse des caractéristiques textuelles...")
        analyze_text_characteristics(articles_df, output_dir=output_dir)

        print("\n3. Analyse de la loi de Zipf...")
        analyze_zipf_law(articles_df, output_dir=output_dir)

        print("\n✅ Toutes les visualisations ont été générées avec succès!")
        print(f"📁 Résultats consultables dans le dossier '{output_dir}'")

    except Exception as e:
        print(f"❌ Erreur lors de l'analyse: {e}")
        raise


if __name__ == "__main__":
    main()
