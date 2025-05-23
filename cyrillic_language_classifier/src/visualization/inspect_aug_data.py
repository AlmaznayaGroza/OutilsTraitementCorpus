"""Script d'inspection et d'analyse qualitative des données augmentées

Ce script fournit des outils pour évaluer la qualité et la cohérence
des données générées par le processus d'augmentation de corpus multilingue.
Il implémente une série d'analyses comparatives entre les données originales
et augmentées pour valider l'efficacité des stratégies d'augmentation.

Analyses principales implémentées:
    * comparaison des distributions de longueur entre corpus original et augmenté
    * évaluation de l'équilibrage linguistique après augmentation
    * analyse de l'entropie pour mesurer la diversité des distributions
    * inspection des caractéristiques textuelles par méthode d'augmentation
    * validation de la cohérence des métadonnées entre les ensembles

Méthodologie d'évaluation:
    Le script suit une approche quantitative rigoureuse qui compare les
    propriétés statistiques des données avant et après augmentation.
    Cette validation est cruciale pour s'assurer que l'augmentation
    améliore effectivement l'équilibrage du corpus sans introduire de biais
    ou de distorsions significatives dans les distributions linguistiques.

Architecture d'analyse :
    Les analyses sont organisées selon une hiérarchie logique qui progresse
    des statistiques descriptives générales vers des analyses spécialisées
    par méthode d'augmentation. Cette approche systématique facilite
    l'identification rapide des succès et des limitations du processus
    d'augmentation.

Applications de validation:
    Adapté pour valider la qualité des corpus augmentés destinés
    à l'entraînement de modèles de traitement automatique des langues,
    où l'équilibrage et la representativité des données sont cruciaux pour
    la performance et l'équité des systèmes développés.
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from typing import Dict


# =============================================================================
# CONSTANTES DE CONFIGURATION POUR L'ANALYSE DES DONNÉES AUGMENTÉES
# =============================================================================

# Configuration des répertoires de données
DATA_PATHS = {
    'original_pattern': 'data/processed/merged/*_articles.csv',
    'augmented_file': 'data/processed/augmented/all_augmented_articles.csv',
    'output_base': 'results/figures/augmentation',
    'metrics_base': 'results/metrics/augmentation'
}

# Paramètres de visualisation
VISUALIZATION_CONFIG = {
    'figure_size': (12, 8),
    'priority_figure_size': (14, 8),
    'heatmap_figure_size': (12, 10),
    'dpi': 300,
    'style': 'seaborn-v0_8-whitegrid'
}

# Configuration des couleurs pour les comparaisons
COLOR_SCHEMES = {
    'comparison_palette': ['#1f77b4', '#ff7f0e'], # bleu/orange pour original/augmenté
    'methods_palette': 'Set2',                        # palette pour les méthodes d'augmentation
    'heatmap_colormap': 'YlGnBu'                      # pour les heatmaps
}

# Langues prioritaires pour l'analyse approfondie
PRIORITY_LANGUAGES = ['ab', 'kbd', 'koi', 'kv', 'mhr']

# Méthodes d'augmentation attendues
AUGMENTATION_METHODS = [
    'data_augmentation',
    'cross_language_augmentation', 
    'data_perturbation'
]

# Paramètres statistiques
STATISTICAL_CONFIG = {
    'histogram_bins': 30,
    'entropy_precision': 4,
    'percentage_precision': 1,
    'sample_size_examples': 3
}


# =================================================
# FONCTIONS UTILITAIRES POUR L'ANALYSE COMPARATIVE
# =================================================


def setup_analysis_environment() -> None:
    """Configure l'environnement d'analyse et les paramètres
    
    Cette fonction initialise Matplotlib et Weaborn avec des paramètres
    adaptés à l'analyse comparative de données, afin de garantir
    une cohérence visuelle dans toutes les visualisations générées.
    
    La configuration privilégie la lisibilité.
    """
    plt.style.use(VISUALIZATION_CONFIG['style'])
    plt.rcParams['figure.figsize'] = VISUALIZATION_CONFIG['figure_size']
    plt.rcParams['savefig.dpi'] = VISUALIZATION_CONFIG['dpi']
    plt.rcParams['font.size'] = 11
    
    # Créer les dossiers de sortie si nécessaire
    os.makedirs(DATA_PATHS['output_base'], exist_ok=True)
    os.makedirs(DATA_PATHS['metrics_base'], exist_ok=True)


def load_original_corpus() -> pd.DataFrame:
    """Charge et unifie tous les fichiers du corpus original
    
    Cette fonction rassemble tous les fichiers de données originales
    en un DataFrame unifié, permettant une comparaison cohérente avec
    les données augmentées. Elle applique une validation de base pour
    s'assurer de la cohérence des données chargées.
    
    Returns:
        pd.DataFrame: corpus original unifié avec marquage de source
        
    Raises:
        FileNotFoundError: si aucun fichier original n'est trouvé
        ValueError: si les données chargées sont incohérentes
        
    Note:
        La fonction ajoute automatiquement une colonne 'source_corpus'
        marquée comme 'original' pour faciliter les comparaisons ultérieures.
    """
    original_files = glob.glob(DATA_PATHS['original_pattern'])
    
    if not original_files:
        raise FileNotFoundError(
            f"Aucun fichier original trouvé: "
            f"{DATA_PATHS['original_pattern']}"
        )
    
    original_dfs = []
    for file in original_files:
        try:
            df = pd.read_csv(file)
            df['source_corpus'] = 'original'
            original_dfs.append(df)
        except Exception as e:
            print(f"Erreur lors du chargement de {file}: {e}")
            continue
    
    if not original_dfs:
        raise ValueError("Aucun fichier original valide n'a pu être chargé")
    
    combined_original = pd.concat(original_dfs, ignore_index=True)
    print(f"Corpus original chargé : {len(combined_original):,} articles")
    
    return combined_original


def load_augmented_corpus() -> pd.DataFrame:
    """Charge le corpus augmenté avec validation de cohérence
    
    Cette fonction charge le fichier de données augmentées et applique
    une validation pour s'assurer que les colonnes nécessaires sont
    présentes et que les données sont cohérentes pour l'analyse comparative.
    
    Returns:
        pd.DataFrame: corpus augmenté avec marquage de source
        
    Raises:
        FileNotFoundError: si le fichier augmenté n'existe pas
        ValueError: si le fichier augmenté est vide ou invalide
        
    Note:
        La fonction ajoute une colonne 'source_corpus' marquée comme 'augmented'
        et valide la présence des colonnes essentielles pour l'analyse.
    """
    augmented_path = DATA_PATHS['augmented_file']
    
    if not os.path.exists(augmented_path):
        raise FileNotFoundError(f"Fichier augmenté non trouvé: {augmented_path}")
    
    try:
        augmented_df = pd.read_csv(augmented_path)
        
        if augmented_df.empty:
            raise ValueError("Le fichier de données augmentées est vide")
        
        # Validation des colonnes essentielles
        required_columns = ['language', 'text', 'token_count', 'source']
        missing_columns = [
            col for col in required_columns
            if col not in augmented_df.columns
        ]
        
        if missing_columns:
            print(f"Colonnes manquantes dans les données augmentées: {missing_columns}")
        
        augmented_df['source_corpus'] = 'augmented'
        print(f"Corpus augmenté chargé : {len(augmented_df):,} articles")
        
        return augmented_df
        
    except Exception as e:
        raise ValueError(f"Erreur lors du chargement des données augmentées: {e}")


def calculate_distribution_entropy(
        df: pd.DataFrame,
        column: str = 'language'
) -> float:
    """Calcule l'entropie de Shannon pour une distribution de données

    Cette fonction mesure la diversité d'une distribution en calculant
    son entropie de Shannon. Une entropie plus élevée indique une
    distribution plus équilibrée, ce qui est généralement souhaitable
    pour les corpus d'entraînement de modèles de machine learning.
    Cette métrique est particulièrement utile pour évaluer
    l'efficacité des stratégies d'augmentation de données.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données à analyser
        column (str): nom de la colonne pour calculer l'entropie
        
    Returns:
        float: entropie de Shannon en bits (0.0 si DataFrame vide)
    """
    if len(df) == 0:
        return 0.0
    
    # Calculer la distribution des valeurs
    value_counts = df[column].value_counts()
    probabilities = value_counts / value_counts.sum()
    
    # Calculer l'entropie de Shannon
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    return entropy


def generate_comparison_statistics(
        original_df: pd.DataFrame, 
        augmented_df: pd.DataFrame
) -> Dict:
    """Génère des statistiques comparatives complètes entre les corpus
    
    Cette fonction calcule un ensemble complet de métriques comparatives
    qui permettent d'évaluer quantitativement l'impact du processus
    d'augmentation sur les caractéristiques du corpus.
    
    Args:
        original_df (pd.DataFrame): corpus original
        augmented_df (pd.DataFrame): corpus augmenté
        
    Returns:
        Dict: dictionnaire contenant toutes les statistiques comparatives
        
    Métriques calculées:
        - tailles des corpus et gains relatifs
        - entropies des distributions linguistiques
        - statistiques de longueur (moyenne, médiane, écart-type)
        - nombres de langues uniques et distributions
        - métriques de diversité et d'équilibrage
    """
    stats = {
        # Stats de base
        'original_size': len(original_df),
        'augmented_size': len(augmented_df),
        'total_size': len(original_df) + len(augmented_df),
        
        # Calcul des gains
        'augmentation_ratio': (
            len(augmented_df) / len(original_df)
            if len(original_df) > 0 else 0
        ),
        'augmentation_percentage': (
            (len(augmented_df) / len(original_df)) * 100
            if len(original_df) > 0 else 0
        ),
        
        # Entropies des distributions linguistiques
        'original_entropy': calculate_distribution_entropy(original_df),
        'augmented_entropy': calculate_distribution_entropy(augmented_df),
        
        # Stats de longueur
        'original_avg_length': (
            original_df['token_count'].mean()
            if 'token_count' in original_df.columns else 0
        ),
        'augmented_avg_length': (
            augmented_df['token_count'].mean()
            if 'token_count' in augmented_df.columns else 0
        ),
        
        # Diversité linguistique
        'original_languages': (
            original_df['language'].nunique()
            if 'language' in original_df.columns else 0
        ),
        'augmented_languages': (
            augmented_df['language'].nunique()
            if 'language' in augmented_df.columns else 0
        ),
    }
    
    # Calcul de l'entropie combinée pour mesurer l'effet global
    if not original_df.empty and not augmented_df.empty:
        combined_df = pd.concat([original_df, augmented_df], ignore_index=True)
        stats['combined_entropy'] = calculate_distribution_entropy(combined_df)
    else:
        stats['combined_entropy'] = 0.0
    
    return stats


# ====================================================
# FONCTIONS D'ANALYSE ET DE VISUALISATION PRINCIPALES
# ====================================================

def analyze_length_distributions(combined_df: pd.DataFrame) -> None:
    """Analyse comparative des distributions de longueur entre corpus
    
    Cette fonction génère des visualisations comparatives qui permettent
    d'évaluer si l'augmentation a préservé les caractéristiques de longueur
    du corpus original ou si elle a introduit des biais de longueur.
    
    Args:
        combined_df (pd.DataFrame): corpus combiné (original + augmenté)
        
    L'analyse produit des histogrammes comparatifs et des statistiques
    descriptives qui révèlent l'impact de l'augmentation sur la distribution
    des longueurs de texte, un facteur crucial pour la qualité du corpus.
    """
    plt.figure(figsize=VISUALIZATION_CONFIG['figure_size'])
    
    # Histogrammes comparatifs des longueurs
    sns.histplot(
        data=combined_df, 
        x='token_count', 
        hue='source_corpus',
        bins=STATISTICAL_CONFIG['histogram_bins'], 
        kde=True, 
        element='step',
        palette=COLOR_SCHEMES['comparison_palette']
    )
    
    plt.title('Comparaison des distributions de longueur d\'articles')
    plt.xlabel('Nombre de tokens')
    plt.ylabel('Nombre d\'articles')
    plt.legend(title='Source du corpus', labels=['Original', 'Augmenté'])
    plt.grid(True, alpha=0.3)
    
    # Sauvegarder la visualisation
    output_path = os.path.join(
        DATA_PATHS['output_base'],
        'length_distribution_comparison.png'
    )
    plt.savefig(
        output_path,
        dpi=VISUALIZATION_CONFIG['dpi'],
        bbox_inches='tight'
    )
    plt.close()
    
    print(f"✅ Analyse des distributions de longueur sauvegardée: {output_path}")


def analyze_language_balance(combined_df: pd.DataFrame) -> None:
    """Analyse de l'équilibrage linguistique après augmentation
    
    Cette fonction évalue l'efficacité de l'augmentation pour améliorer
    l'équilibrage entre les langues, particulièrement importante pour
    les langues sous-représentées dans le corpus original.
    
    Args:
        combined_df (pd.DataFrame): corpus combiné avec marquage de source
        
    L'analyse génère des visualisations qui montrent la répartition
    linguistique avant et après augmentation, permettant d'identifier
    les langues qui ont le plus bénéficié du processus d'augmentation.
    """
    # Statistiques par langue et par source
    lang_stats = combined_df.groupby(['language', 'source_corpus']).agg(
        article_count=('title', 'count'),
        avg_tokens=('token_count', 'mean')
    ).reset_index()
    
    # Tableau pivot pour la visualisation
    lang_pivot = lang_stats.pivot(
        index='language',
        columns='source_corpus',
        values='article_count'
    ).fillna(0)
    
    # Calculer les pourcentages d'augmentation par langue
    if 'original' in lang_pivot.columns and 'augmented' in lang_pivot.columns:
        lang_pivot['augmentation_ratio'] = (
            lang_pivot['augmented'] / lang_pivot['original'].replace(0, 1)
        )
   
    # Visualisation comparative pour les langues prioritaires
    priority_df = combined_df[combined_df['language'].isin(PRIORITY_LANGUAGES)]
    
    if not priority_df.empty:
        plt.figure(figsize=VISUALIZATION_CONFIG['priority_figure_size'])
        sns.boxplot(
            data=priority_df, 
            x='language', 
            y='token_count', 
            hue='source_corpus',
            palette=COLOR_SCHEMES['comparison_palette']
        )
        plt.title('Comparaison des longueurs pour les langues prioritaires')
        plt.xlabel('Langue')
        plt.ylabel('Nombre de tokens')
        plt.legend(title='Source', labels=['Original', 'Augmenté'])
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        output_path = os.path.join(
            DATA_PATHS['output_base'],
            'priority_langs_comparison.png'
        )
        plt.savefig(
            output_path,
            dpi=VISUALIZATION_CONFIG['dpi'],
            bbox_inches='tight'
        )
        plt.close()
        
        print(f"✅ Analyse des langues prioritaires sauvegardée: {output_path}")


def analyze_augmentation_methods(augmented_df: pd.DataFrame) -> None:
    """Analyse détaillée des méthodes d'augmentation utilisées
    
    Cette fonction examine les caractéristiques spécifiques de chaque
    méthode d'augmentation pour évaluer leur contribution respective
    à la diversité et à la qualité du corpus final.
    
    Args:
        augmented_df (pd.DataFrame): corpus augmenté avec métadonnées de méthodes
        
    L'analyse produit des visualisations qui permettent de comparer
    l'efficacité des différentes stratégies d'augmentation et d'identifier
    les méthodes les plus performantes pour chaque type de langue.
    """
    if 'source' not in augmented_df.columns:
        print("⚠️ Colonne 'source' manquante, analyse des méthodes ignorée")
        return
    
    # Distribution des longueurs par méthode d'augmentation
    plt.figure(figsize=VISUALIZATION_CONFIG['figure_size'])
    sns.boxplot(
        data=augmented_df,
        x='source',
        y='token_count',
        palette=COLOR_SCHEMES['methods_palette']
    )
    plt.title('Distribution des longueurs par méthode d\'augmentation')
    plt.xlabel('Méthode d\'augmentation')
    plt.ylabel('Nombre de tokens')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(
        DATA_PATHS['output_base'],
        'length_by_method.png'
    )
    plt.savefig(
        output_path,
        dpi=VISUALIZATION_CONFIG['dpi'],
        bbox_inches='tight'
    )
    plt.close()
    
    # Heatmap des méthodes par langue
    method_lang_crosstab = pd.crosstab(
        index=augmented_df['language'],
        columns=augmented_df['source'],
        margins=False
    )
    
    plt.figure(figsize=VISUALIZATION_CONFIG['heatmap_figure_size'])
    sns.heatmap(
        method_lang_crosstab,
        annot=True,
        cmap=COLOR_SCHEMES['heatmap_colormap'],
        fmt='g',
        cbar_kws={'label': 'Nombre d\'articles'}
    )
    plt.title('Répartition des articles par langue et méthode d\'augmentation')
    plt.xlabel('Méthode d\'augmentation')
    plt.ylabel('Langue')
    plt.tight_layout()
    
    output_path = os.path.join(
        DATA_PATHS['output_base'],
        'language_method_heatmap.png'
    )
    plt.savefig(
        output_path,
        dpi=VISUALIZATION_CONFIG['dpi'],
        bbox_inches='tight'
    )
    plt.close()
    
    print(f"✅ Analyse des méthodes d'augmentation sauvegardée: {output_path}")


def generate_entropy_analysis(
        original_df: pd.DataFrame, 
        augmented_df: pd.DataFrame, 
        combined_df: pd.DataFrame
) -> None:
    """Analyse de l'entropie pour mesurer l'amélioration de la diversité
    
    Cette fonction calcule et visualise les métriques d'entropie qui
    quantifient objectivement l'amélioration de l'équilibrage linguistique
    apportée par le processus d'augmentation.
    
    Args:
        original_df (pd.DataFrame): corpus original
        augmented_df (pd.DataFrame): corpus augmenté
        combined_df (pd.DataFrame): corpus combiné
        
    L'analyse d'entropie fournit une mesure quantitative de la diversité
    qui permet d'évaluer objectivement l'efficacité des stratégies
    d'augmentation pour créer des corpus plus équilibrés.
    """
    # Calculer les entropies
    original_entropy = calculate_distribution_entropy(original_df)
    augmented_entropy = calculate_distribution_entropy(augmented_df)
    combined_entropy = calculate_distribution_entropy(combined_df)
    
    # Visualisation comparative des entropies
    entropies = [original_entropy, augmented_entropy, combined_entropy]
    corpus_types = ['Original', 'Augmenté', 'Combiné']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    plt.figure(figsize=VISUALIZATION_CONFIG['figure_size'])
    bars = plt.bar(corpus_types, entropies, color=colors, alpha=0.7)
    plt.title('Entropie de la distribution des langues par corpus')
    plt.ylabel('Entropie (bits)')
    plt.grid(axis='y', alpha=0.3)
    
    # Ajouter les valeurs sur les barres
    for bar, entropy in zip(bars, entropies):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{entropy:.{STATISTICAL_CONFIG["entropy_precision"]}f}',
            ha='center', va='bottom', fontweight='bold'
        )
    
    plt.tight_layout()
    
    output_path = os.path.join(
        DATA_PATHS['output_base'],
        'language_entropy.png'
    )
    plt.savefig(
        output_path,
        dpi=VISUALIZATION_CONFIG['dpi'],
        bbox_inches='tight'
    )
    plt.close()
    
    print(f"✅ Analyse d'entropie sauvegardée: {output_path}")
    
    # Afficher les résultats numériques
    print(f"\n📊 Résultats de l'analyse d'entropie:")
    print(
        f"   • Corpus original     : {
        original_entropy:.{STATISTICAL_CONFIG['entropy_precision']}f
        } bits"
    )
    print(
        f"   • Corpus augmenté     : {
        augmented_entropy:.{STATISTICAL_CONFIG['entropy_precision']}f
        } bits"
    )
    print(
        f"   • Corpus combiné      : {
        combined_entropy:.{STATISTICAL_CONFIG['entropy_precision']}f
        } bits"
    )


# ==================================
# FONCTION PRINCIPALE D'INSPECTION
# ==================================

def inspect_augmentation_quality() -> Dict:
    """
    Exécute l'inspection complète de la qualité des données augmentées
    
    Cette fonction orchestre l'ensemble du processus d'analyse comparative
    entre les données originales et augmentées, et produit un rapport
    complet sur l'efficacité et la qualité du processus d'augmentation.
    
    Returns:
        Dict: dictionnaire contenant toutes les statistiques d'analyse
        
    Raises:
        FileNotFoundError: si les fichiers de données requis sont manquants
        ValueError: si les données chargées sont incohérentes ou invalides
        
    Le processus d'inspection comprend:
        1. configuration de l'environnement d'analyse
        2. chargement et validation des corpus original et augmenté
        3. génération des statistiques comparatives
        4. analyses visuelles des distributions et caractéristiques
        5. évaluation quantitative de l'amélioration de l'équilibrage
        6. génération d'un rapport de synthèse complet
        
    Cette fonction constitue le point d'entrée principal pour l'évaluation
    de la qualité des données augmentées et peut être utilisée comme
    validation automatisée dans un pipeline de traitement de données.
    """
    
    print("🔍 Démarrage de l'inspection de la qualité des données augmentées")
    
    try:
        # 1. Configuration de l'environnement
        setup_analysis_environment()
        
        # 2. Chargement des données
        print("\n📂 Chargement des corpus...")
        original_df = load_original_corpus()
        augmented_df = load_augmented_corpus()
        
        # 3. Combinaison des corpus pour l'analyse comparative
        combined_df = pd.concat([original_df, augmented_df], ignore_index=True)
        
        # 4. Génération des statistiques comparatives
        print("\n📊 Calcul des statistiques comparatives...")
        stats = generate_comparison_statistics(original_df, augmented_df)
        
        # 5. Analyses visuelles et rapports
        print("\n📈 Génération des analyses visuelles...")
        analyze_length_distributions(combined_df)
        analyze_language_balance(combined_df)
        analyze_augmentation_methods(augmented_df)
        generate_entropy_analysis(original_df, augmented_df, combined_df)
        
        # 6. Affichage du résumé des résultats
        print("\n" + "="*60)
        print("RÉSUMÉ DE L'INSPECTION DES DONNÉES AUGMENTÉES")
        print("="*60)
        
        print(f"📊 Tailles des corpus:")
        print(f"   • Articles originaux      : {stats['original_size']:,}")
        print(f"   • Articles augmentés      : {stats['augmented_size']:,}")
        print(f"   • Total combiné           : {stats['total_size']:,}")
        print(f"   • Ratio d'augmentation    : {stats['augmentation_ratio']:.2f}x")
        print(
            f"   • Pourcentage d'augmentation : +{
            stats['augmentation_percentage']:.{STATISTICAL_CONFIG['percentage_precision']}f
            }%"
        )
        
        print(f"\n🌍 Diversité linguistique:")
        print(f"   • Langues originales      : {stats['original_languages']}")
        print(f"   • Langues augmentées      : {stats['augmented_languages']}")
        
        print(f"\n📏 Longueurs moyennes:")
        print(f"   • Corpus original         : {stats['original_avg_length']:.1f} tokens")
        print(f"   • Corpus augmenté         : {stats['augmented_avg_length']:.1f} tokens")
        
        print(f"\n🎲 Entropie des distributions:")
        print(
            f"   • Corpus original         : {
            stats['original_entropy']:.{STATISTICAL_CONFIG['entropy_precision']}f
            } bits"
        )
        print(
            f"   • Corpus augmenté         : {
            stats['augmented_entropy']:.{STATISTICAL_CONFIG['entropy_precision']}f
            } bits"
        )
        print(
            f"   • Corpus combiné          : {
            stats['combined_entropy']:.{STATISTICAL_CONFIG['entropy_precision']}f
            } bits"
        )
        
        # 7. Exemples d'articles augmentés par méthode
        print(f"\n📝 Exemples d'articles augmentés:")
        if 'source' in augmented_df.columns:
            for method in AUGMENTATION_METHODS:
                method_articles = augmented_df[augmented_df['source'] == method]
                if not method_articles.empty:
                    sample_size = min(
                        STATISTICAL_CONFIG['sample_size_examples'],
                        len(method_articles)
                    )
                    samples = method_articles.sample(sample_size)
                    
                    print(f"\n   {method.replace('_', ' ').title()} :")
                    for i, (_, article) in enumerate(samples.iterrows(), 1):
                        title = article['title'] if 'title' in article else f"Article_{i}"
                        language = article['language'] if 'language' in article else 'Unknown'
                        token_count = article['token_count'] if 'token_count' in article else 0
                        text_preview = (
                            str(article['text'])[:100] + "..."
                            if 'text' in article else "Pas de texte"
                        )
                        
                        print(f"     • {title} ({language}, {token_count} tokens)")
                        print(f"       {text_preview}")
        
        print(f"\n📁 Visualisations sauvegardées dans: {DATA_PATHS['output_base']}")
        print("="*60)
        
        # 8. Sauvegarde des statistiques
        stats_output = os.path.join(DATA_PATHS['metrics_base'], 'augmentation_stats.txt')
        with open(stats_output, 'w', encoding='utf-8') as f:
            f.write("=== RAPPORT D'INSPECTION DES DONNÉES AUGMENTÉES ===\n\n")
            f.write(f"Corpus original : {stats['original_size']:,} articles\n")
            f.write(f"Corpus augmenté : {stats['augmented_size']:,} articles\n")
            f.write(f"Ratio d'augmentation : {stats['augmentation_ratio']:.2f}x\n")
            f.write(f"Entropie originale : {stats['original_entropy']:.4f} bits\n")
            f.write(f"Entropie augmentée : {stats['augmented_entropy']:.4f} bits\n")
            f.write(f"Entropie combinée : {stats['combined_entropy']:.4f} bits\n")
        
        print(f"📄 Rapport statistique sauvegardé: {stats_output}")
        
        return stats
        
    except Exception as e:
        print(f"\n❌ Erreur lors de l'inspection des données augmentées: {e}")
        raise


# =========================
# POINT D'ENTRÉE PRINCIPAL
# =========================

if __name__ == "__main__":
    """Point d'entrée principal avec gestion d'erreurs complète
    
    Exécute l'inspection complète de la qualité des données augmentées
    avec gestion robuste des erreurs et affichage des résultats.
    
    Usage:
        python inspect_aug_data.py
    """
    try:
        print("🚀 Lancement de l'inspection des données augmentées...")
        
        # Exécution de l'inspection complète
        inspection_results = inspect_augmentation_quality()
        
        print(f"\n✅ Inspection terminée avec succès !")
        print(
            f"📊 {inspection_results['augmented_size']:,} "
            f"articles augmentés analysés"
        )
        print(
            f"🎯 Amélioration de l'équilibrage: "
            f"{inspection_results['augmentation_ratio']:.1f}x plus de données"
        )
        
        # Évaluation qualitative basée sur les métriques
        if (
            inspection_results['combined_entropy'] > inspection_results['original_entropy']
        ):
            print("✨ L'augmentation a amélioré la diversité du corpus")
        else:
            print("⚠️ L'augmentation n'a pas significativement amélioré la diversité")
        
    except KeyboardInterrupt:
        print("\n❌ Processus interrompu par l'utilisateur")
        exit(1)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n❌ Erreur de données: {e}")
        print("💡 Vérifiez la présence et le format des fichiers de corpus")
        exit(1)
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        print("💡 Consultez les logs pour plus de détails")
        exit(1)
