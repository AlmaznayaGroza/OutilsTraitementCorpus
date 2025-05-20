import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def split_datasets(
        merged_dir='data/processed/merged', augmented_dir='data/processed/augmented',
        output_dir='data/final', train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
        ):
    """
    Divise le corpus final en ensembles d'entraînement, validation et test.
    
    Args:
        merged_dir: dossier contenant les fichiers originaux nettoyés
        augmented_dir: dossier contenant les fichiers augmentés
        output_dir: dossier où sauvegarder les ensembles
        train_ratio, val_ratio, test_ratio: proportions des ensembles
    """
    # Vérifier que les ratios somment à 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "Les ratios doivent sommer à 1"
    
    # Créer les sous-répertoires
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "validation"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)
    
    # Charger les articles originaux
    original_articles = []
    for file in os.listdir(merged_dir):
        if file.endswith('.csv'):
            file_path = os.path.join(merged_dir, file)
            df = pd.read_csv(file_path)
            original_articles.append(df)
    
    # Charger les articles augmentés
    augmented_articles = []
    for file in os.listdir(augmented_dir):
        if file.endswith('.csv'):
            file_path = os.path.join(augmented_dir, file)
            df = pd.read_csv(file_path)
            augmented_articles.append(df)
    
    # Fusionner tous les articles
    full_corpus = pd.concat(original_articles + augmented_articles, ignore_index=True)

    # Charger les articles originaux
    original_articles = []
    for file in os.listdir(merged_dir):
        if file.endswith('.csv'):
            file_path = os.path.join(merged_dir, file)
            df = pd.read_csv(file_path)
            
            # S'assurer que toutes les colonnes nécessaires existent
            # et sont dans le bon ordre pour les articles originaux
            if 'language' not in df.columns and 'page_id' in df.columns:
                # Renommer page_id en pageid pour cohérence si nécessaire
                if 'pageid' not in df.columns and 'page_id' in df.columns:
                    df = df.rename(columns={'page_id': 'pageid'})
            
            # Ajouter une colonne source si elle n'existe pas
            if 'source' not in df.columns:
                df['source'] = 'original'  # marquer comme original
                
            original_articles.append(df)

    # Charger les articles augmentés
    augmented_articles = []
    for file in os.listdir(augmented_dir):
        if file.endswith('.csv'):
            file_path = os.path.join(augmented_dir, file)
            df = pd.read_csv(file_path)
            
            # S'assurer que toutes les colonnes nécessaires existent
            if 'pageid' not in df.columns:
                df['pageid'] = None         # ajouter une colonne pageid vide
                
            augmented_articles.append(df)

    # Fusionner tous les articles
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
        if 'language' in full_corpus.columns:
            full_corpus['language'] = full_corpus['language'].replace('be-tarask', 'be')
                
        # Réorganiser les colonnes pour mettre language, title, text en premier
        essential_columns = ['language', 'title', 'text']
        other_columns = [ col for col in full_corpus.columns if col not in essential_columns ]
        full_corpus = full_corpus[essential_columns + other_columns]
    else:
        full_corpus = pd.DataFrame()
    
    # Diviser directement par langue pour obtenir une distribution équilibrée
    train_dfs = []
    val_dfs = []
    test_dfs = []
    
    for language in full_corpus['language'].unique():
        lang_df = full_corpus[full_corpus['language'] == language]
        
        # Pour chaque langue, stratifier par source si possible
        if 'source' in lang_df.columns and len(lang_df['source'].unique()) > 1:
            # Grouper par source pour assurer une distribution équilibrée
            for source in lang_df['source'].unique():
                source_df = lang_df[lang_df['source'] == source]
                
                # Si nous avons suffisamment d'exemples, diviser normalement
                if len(source_df) >= 10:  # Assez d'exemples pour une division significative
                    train_size = int(len(source_df) * train_ratio)
                    val_size = int(len(source_df) * val_ratio)
                    
                    # Mélanger les données
                    shuffled_df = source_df.sample(frac=1, random_state=42).reset_index(drop=True)
                    
                    # Diviser
                    train_part = shuffled_df.iloc[:train_size]
                    val_part = shuffled_df.iloc[train_size:train_size+val_size]
                    test_part = shuffled_df.iloc[train_size+val_size:]
                    
                    train_dfs.append(train_part)
                    val_dfs.append(val_part)
                    test_dfs.append(test_part)
                    
                elif len(source_df) >= 3:  # Assez pour au moins un exemple dans chaque ensemble
                    train_size = max(1, int(len(source_df) * train_ratio))
                    val_size = max(1, int(len(source_df) * val_ratio))
                    
                    # Mélanger les données
                    shuffled_df = source_df.sample(frac=1, random_state=42).reset_index(drop=True)
                    
                    # Diviser
                    train_part = shuffled_df.iloc[:train_size]
                    val_part = shuffled_df.iloc[train_size:train_size+val_size]
                    test_part = shuffled_df.iloc[train_size+val_size:]
                    
                    train_dfs.append(train_part)
                    if len(val_part) > 0:
                        val_dfs.append(val_part)
                    if len(test_part) > 0:
                        test_dfs.append(test_part)
                    
                else:  # Très peu d'exemples, allouer à l'entraînement principalement
                    if len(source_df) == 2:
                        train_dfs.append(source_df.iloc[:1])
                        val_dfs.append(source_df.iloc[1:])
                    else:  # Un seul exemple
                        train_dfs.append(source_df)
        else:
            # Si pas de colonne source ou une seule source, diviser directement la langue
            if len(lang_df) >= 10:
                train_size = int(len(lang_df) * train_ratio)
                val_size = int(len(lang_df) * val_ratio)
                
                # Mélanger les données
                shuffled_df = lang_df.sample(frac=1, random_state=42).reset_index(drop=True)
                
                # Diviser
                train_part = shuffled_df.iloc[:train_size]
                val_part = shuffled_df.iloc[train_size:train_size+val_size]
                test_part = shuffled_df.iloc[train_size+val_size:]
                
                train_dfs.append(train_part)
                val_dfs.append(val_part)
                test_dfs.append(test_part)
                
            elif len(lang_df) >= 3:
                train_size = max(1, int(len(lang_df) * train_ratio))
                val_size = max(1, int(len(lang_df) * val_ratio))
                
                # Mélanger les données
                shuffled_df = lang_df.sample(frac=1, random_state=42).reset_index(drop=True)
                
                # Diviser
                train_part = shuffled_df.iloc[:train_size]
                val_part = shuffled_df.iloc[train_size:train_size+val_size]
                test_part = shuffled_df.iloc[train_size+val_size:]
                
                train_dfs.append(train_part)
                if len(val_part) > 0:
                    val_dfs.append(val_part)
                if len(test_part) > 0:
                    test_dfs.append(test_part)
                
            else:  # Très peu d'exemples
                if len(lang_df) == 2:
                    train_dfs.append(lang_df.iloc[:1])
                    val_dfs.append(lang_df.iloc[1:])
                else:  # Un seul exemple
                    train_dfs.append(lang_df)
    
    # Combiner les ensembles
    train_df = pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame()
    val_df = pd.concat(val_dfs, ignore_index=True) if val_dfs else pd.DataFrame()
    test_df = pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame()
    
    # Sauvegarder les ensembles
    train_df.to_csv(os.path.join(output_dir, "train", "train_corpus.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "validation", "validation_corpus.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test", "test_corpus.csv"), index=False)
    
    # Statistiques de base sur la division du corpus
    print(f"\nDivision du corpus en ensembles:")
    print(f"Corpus complet: {len(full_corpus)} articles")
    print(f"Ensemble d'entraînement: {len(train_df)} articles ({len(train_df)/len(full_corpus)*100:.1f}%)")
    print(f"Ensemble de validation: {len(val_df)} articles ({len(val_df)/len(full_corpus)*100:.1f}%)")
    print(f"Ensemble de test: {len(test_df)} articles ({len(test_df)/len(full_corpus)*100:.1f}%)")
    
    # Statistiques par langue
    print("\nDistribution des langues dans chaque ensemble:")
    print("\nEntraînement:")
    print(train_df['language'].value_counts())
    print("\nValidation:")
    print(val_df['language'].value_counts() if len(val_df) > 0 else "Ensemble vide")
    print("\nTest:")
    print(test_df['language'].value_counts() if len(test_df) > 0 else "Ensemble vide")
    
    # Calculer les statistiques de longueur pour vérifier la représentativité
    if 'token_count' in train_df.columns and len(train_df) > 0:
        train_length_stats = train_df['token_count'].describe()
        
        print("\nStatistiques de longueur des textes (en tokens) par ensemble:")
        print(f"Entraînement: moyenne={train_length_stats['mean']:.1f}, médiane={train_length_stats['50%']:.1f}, min={train_length_stats['min']:.1f}, max={train_length_stats['max']:.1f}")
        
        if 'token_count' in val_df.columns and len(val_df) > 0:
            val_length_stats = val_df['token_count'].describe()
            print(f"Validation: moyenne={val_length_stats['mean']:.1f}, médiane={val_length_stats['50%']:.1f}, min={val_length_stats['min']:.1f}, max={val_length_stats['max']:.1f}")
        else:
            print("Validation: pas de statistiques disponibles")
        
        if 'token_count' in test_df.columns and len(test_df) > 0:
            test_length_stats = test_df['token_count'].describe()
            print(f"Test: moyenne={test_length_stats['mean']:.1f}, médiane={test_length_stats['50%']:.1f}, min={test_length_stats['min']:.1f}, max={test_length_stats['max']:.1f}")
        else:
            print("Test: pas de statistiques disponibles")
    
    # Vérifier la distribution des méthodes d'augmentation si applicable
    if 'source' in train_df.columns:
        print("\nDistribution des méthodes d'augmentation:")
        print("Entraînement:")
        print(train_df['source'].value_counts())
        
        print("\nValidation:")
        if 'source' in val_df.columns and len(val_df) > 0:
            print(val_df['source'].value_counts())
        else:
            print("Pas de données disponibles")
        
        print("\nTest:")
        if 'source' in test_df.columns and len(test_df) > 0:
            print(test_df['source'].value_counts())
        else:
            print("Pas de données disponibles")
    
    # Calculer l'entropie pour mesurer l'équilibre des distributions
    def calculate_entropy(df, col='language'):
        """
        Calcule l'entropie de la distribution des valeurs dans une colonne.
        Une entropie plus élevée indique une distribution plus équilibrée.
        """
        if len(df) == 0:
            return 0.0
        
        counts = df[col].value_counts()
        probabilities = counts / counts.sum()
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy
    
    # Calculer l'entropie de chaque ensemble
    train_entropy = calculate_entropy(train_df) if len(train_df) > 0 else 0.0
    val_entropy = calculate_entropy(val_df) if len(val_df) > 0 else 0.0
    test_entropy = calculate_entropy(test_df) if len(test_df) > 0 else 0.0
    
    print("\nEntropie de la distribution des langues (mesure d'équilibre):")
    print(f"Entraînement: {train_entropy:.4f} bits")
    print(f"Validation: {val_entropy:.4f} bits")
    print(f"Test: {test_entropy:.4f} bits")
    
    return train_df, val_df, test_df


# Si le script est exécuté directement
if __name__ == "__main__":
    # Définir les répertoires d'entrée et de sortie
    merged_dir='data/processed/merged'
    augmented_dir='data/processed/augmented'
    output_dir = "data/final"
    
    # Diviser le corpus
    train_df, val_df, test_df = split_datasets(merged_dir, augmented_dir, output_dir)