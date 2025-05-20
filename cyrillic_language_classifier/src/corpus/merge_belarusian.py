import pandas as pd
import os
import shutil


def merge_belarusian_variants(input_dir, output_dir):
    """
    Fusionne les variantes du bélarussien à partir des fichiers nettoyés.
    
    Args:
        input_dir: répertoire contenant les fichiers nettoyés par langue
        output_dir: répertoire où sauvegarder le corpus fusionné
    """
    # Créer le dossier de sortie si nécessaire
    os.makedirs(output_dir, exist_ok=True)
    
    # Charger les articles pour be et be-tarask
    be_path = os.path.join(input_dir, "be_cleaned_articles.csv")
    be_tarask_path = os.path.join(input_dir, "be-tarask_cleaned_articles.csv")
    
    be_df = pd.read_csv(be_path) if os.path.exists(be_path) else pd.DataFrame()
    be_tarask_df = pd.read_csv(be_tarask_path) if os.path.exists(be_tarask_path) else pd.DataFrame()
    
    # Statistiques avant fusion
    print("\nStatistiques avant fusion:")
    print(f"Articles en bélarussien standard (be): {len(be_df)}")
    print(f"Articles en bélarussien traditionnel (be-tarask): {len(be_tarask_df)}")
    
    # Fusionner les variantes
    if not be_tarask_df.empty:
        be_tarask_df['language'] = 'be'  # Remplacer be-tarask par be
        merged_be_df = pd.concat([be_df, be_tarask_df], ignore_index=True)
        print(f"\nFusion réalisée: {len(merged_be_df)} articles au total")
    else:
        merged_be_df = be_df
        print("\nAucun article en be-tarask trouvé, pas de fusion nécessaire")
    
    # Sauvegarder le fichier fusionné
    merged_output_path = os.path.join(output_dir, "be_merged_cleaned_articles.csv")
    merged_be_df.to_csv(merged_output_path, index=False)
    
    # Copier tous les autres fichiers (autres langues)
    for file in os.listdir(input_dir):
        if file.endswith('.csv') and file not in ['be_cleaned_articles.csv', 'be-tarask_cleaned_articles.csv']:
            src_path = os.path.join(input_dir, file)
            dst_path = os.path.join(output_dir, file)
            shutil.copy2(src_path, dst_path)
    
    return merged_be_df


if __name__ == "__main__":
    # Chemins des répertoires d'entrée et de sortie
    input_directory = "data/processed/cleaned"
    output_directory = "data/processed/merged"
    
    # Appel de la fonction
    print("Fusion des variantes du bélarussien...")
    result = merge_belarusian_variants(input_directory, output_directory)
    
    print("\nFusion terminée!")
    print(f"Les fichiers fusionnés ont été sauvegardés dans {output_directory}")