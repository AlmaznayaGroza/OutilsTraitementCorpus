import pandas as pd
import os
import glob

# Liste de tous les fichiers de stats
stats_files = [
    "results/metrics/collection/global/global_stats.csv",
    "results/metrics/collection/global/temp/global_stats_missing.csv",
    "results/metrics/collection/global/temp_final/global_stats_final.csv"
]

# Créer un dictionnaire pour stocker les "meilleures" stats
# ie le plus grand nombre de tokens pour chaque langue
best_stats = {}

# Parcourir tous les fichiers de stats
for stats_file in stats_files:
    if os.path.exists(stats_file):
        print(f"Traitement de {stats_file}...")
        stats = pd.read_csv(stats_file)
        
        for _, row in stats.iterrows():
            lang = row["Language"]
            tokens = row["Total tokens"]
            
            # Si cette langue n'existe pas encore ou si cette collecte a plus de tokens
            if lang not in best_stats or tokens > best_stats[lang]["Total tokens"]:
                best_stats[lang] = row.to_dict()

# Convertir en DataFrame et trier
final_stats = pd.DataFrame.from_dict(best_stats.values())
final_stats = final_stats.sort_values("Total tokens", ascending=False)

# Sauvegarder le résultat
final_stats.to_csv("results/metrics/collection/global/final_consolidated_stats.csv", index=False)

print(f"Statistiques consolidées pour {len(final_stats)} langues sauvegardées.")

# Fusionner également tous les fichiers d'articles
all_article_dirs = [
    "data/raw/intermediate_articles",
    "data/raw/temp_collection/intermediate_articles",
    "data/raw/temp_collection_final/intermediate_articles"
]

# Créer un répertoire pour les articles finaux
final_dir = "data/raw/final_corpus"
os.makedirs(final_dir, exist_ok=True)

# Dictionnaire pour stocker tous les articles par langue
all_articles = {}

# Parcourir tous les répertoires
for article_dir in all_article_dirs:
    if os.path.exists(article_dir):
        for article_file in glob.glob(f"{article_dir}/*.csv"):
            language = os.path.basename(article_file).split("_")[0]
            
            print(f"Traitement des articles pour {language} depuis {article_file}...")
            df = pd.read_csv(article_file)
            
            if language not in all_articles:
                all_articles[language] = df
            else:
                # Concaténer et éliminer les doublons
                combined = pd.concat([all_articles[language], df])
                # Éliminer les doublons basés sur page_id ou url
                if "page_id" in combined.columns:
                    combined = combined.drop_duplicates(subset=["page_id"])
                elif "url" in combined.columns:
                    combined = combined.drop_duplicates(subset=["url"])
                else:
                    combined = combined.drop_duplicates(subset=["text"])
                
                all_articles[language] = combined

# Sauvegarder les articles consolidés
for language, df in all_articles.items():
    output_file = os.path.join(final_dir, f"{language}_articles.csv")
    df.to_csv(output_file, index=False)
    print(f"Sauvegarde de {len(df)} articles pour {language}")

print("Consolidation complète!")