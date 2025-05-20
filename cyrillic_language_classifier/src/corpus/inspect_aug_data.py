import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob


# Créer les répertoires de sortie
os.makedirs('results/figures/augmentation', exist_ok=True)
os.makedirs('results/metrics/augmentation', exist_ok=True)

# Charger les données
augmented_df = pd.read_csv('data/processed/augmented/all_augmented_articles.csv')
original_files = glob.glob('data/processed/merged/*_articles.csv')
original_dfs = []


for file in original_files:
    df = pd.read_csv(file)
    df['source_corpus'] = 'original'
    original_dfs.append(df)

original_df = pd.concat(original_dfs, ignore_index=True)
augmented_df['source_corpus'] = 'augmented'
combined_df = pd.concat([original_df, augmented_df], ignore_index=True)


# Statistiques de base
augmented_stats = augmented_df.groupby(['language', 'source']).agg(
    article_count=('title', 'count'),
    avg_tokens=('token_count', 'mean'),
    min_tokens=('token_count', 'min'),
    max_tokens=('token_count', 'max')
).reset_index()

print("\n=== Statistiques par type d'augmentation et langue ===")
print(augmented_stats)


# Distribution des longueurs
plt.figure(figsize=(12, 8))
sns.histplot(data=combined_df, x='token_count', hue='source_corpus', 
             bins=30, kde=True, element='step')
plt.title('Distribution des longueurs d\'articles (tokens)')
plt.xlabel('Nombre de tokens')
plt.ylabel('Nombre d\'articles')
plt.grid(True, alpha=0.3)
plt.savefig('results/figures/augmentation/length_distribution_comparison.png', dpi=300)
plt.close()


# Comparer les longueurs moyennes par langue
lang_stats = combined_df.groupby(['language', 'source_corpus']).agg(
    avg_tokens=('token_count', 'mean')
).reset_index()

lang_pivot = lang_stats.pivot(index='language', columns='source_corpus', values='avg_tokens').reset_index()
lang_pivot['diff_pct'] = (lang_pivot['augmented'] - lang_pivot['original']) / lang_pivot['original'] * 100

print("\n=== Différence de longueur moyenne par langue ===")
print(lang_pivot.sort_values(by='diff_pct'))


# Visualiser pour les langues prioritaires
priority_langs = ['ab', 'kbd', 'koi', 'kv', 'mhr']
priority_df = combined_df[combined_df['language'].isin(priority_langs)]

plt.figure(figsize=(14, 8))
sns.boxplot(data=priority_df, x='language', y='token_count', hue='source_corpus')
plt.title('Comparaison des longueurs pour les langues prioritaires')
plt.xlabel('Langue')
plt.ylabel('Nombre de tokens')
plt.grid(True, alpha=0.3)
plt.legend(title='Source')
plt.tight_layout()
plt.savefig('results/figures/augmentation/priority_langs_comparison.png', dpi=300)
plt.close()


# Distribution des longueurs par méthode d'augmentation
plt.figure(figsize=(10, 6))
sns.boxplot(data=augmented_df, x='source', y='token_count')
plt.title('Longueur des textes par méthode d\'augmentation')
plt.xlabel('Méthode')
plt.ylabel('Nombre de tokens')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('results/figures/augmentation/length_by_method.png')
plt.close()


# Carte thermique simple des langues augmentées
pivot = pd.crosstab(
    index=augmented_df['language'],
    columns=augmented_df['source'],
    margins=False
)

plt.figure(figsize=(12, 10))
sns.heatmap(pivot, annot=True, cmap='YlGnBu', fmt='g')
plt.title('Nombre d\'articles par langue et méthode d\'augmentation')
plt.tight_layout()
plt.savefig('results/figures/augmentation/language_method_heatmap.png')
plt.close()


# Calculer l'entropie de la distribution des langues
def calculate_entropy(df, col='language'):
    # Calculer les probabilités de chaque langue
    counts = df[col].value_counts()
    probabilities = counts / counts.sum()
    
    # Calculer l'entropie
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

# Calculer pour les corpus original et augmenté
# Filtrer les langues mixtes pour une comparaison équitable
augmented_single_langs = augmented_df[~augmented_df['language'].str.contains('_mix')]
original_entropy = calculate_entropy(original_df)
augmented_entropy = calculate_entropy(augmented_single_langs)
combined_entropy = calculate_entropy(pd.concat([original_df, augmented_single_langs]))

print("\n=== Entropie de la distribution des langues ===")
print(f"Corpus original: {original_entropy:.4f} bits")
print(f"Corpus augmenté: {augmented_entropy:.4f} bits")
print(f"Corpus combiné: {combined_entropy:.4f} bits")

# Visualiser l'entropie
plt.figure(figsize=(8, 6))
entropies = [original_entropy, augmented_entropy, combined_entropy]
corpus_types = ['Original', 'Augmenté', 'Combiné']
plt.bar(corpus_types, entropies, color=['blue', 'green', 'orange'])
plt.title('Entropie de la distribution des langues par corpus')
plt.ylabel('Entropie (bits)')
plt.grid(axis='y', alpha=0.3)
plt.savefig('results/figures/augmentation/language_entropy.png')
plt.close()


# Afficher quelques exemples
for source_type in ['data_augmentation', 'cross_language_augmentation', 'data_perturbation']:
    samples = augmented_df[augmented_df['source'] == source_type].sample(min(3, len(augmented_df[augmented_df['source'] == source_type])))
    
    print(f"\n=== Exemples d'articles {source_type} ===")
    for i, (_, article) in enumerate(samples.iterrows(), 1):
        print(f"\nExemple {i}:")
        print(f"Titre: {article['title']}")
        print(f"Langue: {article['language']}")
        print(f"Longueur: {article['token_count']} tokens")
        print(f"Texte (100 premiers caractères): {article['text'][:100]}...")

print("\nAnalyse de qualité terminée. Consultez les visualisations dans results/figures/augmentation/")