import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import re
import numpy as np
import plotly.express as px
import plotly.io as pio
import matplotlib.font_manager as fm
from collections import Counter
from wordcloud import WordCloud
from matplotlib.ticker import FuncFormatter


# Configuration pour l'affichage des caractères cyrilliques
plt.rcParams['font.family'] = 'DejaVu Sans'

# Configuration générale des graphiques
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


def load_all_articles():
    """Charge tous les articles depuis les fichiers CSV intermédiaires."""
    all_files = glob.glob('data/processed/merged/*_articles.csv')
    
    dataframes = []
    for file in all_files:
        lang_code = os.path.basename(file).split('_')[0]
        try:
            df = pd.read_csv(file)
            
            # Ajouter le code de langue si manquant
            if 'language' not in df.columns:
                df['language'] = lang_code
                
            dataframes.append(df)
        except Exception as e:
            print(f"Erreur lors du chargement de {file}: {e}")
    
    return pd.concat(dataframes, ignore_index=True)


def explore_corpus_stats(articles_df, output_dir='results/figures/distribution'):
    """Analyse statistique générale du corpus"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Statistiques globales
    print("=== Statistiques globales du corpus ===")
    print(f"Nombre total d'articles: {len(articles_df)}")
    print(f"Nombre total de tokens: {articles_df['token_count'].sum():,}")
    print(f"Nombre de langues: {articles_df['language'].nunique()}")
    
    # Distribution des tokens par article (histogramme)
    plt.figure(figsize=(12, 8))
    sns.histplot(articles_df['token_count'], bins=50, kde=True)
    plt.title('Distribution des longueurs d\'articles (tokens)')
    plt.xlabel('Nombre de tokens')
    plt.ylabel('Nombre d\'articles')
    # Formatter pour afficher les nombres grands avec des virgules
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'))
    plt.savefig(f"{output_dir}/token_distribution.png", dpi=300)
    plt.close()
    
    # Top 10 des langues par nombre d'articles
    plt.figure(figsize=(14, 8))
    top_langs = articles_df['language'].value_counts().head(10)
    sns.barplot(x=top_langs.index, y=top_langs.values, hue=top_langs.index, palette='viridis', legend=False)
    plt.title('Top 10 des langues par nombre d\'articles')
    plt.xlabel('Langue')
    plt.ylabel('Nombre d\'articles')
    plt.savefig(f"{output_dir}/top10_languages_by_articles.png", dpi=300)
    plt.close()
    
    # Distribution des catégories (toutes langues confondues)
    # Extraire la catégorie principale (sans sous-catégorie)
    articles_df['main_category'] = articles_df['category'].apply(
        lambda x: str(x).split(' (')[0] if isinstance(x, str) and ' (' in x else x
    )
    top_categories = articles_df['main_category'].value_counts().head(10)
    
    plt.figure(figsize=(14, 8))
    sns.barplot(x=top_categories.index, y=top_categories.values, hue=top_categories.index, palette='Set2', legend=False)
    plt.title('Top 10 des catégories thématiques')
    plt.xlabel('Catégorie')
    plt.ylabel('Nombre d\'articles')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/top10_categories.png", dpi=300)
    plt.close()
    
    # Distribution des tokens par langue (boxplot)
    plt.figure(figsize=(16, 10))
    sns.boxplot(x='language', y='token_count', data=articles_df.sort_values(by='language'))
    plt.title('Distribution des tokens par article pour chaque langue')
    plt.xlabel('Langue')
    plt.ylabel('Nombre de tokens')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/tokens_per_language_boxplot.png", dpi=300)
    plt.close()
    
    return "Exploration des statistiques du corpus terminée!"


def analyze_text_characteristics(articles_df, output_dir='results/figures/distribution'):
    """Analyse des caractéristiques textuelles du corpus"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Distribution des longueurs de texte par groupe de langue
    # Ajouter une colonne pour le groupe de langue
    language_groups = {
        'A': ['ba', 'be', 'be-tarask', 'bg', 'cv', 'kk', 'ky', 'mk', 'mn', 'ru', 'rue', 'sah', 'sr', 'tg', 'tt', 'tyv', 'uk'],
        'B': ['bxr', 'ce', 'myv', 'os', 'udm'],
        'C': ['ab', 'kbd', 'koi', 'kv'],
        'D': ['mhr']
    }
    
    # Fonction pour attribuer un groupe à chaque langue
    def get_group(lang):
        for group, langs in language_groups.items():
            if lang in langs:
                return group
        return 'Other'
    
    articles_df['language_group'] = articles_df['language'].apply(get_group)
    
    # Créer un boxplot par groupe
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='language_group', y='token_count', data=articles_df, order=['A', 'B', 'C', 'D'])
    plt.title('Distribution des longueurs d\'articles par groupe de langue')
    plt.xlabel('Groupe de langue')
    plt.ylabel('Nombre de tokens')
    plt.savefig(f"{output_dir}/token_distribution_by_group.png", dpi=300)
    plt.close()
    
    # 2. Loi de Zipf - pour quelques langues représentatives
    # Sélectionner quelques langues représentatives
    sample_languages = ['ru', 'uk', 'be', 'bg', 'mk', 'rue', 'sr', 'kk', 'mn', 'tt', 'tg',
                        'bxr', 'ce', 'koi', 'mhr']
    
    for lang in sample_languages:
        # Filtrer les articles pour cette langue
        lang_df = articles_df[articles_df['language'] == lang]
        
        if len(lang_df) < 5:  # ignorer les langues avec trop peu d'articles
            continue
            
        # Concaténer tous les textes
        all_text = ' '.join(lang_df['text'].fillna(''))
        
        # Tokeniser simplement par espaces (approx.)
        words = re.findall(r'\b\w+\b', all_text.lower())
        
        # Compter les fréquences
        word_counts = Counter(words)
        most_common = word_counts.most_common(50)
        
        # Créer un DataFrame pour le graphique
        zipf_df = pd.DataFrame(most_common, columns=['word', 'frequency'])
        zipf_df['rank'] = range(1, len(zipf_df) + 1)
        
        # Loi de Zipf: log-log plot
        plt.figure(figsize=(14, 8))
        plt.loglog(zipf_df['rank'], zipf_df['frequency'], marker='o')
        plt.title(f'Loi de Zipf pour la langue {lang}')
        plt.xlabel('Rang (log)')
        plt.ylabel('Fréquence (log)')
        plt.grid(True, which="both", ls="-")
        plt.savefig(f"{output_dir}/zipf_law_{lang}.png", dpi=300)
        plt.close()
        
        # Nuage de mots
        try:
            wordcloud = WordCloud(width=800, height=400, 
                                background_color='white',
                                max_words=100).generate(' '.join(word for word, _ in most_common))
            
            plt.figure(figsize=(16, 8))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Nuage des mots les plus fréquents en {lang}')
            plt.savefig(f"{output_dir}/wordcloud_{lang}.png", dpi=300)
            plt.close()
        except Exception as e:
            print(f"Erreur lors de la création du nuage de mots pour {lang}: {e}")
    
    # 3. Analyse des caractères spécifiques à chaque langue
    # Identifier les caractères cyrilliques spécifiques à chaque langue
    all_texts_by_lang = {}
    
    for lang in articles_df['language'].unique():
        lang_df = articles_df[articles_df['language'] == lang]
        all_texts_by_lang[lang] = ' '.join(lang_df['text'].fillna(''))
    
    # Identifier les caractères cyrilliques spécifiques à chaque langue
    lang_chars = {}
    for lang, text in all_texts_by_lang.items():
        # Compter les caractères cyrilliques
        chars = Counter(c for c in text if '\u0400' <= c <= '\u04FF' or '\u0500' <= c <= '\u052F')
        lang_chars[lang] = chars
    
    # Identifier les caractères distinctifs pour chaque langue
    distinctive_chars = {}
    
    for lang1, chars1 in lang_chars.items():
        # Normaliser par la fréquence totale
        total1 = sum(chars1.values()) or 1
        norm_chars1 = {c: cnt/total1 for c, cnt in chars1.items()}
        
        # Calculer une mesure de distinctivité
        distinctiveness = {}
        
        for char, freq in norm_chars1.items():
            # Calculer la ratio entre cette langue et les autres
            other_langs_avg = 0
            count = 0
            for lang2, chars2 in lang_chars.items():
                if lang1 != lang2:
                    total2 = sum(chars2.values()) or 1
                    other_langs_avg += chars2.get(char, 0) / total2
                    count += 1
            
            # Éviter la division par zéro
            if count > 0:
                other_langs_avg = other_langs_avg / count
            
            # Calculer la distinctivité (ratio entre fréquence dans cette langue vs autres)
            if other_langs_avg > 0:
                distinctiveness[char] = freq / other_langs_avg
            else:
                distinctiveness[char] = freq * 100  # Valeur arbitraire grande si exclusif à cette langue
        
        # Garder les 5 caractères les plus distinctifs (réduire de 10 à 5 pour avoir une visualisation plus gérable)
        distinctive_chars[lang1] = sorted(distinctiveness.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Créer un tableau de données (rows) à partir des caractères distinctifs
    rows = []
    for lang, chars in distinctive_chars.items():
        for char, score in chars:
            rows.append({
                'language': lang,
                'character': char,
                'distinctiveness': score
            })
    
    # Créer un DataFrame à partir des lignes
    char_df = pd.DataFrame(rows)
    
    # Créer une heatmap pour visualiser les caractères distinctifs
    # S'ssurer que toutes les langues sont représentées
    # Étape 1: Obtenir les 5 caractères les plus distinctifs globalement
    all_chars = char_df.groupby('character')['distinctiveness'].sum().nlargest(26).index.tolist()
    
    # Étape 2: Créer un pivot pour toutes les combinaisons langue-caractère
    pivot = pd.pivot_table(
        char_df,
        values='distinctiveness', 
        index='language',
        columns='character',
        fill_value=0
    )
    
    # Étape 3: Ne garder que les caractères sélectionnés
    if len(all_chars) > 0:
        available_chars = [c for c in all_chars if c in pivot.columns]
        pivot_filtered = pivot[available_chars]
    else:
        pivot_filtered = pivot
    
    # Étape 4: Normaliser pour une meilleure visualisation
    for idx in pivot_filtered.index:
        max_val = pivot_filtered.loc[idx].max()
        if max_val > 0:
            pivot_filtered.loc[idx] = pivot_filtered.loc[idx] / max_val
    
    # Étape 5: Trier les langues par groupe
    language_groups = {
        'A': ['ba', 'be', 'be-tarask', 'bg', 'cv', 'kk', 'ky', 'mk', 'mn', 'ru', 'rue', 'sah', 'sr', 'tg', 'tt', 'tyv', 'uk'],
        'B': ['bxr', 'ce', 'myv', 'os', 'udm'],
        'C': ['ab', 'kbd', 'koi', 'kv'],
        'D': ['mhr']
    }
    
    # Créer un ordre pour les langues
    language_order = []
    for group in ['A', 'B', 'C', 'D']:
        for lang in language_groups[group]:
            if lang in pivot_filtered.index:
                language_order.append(lang)
    
    # Trier le DataFrame selon cet ordre
    ordered_langs = [l for l in language_order if l in pivot_filtered.index]
    if ordered_langs:
        pivot_filtered = pivot_filtered.loc[ordered_langs]
    
    # Générer la heatmap
    plt.figure(figsize=(20, 20))
    sns.heatmap(
        pivot_filtered, 
        cmap='viridis', 
        annot=True, 
        fmt='.2f', 
        linewidths=.5
    )
    plt.title('Caractères cyrilliques distinctifs par langue')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/distinctive_characters.png", dpi=300)
    plt.close()
    
    return "Analyse des caractéristiques textuelles terminée!"


def analyze_zipf_law(articles_df, output_dir='results/figures/distribution'):
    """Analyse approfondie de la loi de Zipf pour les principales langues"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Sélectionner les langues les plus représentées
    languages = articles_df['language'].unique().tolist()
    
    print(f"  Analyse de la loi de Zipf pour les langues: {', '.join(languages)}")
    
    for lang in languages:
        # Filtrer les articles pour cette langue
        lang_df = articles_df[articles_df['language'] == lang]
        
        if len(lang_df) < 10:  # Ignorer les langues avec trop peu d'articles
            continue
            
        # Concaténer tous les textes
        all_text = ' '.join(lang_df['text'].fillna(''))
        
        # Tokeniser simplement par espaces (approx.)
        words = re.findall(r'\b\w+\b', all_text.lower())
        
        # Compter les fréquences
        word_counts = Counter(words)
        most_common = word_counts.most_common(100)
        
        # Créer un DataFrame pour le graphique
        zipf_df = pd.DataFrame(most_common, columns=['word', 'frequency'])
        zipf_df['rank'] = range(1, len(zipf_df) + 1)
        
        # Log-log plot plus détaillé
        plt.figure(figsize=(14, 10))
        
        # Données observées
        plt.loglog(zipf_df['rank'], zipf_df['frequency'], 'o', markersize=5, 
                  label='Données observées', alpha=0.7)
        
        # Loi de Zipf théorique (1/rank)
        ideal_zipf = zipf_df['frequency'].iloc[0] / zipf_df['rank']
        plt.loglog(zipf_df['rank'], ideal_zipf, 'r-', linewidth=2, 
                  label='Loi de Zipf idéale (1/rang)')
        
        plt.title(f'Loi de Zipf pour la langue {lang} - Comparaison avec le modèle théorique')
        plt.xlabel('Rang (log)')
        plt.ylabel('Fréquence (log)')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.3)
        
        # Ajouter des annotations pour les mots les plus fréquents
        for i in range(min(5, len(zipf_df))):
            plt.annotate(zipf_df['word'][i], 
                         (zipf_df['rank'][i], zipf_df['frequency'][i]),
                         xytext=(5, 5), textcoords='offset points')
        
        plt.savefig(f"{output_dir}/zipf_detailed_{lang}.png", dpi=300)
        plt.close()
        
        print(f"    Graphique créé pour {lang}")
    
    # Créer une comparaison des distributions entre les langues
    if len(languages) > 1:
        plt.figure(figsize=(14, 10))
        
        for lang in languages:
            lang_df = articles_df[articles_df['language'] == lang]
            if len(lang_df) < 10:
                continue
                
            all_text = ' '.join(lang_df['text'].fillna(''))
            words = re.findall(r'\b\w+\b', all_text.lower())
            word_counts = Counter(words)
            most_common = word_counts.most_common(100)
            
            zipf_df = pd.DataFrame(most_common, columns=['word', 'frequency'])
            zipf_df['rank'] = range(1, len(zipf_df) + 1)
            
            # Normaliser par la fréquence maximale pour comparer les langues
            zipf_df['norm_frequency'] = zipf_df['frequency'] / zipf_df['frequency'].iloc[0]
            
            plt.loglog(zipf_df['rank'], zipf_df['norm_frequency'], 'o-', markersize=3, 
                      alpha=0.6, label=f'{lang}')
        
        plt.title('Comparaison de la loi de Zipf entre les langues')
        plt.xlabel('Rang (log)')
        plt.ylabel('Fréquence normalisée (log)')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.savefig(f"{output_dir}/zipf_comparison.png", dpi=300)
        plt.close()
        
        print(f"  Graphique de comparaison des langues créé")
    
    return "Analyse approfondie de la loi de Zipf terminée!"


def analyze_text_length_correlations(articles_df, output_dir='results/figures/distribution'):
    """Analyse des corrélations entre la longueur des textes et d'autres variables"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("  Analyse des corrélations de longueur...")
    
    # 1. Relation entre longueur et langue
    plt.figure(figsize=(14, 10))
    
    languages = articles_df['language'].unique().tolist()
    filtered_df = articles_df[articles_df['language'].isin(languages)]
    
    # Tracer la relation entre longueur et langue
    ax = sns.boxplot(x='language', y='token_count', data=filtered_df, palette='viridis')
    plt.title('Distribution des longueurs de textes par langue')
    plt.xlabel('Langue')
    plt.ylabel('Nombre de tokens')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/token_count_by_language.png", dpi=300)
    plt.close()
    
    # 2. Relation entre longueur et type de source
    plt.figure(figsize=(12, 8))
    
    # Vérifier si la colonne source_type existe, sinon la créer
    if 'source_type' not in articles_df.columns:
        articles_df['source_type'] = 'Aléatoire'
        articles_df.loc[articles_df['category'].str.contains('\(Sous-catégorie\)', na=False), 'source_type'] = 'Sous-catégorie'
        articles_df.loc[(~articles_df['category'].str.contains('\(Sous-catégorie\)', na=False)) & 
                      (articles_df['category'] != 'Random'), 'source_type'] = 'Catégorie principale'
    
    # Créer le boxplot
    ax = sns.boxplot(x='source_type', y='token_count', data=articles_df, palette='Set2')
    plt.title('Distribution des longueurs de textes par type de source')
    plt.xlabel('Type de source')
    plt.ylabel('Nombre de tokens')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/token_count_by_source.png", dpi=300)
    plt.close()
    
    # 3. Heatmap de longueur moyenne par langue et type de source
    plt.figure(figsize=(16, 12))
    
    # Calculer la longueur moyenne par langue et type de source
    pivot_df = articles_df.pivot_table(
        index='language', 
        columns='source_type', 
        values='token_count',
        aggfunc='mean'
    )
    
    # Filtrer pour ne garder que les langues avec des données suffisantes
    pivot_filtered = pivot_df.dropna(how='all').head(20)
    
    # Créer la heatmap
    sns.heatmap(pivot_filtered, cmap='viridis', annot=True, fmt='.0f')
    plt.title('Longueur moyenne des textes par langue et type de source')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/token_count_heatmap.png", dpi=300)
    plt.close()
    
    # 4. Scatter plot pour les langues majeures
    plt.figure(figsize=(12, 10))
    
    # Sélectionner les 5 langues principales
    major_langs = articles_df['language'].value_counts().head(5).index.tolist()
    major_df = articles_df[articles_df['language'].isin(major_langs)]
    
    # Échantillonner pour plus de clarté visuelle si nécessaire
    if len(major_df) > 2000:
        sample_df = major_df.sample(2000, random_state=42)
    else:
        sample_df = major_df
    
    # Créer le scatter plot
    sns.scatterplot(
        data=sample_df,
        x='token_count',
        y='language',
        hue='source_type',
        palette='deep',
        alpha=0.7,
        s=50  # taille des points
    )
    
    plt.title('Répartition des longueurs par langue et type de source')
    plt.xlabel('Nombre de tokens')
    plt.ylabel('Langue')
    plt.legend(title='Type de source')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/length_scatter_by_language.png", dpi=300)
    plt.close()
    
    print("  Visualisations des corrélations de longueur créées")
    
    return "Analyse des corrélations de longueur terminée!"


def visualize_corpus_by_category(articles_df, output_dir='results/figures/distribution'):
    """Visualise la répartition des articles par catégorie et par langue"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Créer une colonne pour le type de source
    articles_df['source_type'] = 'Aléatoire'
    articles_df.loc[articles_df['category'].str.contains('\(Sous-catégorie\)', na=False), 'source_type'] = 'Sous-catégorie'
    articles_df.loc[(~articles_df['category'].str.contains('\(Sous-catégorie\)', na=False)) & 
                   (articles_df['category'] != 'Random'), 'source_type'] = 'Catégorie principale'
    
    # Graphique global de la répartition
    source_counts = articles_df['source_type'].value_counts()
    
    plt.figure(figsize=(10, 8))
    plt.pie(source_counts, labels=source_counts.index, autopct='%1.1f%%', 
            startangle=90, shadow=True, explode=[0.05]*len(source_counts),
            colors=sns.color_palette('Set2'))
    plt.title('Répartition globale des articles par type de source')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/global_source_distribution.png", dpi=300)
    plt.close()
    
    # Répartition par groupe de langue
    source_by_group = pd.crosstab(articles_df['language_group'], articles_df['source_type'])
    source_by_group_pct = source_by_group.div(source_by_group.sum(axis=1), axis=0) * 100
    
    # Créer un grouped bar chart
    plt.figure(figsize=(14, 8))
    source_by_group_pct.plot(kind='bar', stacked=True, colormap='Set2')
    plt.title('Répartition des sources par groupe de langue')
    plt.xlabel('Groupe de langue')
    plt.ylabel('Pourcentage')
    plt.legend(title='Type de source')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/source_by_language_group.png", dpi=300)
    plt.close()
    
    # Création d'une visualisation interactive
    try:
        import plotly.express as px
        import plotly.io as pio
        
        # Tableau croisé avec les langues en ligne et les types de source en colonne
        source_by_lang = pd.crosstab(articles_df['language'], articles_df['source_type'])
        source_by_lang_pct = source_by_lang.div(source_by_lang.sum(axis=1), axis=0) * 100
        
        # Convertir en format long pour Plotly
        source_by_lang_pct_reset = source_by_lang_pct.reset_index()
        source_by_lang_long = pd.melt(
            source_by_lang_pct_reset, 
            id_vars=['language'], 
            value_vars=source_by_lang_pct.columns,
            var_name='source_type', 
            value_name='percentage'
        )
        
        # Créer un graphique en barres interactif
        fig = px.bar(
            source_by_lang_long,
            x='language',
            y='percentage',
            color='source_type',
            barmode='stack',
            title='Répartition des sources par langue',
            labels={'language': 'Langue', 'percentage': 'Pourcentage', 'source_type': 'Type de source'}
        )
        
        # Sauvegarder au format HTML pour l'interaction
        pio.write_html(fig, f"{output_dir}/interactive_source_by_language.html")
        
        # Également en format image pour le rapport
        pio.write_image(fig, f"{output_dir}/source_by_language.png", scale=2)

    except ImportError:
        print("Plotly n'est pas installé. Visualisation interactive ignorée.")
    
    return "Visualisation par catégorie terminée!"


def main():
    """Fonction principale pour l'analyse et la visualisation"""
    # Créer les dossiers de sortie
    output_dir = 'results/figures/distribution'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Chargement des articles...")
    articles_df = load_all_articles()
    print(f"Corpus chargé: {len(articles_df)} articles, {articles_df['language'].nunique()} langues")
    
    print("\n1. Exploration des statistiques générales...")
    explore_corpus_stats(articles_df, output_dir=output_dir)
    
    print("\n2. Analyse des caractéristiques textuelles...")
    analyze_text_characteristics(articles_df, output_dir=output_dir)
    
    print("\n3. Analyse approfondie de la loi de Zipf...")
    analyze_zipf_law(articles_df, output_dir=output_dir)

    print("\n4. Analyse des corrélations entre longueur et autres variables...")
    analyze_text_length_correlations(articles_df, output_dir=output_dir)
    
    print("\n5. Visualisation par catégorie...")
    visualize_corpus_by_category(articles_df, output_dir=output_dir)
    
    print("\nToutes les visualisations ont été générées avec succès!")
    print("Les résultats sont consultables dans le dossier 'reports/figures'")


if __name__ == "__main__":
    main()