import pandas as pd
import numpy as np
import os
import glob
import re
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib


class CyrillicCorpusCleaner:
    """
    Classe pour nettoyer, valider et préparer un corpus de langues écrites en cyrillique.
    
    Cette classe implémente plusieurs méthodes de nettoyage et validation:
        - détection et suppression des doublons
        - identification et gestion des valeurs aberrantes
        - nettoyage du texte
        - normalisation des données
    """
    
    def __init__(self, input_dir='data/raw/final_corpus', output_dir='data/processed/cleaned', 
             figures_dir='results/figures/cleaning', metrics_dir='results/metrics/cleaning'):
        
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.figures_dir = figures_dir
        self.metrics_dir = metrics_dir
        
        # Créer les répertoires s'ils n'existent pas
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)
        os.makedirs(f"{metrics_dir}/outliers", exist_ok=True)
        
        # Statistiques de nettoyage
        self.cleaning_stats = {
            'original_count': 0,
            'duplicates_removed': 0,
            'outliers_removed': 0,
            'cleaned_count': 0,
            'by_language': {}
        }
    

    def load_corpus(self):
        """
        Charge tous les articles du corpus depuis les fichiers CSV.
        
        Returns:
            DataFrame pandas contenant tous les articles
        """
        print("Chargement du corpus brut...")
        
        all_files = glob.glob(f'{self.input_dir}/*_articles.csv')
        all_dfs = []
        
        for file in all_files:
            lang_code = os.path.basename(file).split('_')[0]
            
            try:
                df = pd.read_csv(file)
                
                # Vérifier les colonnes nécessaires
                if 'text' not in df.columns:
                    print(f"  Attention: colonne 'text' manquante dans {file}, fichier ignoré")
                    continue
                
                # Ajouter la langue si non présente
                if 'language' not in df.columns:
                    df['language'] = lang_code
                
                # Ajouter à la liste
                all_dfs.append(df)
                print(f"  Chargé {len(df)} articles pour {lang_code}")
                
                # Mettre à jour les statistiques
                if lang_code not in self.cleaning_stats['by_language']:
                    self.cleaning_stats['by_language'][lang_code] = {
                        'original': len(df),
                        'duplicates': 0,
                        'outliers': 0,
                        'cleaned': 0
                    }
                else:
                    self.cleaning_stats['by_language'][lang_code]['original'] += len(df)
                
            except Exception as e:
                print(f"  Erreur lors du chargement de {file}: {e}")
        
        # Fusionner tous les DataFrames
        if all_dfs:
            corpus_df = pd.concat(all_dfs, ignore_index=True)
            self.cleaning_stats['original_count'] = len(corpus_df)
            print(f"Corpus chargé: {len(corpus_df)} articles au total")
            return corpus_df
        else:
            print("Aucun article n'a pu être chargé!")
            return pd.DataFrame()
        
    
    def save_cleaned_corpus(self, df):
        """
        Sauvegarde le corpus nettoyé.
        
        Args:
            df: DataFrame du corpus nettoyé
        """
        print("\nSauvegarde du corpus nettoyé...")
        
        # Vérifier si le DataFrame est vide
        if df.empty:
            print("  Aucun article à sauvegarder!")
            return
        
        # Sauvegarder le corpus complet
        df.to_csv(f"{self.output_dir}/cleaned_corpus.csv", index=False)
        
        # Sauvegarder par langue
        for lang in df['language'].unique():
            lang_df = df[df['language'] == lang]
            lang_df.to_csv(f"{self.output_dir}/{lang}_cleaned_articles.csv", index=False)
            
            # Mettre à jour les statistiques
            self.cleaning_stats['by_language'][lang]['cleaned'] = len(lang_df)
        
        # Mettre à jour les statistiques globales
        self.cleaning_stats['cleaned_count'] = len(df)
        
        # Sauvegarder les statistiques de nettoyage
        with open(f"{self.metrics_dir}/cleaning_stats.txt", 'w') as f:
            f.write("=== Statistiques de nettoyage du corpus ===\n\n")
            f.write(f"Nombre d'articles originaux: {self.cleaning_stats['original_count']}\n")
            f.write(f"Doublons supprimés: {self.cleaning_stats['duplicates_removed']}\n")
            f.write(f"Outliers supprimés: {self.cleaning_stats['outliers_removed']}\n")
            f.write(f"Articles conservés: {self.cleaning_stats['cleaned_count']}\n\n")
            
            f.write("Statistiques par langue:\n")
            for lang, stats in self.cleaning_stats['by_language'].items():
                f.write(f"\n{lang}:\n")
                f.write(f"  Articles originaux: {stats.get('original', 0)}\n")
                f.write(f"  Doublons supprimés: {stats.get('duplicates', 0)}\n")
                f.write(f"  Outliers supprimés: {stats.get('outliers', 0)}\n")
                f.write(f"  Articles conservés: {stats.get('cleaned', 0)}\n")
        
        print(f"  Corpus nettoyé sauvegardé dans {self.output_dir}")
        print(f"  {len(df)} articles au total")


    def detect_duplicates(self, df):
        """
        Détecte et supprime les doublons dans le corpus.
        
        Args:
            df: DataFrame du corpus
            
        Returns:
            DataFrame sans doublons
        """
        print("\nDétection des doublons...")
        
        # Vérifier si le DataFrame est vide
        if df.empty:
            return df
        
        # S'assurer que les colonnes nécessaires existent
        if 'text' not in df.columns or 'language' not in df.columns:
            print("  Colonnes nécessaires manquantes, impossible de détecter les doublons")
            return df
        
        # Nombre d'articles avant dédoublonnage
        initial_count = len(df)
        
        # 1. Supprimer les doublons exacts (même texte, même langue)
        df_no_exact_dupes = df.drop_duplicates(subset=['text', 'language'], keep='first')
        exact_dupes_removed = initial_count - len(df_no_exact_dupes)
        
        # 2. Créer un hash du texte normalisé pour détecter les doublons similaires
        def create_hash(text):
            if not isinstance(text, str):
                return ""
            # Normaliser le texte: supprimer la ponctuation, mettre en minuscules
            text = re.sub(r'[^\w\s]', '', text.lower())
            # Créer un hash du texte normalisé
            return hashlib.md5(text.encode('utf-8')).hexdigest()
        
        # Appliquer la fonction de hachage
        df_no_exact_dupes['text_hash'] = df_no_exact_dupes['text'].apply(create_hash)
        
        # Supprimer les doublons basés sur le hash
        df_no_dupes = df_no_exact_dupes.drop_duplicates(subset=['text_hash', 'language'], keep='first')
        hash_dupes_removed = len(df_no_exact_dupes) - len(df_no_dupes)
        
        # Supprimer la colonne de hash
        df_no_dupes = df_no_dupes.drop(columns=['text_hash'])
        
        # Calculer le nombre total de doublons supprimés
        total_dupes_removed = exact_dupes_removed + hash_dupes_removed
        self.cleaning_stats['duplicates_removed'] = total_dupes_removed
        
        # Mettre à jour les statistiques par langue
        for lang in df_no_dupes['language'].unique():
            initial_lang_count = len(df[df['language'] == lang])
            final_lang_count = len(df_no_dupes[df_no_dupes['language'] == lang])
            dupes_removed = initial_lang_count - final_lang_count
            
            self.cleaning_stats['by_language'][lang]['duplicates'] = dupes_removed
        
        print(f"  {exact_dupes_removed} doublons exacts supprimés")
        print(f"  {hash_dupes_removed} doublons similaires supprimés")
        print(f"  {len(df_no_dupes)} articles uniques conservés")
        
        return df_no_dupes
    

    def clean_text(self, df):
        """
        Nettoie le texte des articles.
        
        Args:
            df: DataFrame du corpus
            
        Returns:
            DataFrame avec texte nettoyé
        """
        print("\nNettoyage du texte des articles...")
        
        # Vérifier si le DataFrame est vide
        if df.empty:
            return df
        
        # S'assurer que la colonne de texte existe
        if 'text' not in df.columns:
            print("  Colonne 'text' manquante, impossible de nettoyer le texte")
            return df
        
        # Fonction de nettoyage du texte
        def clean_article_text(text):
            if not isinstance(text, str):
                return ""
                
            # 1. Supprimer les caractères HTML/XML
            text = re.sub(r'<[^>]+>', '', text)
            
            # 2. Supprimer les URL
            text = re.sub(r'http[s]?://\S+', '', text)
            
            # 3. Supprimer les espaces multiples
            text = re.sub(r'\s+', ' ', text)
            
            # 4. Supprimer les lignes vides
            text = re.sub(r'\n\s*\n', '\n', text)
            
            # 5. Supprimer les espaces en début et fin
            text = text.strip()
            
            return text
        
        # Appliquer la fonction de nettoyage
        df['text'] = df['text'].apply(clean_article_text)
        
        # Recalculer le nombre de tokens après nettoyage
        df['token_count'] = df['text'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
        
        # Supprimer les articles dont le texte est devenu vide après nettoyage
        empty_text_count = len(df[df['text'] == ""])
        df = df[df['text'] != ""]
        
        print(f"  {empty_text_count} articles avec texte vide supprimés après nettoyage")
        print(f"  {len(df)} articles conservés")
        
        return df
    
        
    def normalize_special_characters(self, df):
        """
        Normalise les caractères spéciaux dans les textes.
        
        Args:
            df: DataFrame du corpus
                
        Returns:
            DataFrame avec les caractères normalisés
        """
        print("\nNormalisation des caractères spéciaux...")
        
        # Vérifier si le DataFrame est vide
        if df.empty:
            return df
        
        # S'assurer que la colonne de texte existe
        if 'text' not in df.columns:
            print("  Colonne 'text' manquante, impossible de normaliser les caractères")
            return df
        
        # Dictionnaire de correspondance pour les caractères à normaliser
        char_map = {
            # Différents types d'apostrophes et guillemets
            ''': "'", ''': "'", '`': "'", '´': "'", '"': '"', '"': '"', '„': '"',
            # Différents types de tirets et traits d'union
            '–': '-', '—': '-', '―': '-', '‐': '-',
            # Caractères d'espacement
            '\u200b': '', '\u200c': '', '\u200d': '', '\xa0': ' ',
            # Caractères de contrôle et autres caractères invisibles
            '\t': ' ', '\r': ' ', '\u2028': ' ', '\u2029': ' ',
            # Autres normalisations potentiellement utiles
            '…': '...', '№': 'No'
        }
        
        # Fonction pour normaliser un texte
        def normalize_text(text):
            if not isinstance(text, str):
                return ""
            
            # Remplacer les caractères selon le dictionnaire
            for old_char, new_char in char_map.items():
                text = text.replace(old_char, new_char)
            
            # Normaliser les espaces multiples
            text = re.sub(r'\s+', ' ', text)
            
            return text
        
        # Appliquer la normalisation
        orig_text = df['text'].copy()
        df['text'] = df['text'].apply(normalize_text)
        
        # Compter combien de textes ont été modifiés
        modified_count = (orig_text != df['text']).sum()
        
        print(f"  {modified_count} articles modifiés par la normalisation des caractères")
        
        return df
    

    def clean_wiki_markup(self, df):
        """
        Supprime les balises wiki résiduelles des articles.
        
        Args:
            df: DataFrame du corpus
                
        Returns:
            DataFrame avec les balises wiki nettoyées
        """
        print("\nNettoyage des balises wiki résiduelles...")
        
        # Vérifier si le DataFrame est vide
        if df.empty:
            return df
        
        # S'assurer que la colonne de texte existe
        if 'text' not in df.columns:
            print("  Colonne 'text' manquante, impossible de nettoyer les balises wiki")
            return df
        
        # Patrons de balises wiki courantes
        wiki_patterns = [
            r'\[\[.*?\]\]',                   # Liens internes [[lien]]
            r'\{\{.*?\}\}',                   # Templates {{template}}
            r'\[\s*https?://\S+\s+[^\]]+\]',  # Liens externes avec texte [http://... texte]
            r'\[\s*https?://\S+\s*\]',        # Liens externes simples [http://...]
            r"'{2,5}",                        # Formatage gras/italique '''/'''/''
            r'<ref[^>]*>.*?</ref>',           # Références <ref>...</ref>
            r'<ref[^/]*/>',                   # Références vides <ref />
            r'<!--.*?-->',                    # Commentaires HTML <!-- ... -->
            r'<(gallery|blockquote|nowiki|pre|source|syntaxhighlight|poem|includeonly|noinclude)[^>]*>.*?</\1>',
                                              # Balises de blocs spéciaux
            r'__(NOTOC|NOEDITSECTION|FORCETOC|TOC|NOCONTENTS|NEWSECTIONLINK)__',
                                              # Commandes magiques
            r'<[^>]*>',                       # Autres balises HTML
        ]
        
        # Fonction pour nettoyer les balises wiki
        def clean_wiki_markup_from_text(text):
            if not isinstance(text, str):
                return ""
            
            # Appliquer tous les patrons
            for pattern in wiki_patterns:
                text = re.sub(pattern, ' ', text)
            
            # Nettoyer les catégories et interwikis qui peuvent apparaître à la fin
            text = re.sub(r'\[\[(Category|Catégorie|Категория|Категорія|Катэгорыя):[^\]]+\]\]', '', text, flags=re.IGNORECASE)
            
            # Nettoyer les sections vides (==Section== suivie d'une autre section ou de la fin)
            text = re.sub(r'==+\s*[^=]+\s*==+\s*(?===|\Z)', '', text)
            
            # Normaliser les espaces multiples et les sauts de ligne
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            
            return text
        
        # Appliquer le nettoyage
        orig_text = df['text'].copy()
        df['text'] = df['text'].apply(clean_wiki_markup_from_text)
        
        # Compter combien de textes ont été modifiés
        modified_count = (orig_text != df['text']).sum()
        
        print(f"  {modified_count} articles modifiés par le nettoyage des balises wiki")
        
        return df
    

    def detect_multilingual_content(self, df):
        """
        Détecte et traite les articles contenant du contenu dans plusieurs langues.
        
        Args:
            df: DataFrame du corpus
                
        Returns:
            DataFrame avec les articles multilingues identifiés et traités
        """
        print("\nDétection des articles multilingues...")
        
        # Vérifier si le DataFrame est vide
        if df.empty:
            return df
        
        # S'assurer que les colonnes nécessaires existent
        if 'text' not in df.columns or 'language' not in df.columns:
            print("  Colonnes nécessaires manquantes, impossible de détecter le contenu multilingue")
            return df
        
        # Fonction pour détecter les langues dans un texte
        def detect_languages(text, expected_lang):
            if not isinstance(text, str) or len(text) < 50:
                return [expected_lang]
            
            try:
                # Option 1: Utiliser la bibliothèque langdetect (à installer avec pip)
                from langdetect import detect_langs
                
                # Détecter jusqu'à 3 langues potentielles
                detected = detect_langs(text[:5000])  # Limiter à 5000 caractères pour la performance
                
                # Filtrer les langues avec une probabilité > 0.2
                significant_langs = [lang.lang for lang in detected if lang.prob > 0.2]
                
                return significant_langs
            except:
                # Si langdetect n'est pas disponible, utiliser une heuristique simple basée sur les alphabets
                # Approche moins précise, mais ne nécessite pas de bibliothèque externe
                
                # Définir les plages de caractères
                cyrillic = set(range(0x0400, 0x04FF))
                latin = set(range(0x0041, 0x007A))
                
                # Compter les caractères dans chaque plage
                cyrillic_count = sum(1 for c in text if ord(c) in cyrillic)
                latin_count = sum(1 for c in text if ord(c) in latin)
                
                # Déterminer les langues présentes
                langs = []
                if cyrillic_count > 20:  # Seuil arbitraire
                    langs.append('cyrillic')
                if latin_count > 20:
                    langs.append('latin')
                
                return langs if langs else [expected_lang]
        
        # Ajouter une colonne pour indiquer les langues détectées
        df['detected_languages'] = df.apply(
            lambda row: detect_languages(row['text'], row['language']), axis=1
        )
        
        # Identifier les articles multilingues
        df['is_multilingual'] = df['detected_languages'].apply(lambda langs: len(langs) > 1)
        
        multilingual_count = df['is_multilingual'].sum()
        print(f"  {multilingual_count} articles multilingues détectés")
        
        if multilingual_count > 0:
            # Option 1: Garder uniquement le texte dans la langue principale (méthode avancée)
            # Cela nécessiterait une bibliothèque comme polyglot ou spaCy
            # Option 2: Simplement marquer les articles pour une inspection manuelle
            print("  Les articles multilingues ont été marqués pour analyse ultérieure")
        
        return df
    
    
    def detect_outliers(self, df):
        """
        Détecte et gère les valeurs aberrantes dans le corpus.
        
        Args:
            df: DataFrame du corpus
            
        Returns:
            DataFrame sans outliers
        """
        print("\nDétection des outliers...")
        
        # Vérifier si le DataFrame est vide
        if df.empty:
            return df
        
        # S'assurer que les colonnes nécessaires existent
        if 'token_count' not in df.columns or 'language' not in df.columns:
            print("  Ajout de la colonne token_count manquante")
            # Créer une colonne de nombre de tokens si elle n'existe pas
            df['token_count'] = df['text'].apply(lambda x: len(str(x).split()) if isinstance(x, str) else 0)
        
        # DataFrame pour stocker les articles sans outliers
        df_no_outliers = pd.DataFrame()
        
        # Traiter chaque langue séparément
        for lang in df['language'].unique():
            lang_df = df[df['language'] == lang]
            
            # Calculer les quartiles de token_count
            q1 = lang_df['token_count'].quantile(0.25)
            q3 = lang_df['token_count'].quantile(0.75)
            iqr = q3 - q1
            
            # Définir les limites pour les outliers
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Identifier les outliers
            outliers = lang_df[(lang_df['token_count'] < lower_bound) | 
                              (lang_df['token_count'] > upper_bound)]
            
            # Visualiser la distribution et les outliers
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=lang_df['token_count'])
            plt.title(f'Distribution des tokens pour {lang}')
            plt.xlabel('Nombre de tokens')
            plt.savefig(f"{self.metrics_dir}/outliers/outliers_boxplot_{lang}.png")
            plt.close()
            
            # Conserver les articles non aberrants
            normal_articles = lang_df[(lang_df['token_count'] >= lower_bound) & 
                                    (lang_df['token_count'] <= upper_bound)]
            
            # Ajouter au DataFrame final
            df_no_outliers = pd.concat([df_no_outliers, normal_articles], ignore_index=True)
            
            # Mettre à jour les stats
            outliers_count = len(outliers)
            self.cleaning_stats['by_language'][lang]['outliers'] = outliers_count
            self.cleaning_stats['outliers_removed'] += outliers_count
            
            print(f"  {lang}: {outliers_count} outliers identifiés et supprimés")
            print(f"    Limites: {lower_bound:.1f} - {upper_bound:.1f} tokens")
        
        print(f"  Total: {self.cleaning_stats['outliers_removed']} outliers supprimés")
        print(f"  {len(df_no_outliers)} articles conservés")
        
        return df_no_outliers
    
    
    def normalize_categories(self, df):
        """
        Normalise les catégories des articles.
        
        Args:
            df: DataFrame du corpus
            
        Returns:
            DataFrame avec catégories normalisées
        """
        print("\nNormalisation des catégories...")
        
        # Vérifier si le DataFrame est vide
        if df.empty:
            return df
        
        # S'assurer que la colonne de catégorie existe
        if 'category' not in df.columns:
            print("  Colonne 'category' manquante, impossible de normaliser les catégories")
            return df
        
        # Remplacer les valeurs nulles
        df['category'] = df['category'].fillna('Unknown')
        
        # Extraire la catégorie principale (sans sous-catégorie)
        df['main_category'] = df['category'].apply(
            lambda x: str(x).split(' (')[0] if isinstance(x, str) and ' (' in x else x
        )
        
        # Normaliser le type de source
        df['source_type'] = 'Aléatoire'
        df.loc[df['category'].str.contains('\(Sous-catégorie\)', na=False), 'source_type'] = 'Sous-catégorie'
        df.loc[(~df['category'].str.contains('\(Sous-catégorie\)', na=False)) & 
              (df['category'] != 'Random'), 'source_type'] = 'Catégorie principale'
        
        print(f"  Catégories normalisées pour {len(df)} articles")
        
        return df
    
    
    def visualize_cleaning_impact(self):
        """
            Visualise l'impact du nettoyage sur le corpus.
        """
        print("\nCréation des visualisations de l'impact du nettoyage...")
        
        # 1. Graphique des articles avant/après nettoyage
        plt.figure(figsize=(12, 8))
        
        # Préparer les données
        languages = list(self.cleaning_stats['by_language'].keys())
        original_counts = [self.cleaning_stats['by_language'][lang].get('original', 0) for lang in languages]
        cleaned_counts = [self.cleaning_stats['by_language'][lang].get('cleaned', 0) for lang in languages]
        
        # Trier par nombre d'articles originaux
        sorted_indices = np.argsort(original_counts)[::-1]
        languages = [languages[i] for i in sorted_indices]
        original_counts = [original_counts[i] for i in sorted_indices]
        cleaned_counts = [cleaned_counts[i] for i in sorted_indices]
        
        # Créer le graphique
        x = np.arange(len(languages))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(14, 8))
        rects1 = ax.bar(x - width/2, original_counts, width, label='Original', color='skyblue')
        rects2 = ax.bar(x + width/2, cleaned_counts, width, label='Nettoyé', color='lightgreen')
        
        ax.set_title('Impact du nettoyage sur le nombre d\'articles par langue')
        ax.set_xlabel('Langue')
        ax.set_ylabel('Nombre d\'articles')
        ax.set_xticks(x)
        ax.set_xticklabels(languages, rotation=45, ha='right')
        ax.legend()
        
        # Ajouter les valeurs sur les barres
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height}',
                            xy=(rect.get_x() + rect.get_width()/2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')
        
        autolabel(rects1)
        autolabel(rects2)
        
        plt.tight_layout()
        plt.savefig(f"{self.figures_dir}/cleaning_impact.png", dpi=300)
        plt.close()
        
        # 2. Graphique des types de nettoyage
        plt.figure(figsize=(12, 8))
        
        # Préparer les données pour les 10 premières langues
        top_languages = languages[:10]
        dupes_removed = [self.cleaning_stats['by_language'][lang].get('duplicates', 0) for lang in top_languages]
        outliers_removed = [self.cleaning_stats['by_language'][lang].get('outliers', 0) for lang in top_languages]
        
        # Créer le graphique
        x = np.arange(len(top_languages))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(14, 8))
        rects1 = ax.bar(x - width/2, dupes_removed, width, label='Doublons', color='salmon')
        rects2 = ax.bar(x + width/2, outliers_removed, width, label='Outliers', color='lightblue')
        
        ax.set_title('Types d\'éléments supprimés par langue')
        ax.set_xlabel('Langue')
        ax.set_ylabel('Nombre d\'articles supprimés')
        ax.set_xticks(x)
        ax.set_xticklabels(top_languages, rotation=45, ha='right')
        ax.legend()
        
        # Ajouter les valeurs sur les barres
        autolabel(rects1)
        autolabel(rects2)
        
        plt.tight_layout()
        plt.savefig(f"{self.figures_dir}/cleaning_types.png", dpi=300)
        plt.close()
        
        # 3. Diagramme circulaire global
        plt.figure(figsize=(10, 8))
        
        # Préparer les données
        labels = ['Articles conservés', 'Doublons supprimés', 'Outliers supprimés']
        sizes = [
            self.cleaning_stats['cleaned_count'],
            self.cleaning_stats['duplicates_removed'],
            self.cleaning_stats['outliers_removed']
        ]
        
        # Créer le graphique
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, 
                colors=['lightgreen', 'salmon', 'lightblue'],
                explode=(0.1, 0, 0))
        plt.axis('equal')
        plt.title('Répartition globale des articles après nettoyage')
        plt.tight_layout()
        plt.savefig(f"{self.figures_dir}/cleaning_pie.png", dpi=300)
        plt.close()
        
        print("  Visualisations de l'impact du nettoyage créées")

    
    def process_corpus(self):
        """
            Traite l'ensemble du corpus: chargement, nettoyage, sauvegarde.
        """
        # 1. Charger le corpus
        corpus_df = self.load_corpus()
        
        if corpus_df.empty:
            print("Aucun article à traiter. Arrêt du traitement.")
            return
        
        # 2. Détecter et supprimer les doublons
        corpus_df = self.detect_duplicates(corpus_df)
        
        # 3. Nettoyer le texte de base
        corpus_df = self.clean_text(corpus_df)
        
        # 4. Nouvelles étapes de nettoyage
        corpus_df = self.normalize_special_characters(corpus_df)
        corpus_df = self.detect_multilingual_content(corpus_df)
        corpus_df = self.clean_wiki_markup(corpus_df)
        
        # 5. Détecter et gérer les outliers
        corpus_df = self.detect_outliers(corpus_df)
        
        # 6. Normaliser les catégories
        corpus_df = self.normalize_categories(corpus_df)
        
        # 7. Sauvegarder le corpus nettoyé
        self.save_cleaned_corpus(corpus_df)
        
        # 8. Visualiser l'impact du nettoyage
        self.visualize_cleaning_impact()
        
        print("\nTraitement du corpus terminé avec succès!")
        print(f"Corpus original: {self.cleaning_stats['original_count']} articles")
        print(f"Corpus nettoyé: {self.cleaning_stats['cleaned_count']} articles")
        print(f"Réduction: {(1 - self.cleaning_stats['cleaned_count']/self.cleaning_stats['original_count'])*100:.1f}%")


# Lancer le nettoyage
if __name__ == "__main__":
    cleaner = CyrillicCorpusCleaner()
    cleaner.process_corpus()