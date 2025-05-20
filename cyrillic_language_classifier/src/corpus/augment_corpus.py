import pandas as pd
import numpy as np
import random
import os
import re
from collections import Counter
import nltk
from nltk.util import ngrams
import glob


# Téléchargement des ressources NLTK nécessaires
try:
    nltk.download('punkt', quiet=True)
except:
    print("Impossible de télécharger les ressources NLTK. Continuer sans...")


class CyrillicDataAugmenter:
    """Classe pour l'augmentation de données dans les langues cyrilliques"""
    
    def __init__(self, input_dir='data/processed/merged', output_dir='data/processed/augmented'):
        """Initialise l'augmenteur de données"""
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Dictionnaires pour stocker les modèles de langue par langue
        self.char_ngram_models = {}  # modèles de n-grammes de caractères
        self.word_ngram_models = {}  # modèles de n-grammes de mots
        self.articles_by_language = {}  # articles originaux par langue


    def load_data(self):
        """Charge les données depuis les fichiers CSV"""
        print("Chargement des données originales...")
        
        all_files = glob.glob(f'{self.input_dir}/*_articles.csv')
        
        for file in all_files:
            lang_code = os.path.basename(file).split('_')[0]
            
            try:
                df = pd.read_csv(file)
                
                # S'assurer que les colonnes nécessaires existent
                if 'text' not in df.columns:
                    print(f"Colonne 'text' manquante dans {file}, fichier ignoré")
                    continue
                
                # Filtrer les lignes avec du texte non vide
                df = df[df['text'].notna() & (df['text'] != '')]
                
                # Stocker les articles
                self.articles_by_language[lang_code] = df
                
                print(f"  Chargé {len(df)} articles pour {lang_code}")
            except Exception as e:
                print(f"Erreur lors du chargement de {file}: {e}")
        
        print(f"Données chargées pour {len(self.articles_by_language)} langues")
    

    def build_language_models(self):
        """Construit des modèles statistiques pour chaque langue"""
        print("Construction des modèles de langue...")
        
        for lang, articles_df in self.articles_by_language.items():
            print(f"  Modélisation de la langue {lang}...")
            
            # Concaténer tous les textes
            all_text = ' '.join(articles_df['text'].fillna(''))
            
            # Modèle de n-grammes de caractères (pour capturer les schémas orthographiques)
            char_ngram_dict = {}
            for n in range(2, 5):  # Bi-grammes, tri-grammes, et quadri-grammes
                char_ngram_dict[n] = Counter()
                for i in range(len(all_text) - n + 1):
                    ngram = all_text[i:i+n]
                    char_ngram_dict[n][ngram] += 1
            
            self.char_ngram_models[lang] = char_ngram_dict
            
            # Modèle de n-grammes de mots (pour la structure syntaxique)
            try:
                # Tokeniser par mots (approx.)
                words = re.findall(r'\b\w+\b', all_text.lower())
                
                word_ngram_dict = {}
                for n in range(1, 4):  # Unigrammes, bigrammes et trigrammes
                    word_ngram_dict[n] = Counter(ngrams(words, n))
                
                self.word_ngram_models[lang] = word_ngram_dict
            except Exception as e:
                print(f"    Erreur lors de la construction du modèle de mots pour {lang}: {e}")
                # Créer un modèle vide
                self.word_ngram_models[lang] = {n: Counter() for n in range(1, 4)}
            
            print(f"    Modèle construit avec {len(words)} mots")
        
        print("Tous les modèles de langue construits")
    

    def generate_synthetic_text(self, lang, length=250):
        """Génère un texte synthétique basé sur le modèle de langue"""
        if lang not in self.char_ngram_models or lang not in self.word_ngram_models:
            print(f"Pas de modèle disponible pour la langue {lang}")
            return ""
        
        # Approche 1: Utiliser des segments de textes existants
        original_articles = self.articles_by_language[lang]['text'].tolist()
        
        if not original_articles:
            return ""
            
        # Sélectionner quelques articles aléatoires
        selected_articles = random.sample(
            original_articles, 
            min(3, len(original_articles))
        )
        
        # Découper les articles en segments plus courts
        segments = []
        for article in selected_articles:
            words = article.split()
            segment_length = min(50, len(words) // 2)
            
            for i in range(0, len(words), segment_length):
                if i + segment_length <= len(words):
                    segments.append(' '.join(words[i:i+segment_length]))
        
        # Si pas assez de segments, utiliser l'article complet
        if len(segments) < 3:
            segments = selected_articles
        
        # Approche 2: Pour les langues des groupes C et D, utiliser aussi le modèle n-grammes
        if lang in ['ab', 'kbd', 'koi', 'kv', 'mhr']:
            # Utiliser le modèle de trigrammes pour générer du texte à partir de zéro
            char_model = {}
            all_text = ' '.join(original_articles)
            
            # Construire un modèle de trigrammes de caractères
            for i in range(len(all_text) - 3):
                trigram = all_text[i:i+3]
                next_char = all_text[i+3]
                if trigram not in char_model:
                    char_model[trigram] = []
                char_model[trigram].append(next_char)
            
            # Générer du texte avec ce modèle
            if char_model:
                try:
                    # Sélectionner un trigramme aléatoire pour commencer
                    current = random.choice(list(char_model.keys()))
                    generated_text = current
                    
                    # Générer environ 100 caractères
                    for _ in range(100):
                        if current in char_model and char_model[current]:
                            next_char = random.choice(char_model[current])
                            generated_text += next_char
                            current = current[1:] + next_char
                        else:
                            # Si le trigramme n'existe pas, prendre un autre au hasard
                            current = random.choice(list(char_model.keys()))
                    
                    # Ajouter ce texte généré comme segment supplémentaire
                    segments.append(generated_text)
                except IndexError:
                    # En cas d'erreur, ignorer la génération par n-grammes
                    pass
        
        # Mélanger et combiner des segments
        random.shuffle(segments)
        
        # Prendre suffisamment de segments pour atteindre la longueur cible
        synthetic_text = ' '.join(segments[:min(8, len(segments))])
        
        # Ajuster pour obtenir une longueur approximative
        words = synthetic_text.split()
        if len(words) > length:
            synthetic_text = ' '.join(words[:length])
        
        # Répéter la génération jusqu'à atteindre au moins 80% de la longueur cible
        while len(synthetic_text.split()) < 0.8 * length and len(segments) > 5:
            random.shuffle(segments)
            additional_text = ' '.join(segments[:3])  # prendre 3 segments supplémentaires
            synthetic_text += " " + additional_text
            
            # Revérifier qu'on ne dépasse pas la longueur cible
            words = synthetic_text.split()
            if len(words) > length:
                synthetic_text = ' '.join(words[:length])
                break

        return synthetic_text
    

    def generate_synthetic_dataset(self, count_per_language=20):
        """Génère un ensemble de données synthétiques pour toutes les langues"""
        print(f"Génération de {count_per_language} articles synthétiques par langue...")
        
        all_synthetic_articles = []
        
        for lang in self.articles_by_language.keys():
            print(f"  Génération pour {lang}...")
            
            # Calculer la longueur moyenne des articles originaux
            original_lengths = self.articles_by_language[lang]['token_count'].tolist()
            avg_length = int(np.mean(original_lengths)) if original_lengths else 200

            # Ajuster la longueur cible en fonction de la langue
            if lang in ['sr', 'ba', 'ru', 'mk']:
                target_ratio = 0.85  # viser 85% de la longueur originale
            elif lang in ['bxr', 'tyv', 'rue', 'ab', 'be']:
                target_ratio = 0.90  # viser 90% de la longueur originale
            else:
                target_ratio = 0.95  # viser 95% de la longueur originale

            target_length = int(avg_length * target_ratio)
            
            # Générer des articles synthétiques
            for i in range(count_per_language):
                # Varier un peu la longueur
                target_length = int(avg_length * random.uniform(0.8, 1.2))
                
                # Générer le texte
                synthetic_text = self.generate_synthetic_text(lang, length=target_length)
                
                if synthetic_text:
                    # Créer un article synthétique
                    synthetic_article = {
                        'language': lang,
                        'title': f"Synthetic_{lang}_{i+1}",
                        'text': synthetic_text,
                        'token_count': len(synthetic_text.split()),
                        'category': 'Synthetic',
                        'source': 'data_augmentation'
                    }
                    
                    all_synthetic_articles.append(synthetic_article)
            
            print(f"    {min(count_per_language, len(all_synthetic_articles))} articles générés")
        
        # Créer un DataFrame avec tous les articles synthétiques
        synthetic_df = pd.DataFrame(all_synthetic_articles)
        
        # Sauvegarder les données synthétiques
        synthetic_df.to_csv(f"{self.output_dir}/synthetic_articles.csv", index=False)
        
        print(f"Dataset synthétique créé avec {len(synthetic_df)} articles")
        return synthetic_df
    

    def generate_cross_language_articles(self, pairs=None, count_per_pair=5):
        """Génère des articles synthétiques en mélangeant deux langues proches"""
        # Définir des paires de langues proches si non fournies
        if pairs is None:
            pairs = [
                # Langues slaves de l'Est
                ('ru', 'uk'),      # russe - ukrainien
                ('ru', 'be'),      # russe - bélarussien
                ('uk', 'be'),      # ukrainien - bélarussien
                ('uk', 'rue'),     # ukrainien - rusyn
                ('be', 'rue'),     # bélarussien - rusyn
                # Langues slaves du Sud
                ('bg', 'mk'),      # bulgare - macédonien
                ('sr', 'bg'),      # serbe - bulgare
                ('mk', 'sr'),      # macédonien - serbe
                
                # Langues turciques
                ('kk', 'ky'),      # kazakh - kirghize
                ('tt', 'ba'),      # tatar - bachkir
                
                # Langues finno-ougriennes (priorité pour les groupes C et D)
                ('koi', 'kv'),     # komi-permyak - komi (prioritaire)
                ('udm', 'koi'),    # oudmourte - komi-permyak (prioritaire)
                ('udm', 'kv'),     # oudmourte - komi (prioritaire)
                ('myv', 'mhr'),    # erzya - mari (prioritaire)
                
                # Mélanges expérimentaux pour les langues du groupe C
                ('ab', 'kbd'),     # abkhaze - kabarde (minoritaires)
                ('ab', 'ce'),      # abkhaze - tchétchène (minoritaire avec majoritaire)
                ('kbd', 'ce'),     # kabardien - tchétchène (minoritaire avec majoritaire)
            ]
        
        print(f"Génération d'articles synthétiques par mélange de langues...")
        mixed_articles = []
        
        # Définir le nombre d'articles à générer par paire en fonction des groupes
        priority_pairs = [('koi', 'kv'), ('udm', 'koi'), ('udm', 'kv'), ('myv', 'mhr'),
                          ('ab', 'kbd'), ('ab', 'ce'), ('kbd', 'ce')]
        
        for lang1, lang2 in pairs:
            # Vérifier si les deux langues sont disponibles
            if lang1 not in self.articles_by_language or lang2 not in self.articles_by_language:
                print(f"  Paire {lang1}-{lang2} ignorée: une des langues manque")
                continue
            
            # Déterminer combien d'articles générer pour cette paire
            if (lang1, lang2) in priority_pairs or (lang2, lang1) in priority_pairs:
                pair_count = count_per_pair * 2  # Double d'articles pour les paires prioritaires
            else:
                pair_count = count_per_pair
                
            print(f"  Mélange des langues {lang1} et {lang2} (objectif: {pair_count} articles)...")
            
            # Récupérer des articles pour les deux langues
            articles1 = self.articles_by_language[lang1]['text'].tolist()
            articles2 = self.articles_by_language[lang2]['text'].tolist()
            
            if not articles1 or not articles2:
                continue
                
            # Générer des articles mélangés
            articles_created = 0
            max_attempts = pair_count * 2  # Permettre plus de tentatives pour atteindre l'objectif
            
            for i in range(max_attempts):
                if articles_created >= pair_count:
                    break
                    
                # Sélectionner des articles aléatoires
                article1 = random.choice(articles1)
                article2 = random.choice(articles2)
                
                # Découper en phrases (approx.)
                sentences1 = re.split(r'[.!?]+', article1)
                sentences2 = re.split(r'[.!?]+', article2)
                
                # Filtrer les phrases vides
                sentences1 = [ s.strip() for s in sentences1 if s.strip() ]
                sentences2 = [ s.strip() for s in sentences2 if s.strip() ]
                
                if not sentences1 or not sentences2:
                    continue
                
                # Mélanger des phrases des deux langues (avec deux stratégies)
                mixed_sentences = []
                
                # Stratégie 1: Alternance simple (utilisée pour les articles pairs)
                if i % 2 == 0:
                    for j in range(min(10, max(len(sentences1), len(sentences2)))):
                        if j % 2 == 0 and j // 2 < len(sentences1):
                            mixed_sentences.append(sentences1[j // 2])
                        elif j // 2 < len(sentences2):
                            mixed_sentences.append(sentences2[j // 2])
                
                # Stratégie 2: Blocs de phrases (utilisée pour les articles impairs)
                else:
                    # Premier bloc: langue 1
                    start_idx = random.randint(0, max(0, len(sentences1) - 3))
                    end_idx = min(start_idx + 3, len(sentences1))
                    mixed_sentences.extend(sentences1[start_idx:end_idx])
                    
                    # Deuxième bloc: langue 2
                    start_idx = random.randint(0, max(0, len(sentences2) - 3))
                    end_idx = min(start_idx + 3, len(sentences2))
                    mixed_sentences.extend(sentences2[start_idx:end_idx])
                    
                    # Troisième bloc: langue 1
                    if len(sentences1) > end_idx + 3:
                        mixed_sentences.extend(sentences1[end_idx:end_idx+2])
                
                # Créer le texte mélangé
                mixed_text = '. '.join(mixed_sentences)
                
                # S'assurer que le texte est suffisamment long
                if len(mixed_text.split()) < 50:
                    continue
                
                # Ajouter l'article mélangé
                mixed_article = {
                    'language': f"{lang1}_{lang2}_mix",
                    'title': f"Mixed_{lang1}_{lang2}_{articles_created+1}",
                    'text': mixed_text,
                    'token_count': len(mixed_text.split()),
                    'category': 'Mixed_Language',
                    'source': 'cross_language_augmentation',
                    'mixing_strategy': 'alternating' if i % 2 == 0 else 'blocks'
                }
                
                mixed_articles.append(mixed_article)
                articles_created += 1
            
            print(f"    {articles_created} articles mélangés créés")
        
        # Créer un DataFrame avec les articles mélangés
        mixed_df = pd.DataFrame(mixed_articles)
        
        # Sauvegarder les données
        if not mixed_df.empty:
            mixed_df.to_csv(f"{self.output_dir}/mixed_language_articles.csv", index=False)
            print(f"  {len(mixed_df)} articles de langues mélangées créés au total")
        else:
            print("  Aucun article de langues mélangées n'a pu être créé")
        
        return mixed_df
    

    def data_perturbation(self, character_swap_prob=0.01, deletion_prob=0.01):
        """
        Crée des variations des articles existants par perturbation légère du texte.
        Cette méthode simule des erreurs typographiques et des variations dialectales.
        
        Args:
            character_swap_prob: Probabilité de remplacer un caractère
            deletion_prob: Probabilité de supprimer un caractère
        """
        print("Génération d'articles par perturbation de texte...")

        all_perturbed_articles = []
        
        # Définir des substitutions de caractères spécifiques aux langues
        char_substitutions = {
            # Russe <-> Ukrainien <-> Biélorusse <-> Rusyn
            'ru': {'и': 'і',
                   'е': 'є',
                   'э': 'є',
                   'ы': 'и',
                   'г': 'ґ'},
            'uk': {'і': 'e',
                   'и': 'ы',
                   'ї': 'й',
                   'є': 'e',
                   'е': 'э',
                   'ь': '\'',
                   'щ': 'шч'},
            'be': {'і': 'и',
                   'ы': 'и',
                   'ў': 'в',
                   'э': 'е',
                   'я': 'е',
                   '\'': 'ъ',
                   'а': 'о',
                   'шч': 'щ',
                   'г': 'ґ',
                   'ё': 'е',
                   'ц': 'т',
                   'дз': 'д'},
            'rue': {'ы': 'и',
                    'и': 'і',
                    'ї': 'й',
                    'е': 'є',
                    'э': 'е',
                    'ё': 'о',
                    'г': 'ґ',
                    'ь': '\'',
                    'ѣ': 'е'},
            
            # Bulgare <-> Macédonien
            'bg': {'ъ': 'о',
                   'я': 'ја',
                   'ьо': 'јо',
                   'ю': 'ју',
                   'щ': 'шт',
                   'ь': 'ј',
                   'ж': 'џ',
                   'не': 'ње',
                   'ле': 'ље'},
            'mk': {'ј': 'й',
                   'ќ': 'щ',
                   'ѓ': 'жд',
                   'ја': 'е',
                   'џ': 'ж',
                   'о': 'ъ',
                   'а': 'ъ',
                   'њ': 'н',
                   'ље': 'ле',
                   'ќ': 'щ'},
            
            # Groupe C
            'ab': {'ҧ': 'п', 'ҵ': 'ц', 'ӷ': 'г'},
            'kbd': {'ӏ': 'і', 'э': 'е', 'щ': 'шч'},
            'koi': {'ӧ': 'о', 'і': 'и'},
            'kv': {'ӧ': 'о', 'і': 'и'},
            
            # Groupe D
            'mhr': {'ӱ': 'у', 'ӧ': 'о', 'ӹ': 'ы'},
            
            # Par défaut pour toutes les langues
            'default': {'е': 'э', 'и': 'й', 'о': 'а'}
        }
        
        # Intensifier la perturbation pour les langues des groupes C et D
        perturbation_intensity = {
            'ab': 3,
            'kbd': 3,
            'koi': 3,
            'kv': 3,
            'mhr': 3,
            'default': 1
        }
        
        for lang, articles_df in self.articles_by_language.items():
            print(f"  Perturbation pour {lang}...")
            
            # Définir des substitutions spécifiques à cette langue
            lang_subs = char_substitutions.get(lang, char_substitutions['default'])
            
            # Intensité de perturbation pour cette langue
            intensity = perturbation_intensity.get(lang, perturbation_intensity['default'])
            
            # Ajuster les probabilités selon l'intensité
            adjusted_swap_prob = character_swap_prob * intensity
            adjusted_deletion_prob = deletion_prob * intensity
            
            # Nombre d'articles à perturber (plus d'articles pour les langues prioritaires)
            if lang in ['ab', 'kbd', 'koi', 'kv', 'mhr']:
                sample_size = min(20, len(articles_df))  # Plus d'articles pour les groupes C et D
            else:
                sample_size = min(10, len(articles_df))
            
            # Pour chaque article, créer une version perturbée
            for _, article in articles_df.sample(sample_size).iterrows():
                text = article['text']
                
                if not isinstance(text, str) or not text:
                    continue
                
                # Créer une version perturbée
                perturbed_text = ""
                for char in text:
                    # Possibilité de supprimer un caractère
                    if random.random() < adjusted_deletion_prob:
                        continue
                    
                    # Possibilité de remplacer un caractère
                    if random.random() < adjusted_swap_prob:
                        if char.lower() in lang_subs:
                            char = lang_subs[char.lower()]
                            # Préserver la casse
                            if char.isupper():
                                char = char.upper()
                    
                    perturbed_text += char
                
                # Créer l'article perturbé
                perturbed_article = {
                    'language': lang,
                    'title': f"Perturbed_{article['title']}",
                    'text': perturbed_text,
                    'token_count': len(perturbed_text.split()),
                    'category': 'Perturbed',
                    'source': 'data_perturbation',
                    'original_id': article.get('id', '')
                }
                
                all_perturbed_articles.append(perturbed_article)
            
            print(f"    {sum(1 for a in all_perturbed_articles if a['language'] == lang)} articles perturbés créés pour {lang}")
        
        # Créer un DataFrame avec tous les articles perturbés
        perturbed_df = pd.DataFrame(all_perturbed_articles)
        
        # Sauvegarder les données
        if not perturbed_df.empty:
            perturbed_df.to_csv(f"{self.output_dir}/perturbed_articles.csv", index=False)
            print(f"Dataset d'articles perturbés créé avec {len(perturbed_df)} articles")
        else:
            print("Aucun article perturbé n'a pu être créé")
        
        return perturbed_df
    

    def combine_augmented_datasets(self, synthetic_df, mixed_df, perturbed_df):
        """
        Combine tous les datasets augmentés en un seul.
        
        Args:
            synthetic_df: DataFrame des articles synthétiques
            mixed_df: DataFrame des articles de langues mélangées
            perturbed_df: DataFrame des articles perturbés
        """
        print("Combinaison de tous les datasets augmentés...")
        
        # Liste pour stocker tous les DataFrames
        all_dfs = []
        
        # Ajouter chaque DataFrame non vide
        if synthetic_df is not None and not synthetic_df.empty:
            all_dfs.append(synthetic_df)
        
        if mixed_df is not None and not mixed_df.empty:
            all_dfs.append(mixed_df)
        
        if perturbed_df is not None and not perturbed_df.empty:
            all_dfs.append(perturbed_df)
        
        # Combiner tous les DataFrames
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            combined_df.to_csv(f"{self.output_dir}/all_augmented_articles.csv", index=False)
            print(f"Dataset augmenté complet créé avec {len(combined_df)} articles")
            return combined_df
        else:
            print("Aucun article augmenté n'a été créé")
            return pd.DataFrame()
    
    
    def augment_data(self, synthetic_count=20, mixed_pairs=None, mixed_count=5):
        """
        Exécute le processus complet d'augmentation des données.
        
        Args:
            synthetic_count: Nombre d'articles synthétiques à générer par langue
            mixed_pairs: Paires de langues à mélanger (None pour utiliser les paires par défaut)
            mixed_count: Nombre d'articles mélangés à générer par paire de langues
        """
        # 1. Charger les données
        self.load_data()
        
        # 2. Construire les modèles de langue
        self.build_language_models()
        
        # 3. Générer des articles synthétiques
        synthetic_df = self.generate_synthetic_dataset(count_per_language=synthetic_count)
        
        # 4. Générer des articles de langues mélangées
        mixed_df = self.generate_cross_language_articles(pairs=mixed_pairs, count_per_pair=mixed_count)
        
        # 5. Générer des articles perturbés
        perturbed_df = self.data_perturbation()
        
        # 6. Combiner tous les datasets
        combined_df = self.combine_augmented_datasets(synthetic_df, mixed_df, perturbed_df)
        
        # 7. Afficher un résumé
        print("\n=== Résumé de l'augmentation des données ===")
        print(f"Articles synthétiques: {len(synthetic_df) if synthetic_df is not None else 0}")
        print(f"Articles de langues mélangées: {len(mixed_df) if mixed_df is not None else 0}")
        print(f"Articles perturbés: {len(perturbed_df) if perturbed_df is not None else 0}")
        print(f"Total des articles augmentés: {len(combined_df) if combined_df is not None else 0}")
        
        return combined_df


# Exécution si le script est lancé directement
if __name__ == "__main__":
    augmenter = CyrillicDataAugmenter(input_dir='data/processed/merged', output_dir='data/processed/augmented')
    augmenter.augment_data()