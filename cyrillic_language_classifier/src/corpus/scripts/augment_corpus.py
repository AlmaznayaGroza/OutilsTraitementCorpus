"""Module d'augmentation de données pour corpus multilingues cyrilliques

Ce module fournit des outils pour l'augmentation de données textuelles
dans les langues utilisant l'alphabet cyrillique. Il permet de générer du
contenu synthétique de qualité pour enrichir des corpus de langues
sous-représentées et améliorer l'équilibrage des datasets multilingues.

Le module contient la classe CyrillicDataAugmenter, qui implémente
3 stratégies d'augmentation linguistiquement motivées :
    * génération de texte synthétique basée sur des modèles n-grammes adaptatifs
    * création d'articles multilingues par mélange de langues linguistiquement proches
    * perturbation de texte par substitution et suppression de caractères motivées

Architecture adaptive:
    Le système d'augmentation adapte automatiquement ses paramètres selon le groupe
    linguistique de chaque langue (familles slaves, turciques, finno-ougriennes, etc.)
    pour optimiser la qualité des données générées tout en respectant les
    spécificités orthographiques et morphologiques de chaque langue.

Innovation méthodologique:
    Les stratégies implémentées s'appuient sur des principes de linguistique
    computationnelle pour générer du contenu qui préserve les caractéristiques
    stylistiques et structurelles des langues sources, permettant un enrichissement
    de corpus qui maintient la validité linguistique des données augmentées.

Applications :
    Ce module est particulièrement adapté aux projets multilingues, où
    l'équilibrage des corpus est crucial pour l'entraînement de modèles équitables
    et performants sur des langues à ressources limitées.
"""

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
    nltk.download("punkt", quiet=True)
except Exception as e:
    print("Impossible de télécharger les ressources NLTK: {e}. Continuer sans...")


# ===========================================================
# CONSTANTES DE CONFIGURATION POUR L'AUGMENTATION DE CORPUS
# ===========================================================

# Langues des groupes C et D (langues très minoritaires)
# Nécessitent un traitement spécial avec génération de n-grammes
MINORITY_LANGUAGES = ["ab", "kbd", "koi", "kv", "mhr"]

# Paires de langues linguistiquement proches pour génération multilingue
# Organisées par famille linguistique pour maximiser la cohérence
LANGUAGE_PAIRS = [
    # Langues slaves de l'Est
    ("ru", "uk"),  # russe - ukrainien
    ("ru", "be"),  # russe - bélarussien
    ("uk", "be"),  # ukrainien - bélarussien
    ("uk", "rue"),  # ukrainien - rusyn
    ("be", "rue"),  # bélarussien - rusyn
    # Langues slaves du Sud
    ("bg", "mk"),  # bulgare - macédonien
    ("sr", "bg"),  # serbe - bulgare
    ("mk", "sr"),  # macédonien - serbe
    # Langues turciques
    ("kk", "ky"),  # kazakh - kirghize
    ("tt", "ba"),  # tatar - bachkir
    # Langues finno-ougriennes (priorité pour les groupes C et D)
    ("koi", "kv"),  # komi-permyak - komi (prioritaire)
    ("udm", "koi"),  # oudmourte - komi-permyak (prioritaire)
    ("udm", "kv"),  # oudmourte - komi (prioritaire)
    ("myv", "mhr"),  # erzya - mari (prioritaire)
    # Langues caucasiennes
    ("ab", "kbd"),  # abkhaze - kabarde (minoritaires)
    ("ab", "ce"),  # abkhaze - tchétchène (minoritaire avec majoritaire)
    ("kbd", "ce"),  # kabardien - tchétchène (minoritaire avec majoritaire)
]

# Paires prioritaires (groupes C et D) qui génèrent plus d'articles
PRIORITY_PAIRS = [
    ("koi", "kv"),
    ("udm", "koi"),
    ("udm", "kv"),
    ("myv", "mhr"),
    ("ab", "kbd"),
    ("ab", "ce"),
    ("kbd", "ce"),
]

# Substitutions de caractères pour la perturbation de texte
# Basées sur les variations orthographiques entre langues proches
CHAR_SUBSTITUTIONS = {
    # Langues slaves de l'Est
    "ru": {"и": "і", "е": "є", "э": "є", "ы": "и", "г": "ґ"},
    "uk": {"і": "e", "и": "ы", "ї": "й", "є": "e", "е": "э", "ь": "'", "щ": "шч"},
    "be": {
        "і": "и",
        "ы": "и",
        "ў": "в",
        "э": "е",
        "я": "е",
        "'": "ъ",
        "а": "о",
        "шч": "щ",
        "г": "ґ",
        "ё": "е",
        "ц": "т",
        "дз": "д",
    },
    "rue": {
        "ы": "и",
        "и": "і",
        "ї": "й",
        "е": "є",
        "э": "е",
        "ё": "о",
        "г": "ґ",
        "ь": "'",
        "ѣ": "е",
    },
    # Langues slaves du Sud
    "bg": {
        "ъ": "о",
        "я": "ја",
        "ьо": "јо",
        "ю": "ју",
        "щ": "шт",
        "ь": "ј",
        "ж": "џ",
        "не": "ње",
        "ле": "ље",
    },
    "mk": {
        "ј": "й",
        "ќ": "щ",
        "ѓ": "жд",
        "ја": "е",
        "џ": "ж",
        "о": "ъ",
        "а": "ъ",
        "њ": "н",
        "ље": "ле",
    },
    # Langues du groupe C - caractères spéciaux
    "ab": {"ҧ": "п", "ҵ": "ц", "ӷ": "г"},
    "kbd": {"ӏ": "і", "э": "е", "щ": "шч"},
    "koi": {"ӧ": "о", "і": "и"},
    "kv": {"ӧ": "о", "і": "и"},
    # Langue du groupe D
    "mhr": {"ӱ": "у", "ӧ": "о", "ӹ": "ы"},
    # Substitutions par défaut pour toutes les autres langues
    "default": {"е": "э", "и": "й", "о": "а"},
}

# Multiplicateur d'intensité de perturbation par langue
# Les langues minoritaires ont une intensité plus élevée
# pour compenser leur sous-représentation
PERTURBATION_INTENSITY = {
    "ab": 3,
    "kbd": 3,
    "koi": 3,
    "kv": 3,
    "mhr": 3,  # groupes C et D
    "default": 1,  # toutes les autres langues
}

# Paramètres de configuration
SEGMENT_LENGTH_DEFAULT = 50
MAX_SEGMENTS_PER_ARTICLE = 8
MIN_MIXED_TEXT_LENGTH = 50
NGRAM_TEXT_LENGTH = 100


# ======================
# FONCTIONS UTILITAIRES
# ======================


def create_text_segments(article, segment_length=SEGMENT_LENGTH_DEFAULT):
    """Découpe un article en segments de longueur donnée

    Cette fonction prend un article et le divise en morceaux plus petits
    pour faciliter la recombinaison lors de la génération synthétique.

    Args:
        article (str): texte de l'article à découper
        segment_length (int): longueur des segments en mots

    Returns:
        list: liste des segments de texte
    """
    if not isinstance(article, str) or not article.strip():
        return []

    words = article.split()
    segments = []
    actual_segment_length = min(segment_length, len(words) // 2)

    for i in range(0, len(words), actual_segment_length):
        if i + actual_segment_length <= len(words):
            segments.append(" ".join(words[i: i + actual_segment_length]))

    return segments


def split_text_into_sentences(text):
    """Divise un texte en phrases pour le mélange multilingue

    Utilise une approche simple basée sur la ponctuation pour identifier
    les limites de phrases dans les textes cyrilliques.

    Args:
        text (str): texte à diviser

    Returns:
        list: liste des phrases non vides
    """
    if not isinstance(text, str) or not text.strip():
        return []

    # Découper en phrases (approximation basée sur la ponctuation)
    sentences = re.split(r"[.!?]+", text)

    # Filtrer les phrases vides et nettoyer
    return [s.strip() for s in sentences if s.strip()]


def get_language_target_ratio(lang):
    """Détermine le ratio de longueur cible selon la langue

    Les langues avec plus de données peuvent avoir des articles synthétiques
    plus courts, tandis que les langues rares nécessitent
    des articles plus longs.

    Args:
        lang (str): code de la langue

    Returns:
        float: ratio de longueur (entre 0 et 1)
    """
    # Groupe A: langues avec beaucoup de données
    if lang in ["sr", "ba", "ru", "mk"]:
        return 0.85  # 85% de la longueur originale
    # Groupe B: langues intermédiaires
    elif lang in ["bxr", "tyv", "rue", "ab", "be"]:
        return 0.90  # 90% de la longueur originale
    else:
        return 0.95  # 95% de la longueur originale (groupes C et D)


class CyrillicDataAugmenter:
    """Cette classe implémente plusieurs stratégies d'augmentation de données
    pour enrichir des corpus de langues cyrilliques sous-représentées.

    Attributes:
        input_dir (str): dossier contenant les fichiers CSV d'entrée
        output_dir (str): dossier de sortie pour les données augmentées
        char_ngram_models (dict): modèles de n-grammes de caractères par langue
        word_ngram_models (dict): modèles de n-grammes de mots par langue
        articles_by_language (dict): articles originaux organisés par langue

    Methods:
        load_data(): charge les données depuis les fichiers CSV
        build_language_models(): construit des modèles statistiques par langue
        generate_synthetic_dataset(): génère des articles synthétiques
        generate_cross_language_articles(): crée des articles multilingues
        data_perturbation(): applique des perturbations aux textes existants
        augment_data(): exécute le processus complet d'augmentation

    Examples:
        Utilisation basique :
            >>> augmenter = CyrillicDataAugmenter()
            >>> corpus_augmente = augmenter.augment_data()

        Contrôle fin des paramètres :
            >>> augmenter = CyrillicDataAugmenter(
            ...     input_dir="mon_corpus",
            ...     output_dir="corpus_augmente"
            ... )
            >>> synthetics = augmenter.generate_synthetic_dataset(
            ...     count_per_language=25
            ...     )
    """

    def __init__(
        self, input_dir="data/processed/merged",
        output_dir="data/processed/augmented"
    ):
        """Initialise l'augmenteur de données

        Args:
            input_dir (str, optional): dossier contenant les fichiers CSV
                d'articles initiaux (par défaut: "data/processed/merged")
            output_dir (str, optional): dossier de sortie pour sauvegarder
                les données augmentées (par défaut: "data/processed/augmented")

        Raises:
            OSError: si impossible de créer le répertoire de sortie
        """
        if not os.path.exists(input_dir):
            raise FileNotFoundError(
                f"Le répertoire d'entrée '{input_dir}' n'existe pas"
            )

        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Dictionnaires pour stocker les modèles de langue par langue
        self.char_ngram_models = {}  # modèles de n-grammes de caractères
        self.word_ngram_models = {}  # modèles de n-grammes de mots
        self.articles_by_language = {}  # articles originaux par langue

    def load_data(self):
        """Charge les données depuis les fichiers CSV

        Lit tous les fichiers CSV du dossier d'entrée et les organise
        par langue. Chaque fichier doit être nommé selon le format:
        "{code_langue}_articles.csv".

        Raises:
            FileNotFoundError: si le répertoire d'entrée n'existe pas
            ValueError: si aucun fichier CSV valide n'est trouvé

        Note:
            les fichiers sans colonne 'text' sont ignorés avec un avertissement
        """
        print("Chargement des données originales...")

        all_files = glob.glob(f"{self.input_dir}/*_articles.csv")

        if not all_files:
            raise FileNotFoundError(f"Aucun fichier CSV trouvé dans {self.input_dir}")

        for file in all_files:
            # Extraction du code de langue depuis le nom de fichier
            lang_code = os.path.basename(file).split("_")[0]

            try:
                df = pd.read_csv(file)

                # Vérifier que les colonnes nécessaires existent
                if "text" not in df.columns:
                    print(f"Colonne 'text' manquante dans {file}, fichier ignoré")
                    continue

                # Filtrer les lignes avec du texte non vide
                df = df[df["text"].notna() & (df["text"] != "")]

                if df.empty:
                    print(f"Aucun texte valide dans {file}, fichier ignoré")
                    continue

                # Stocker les articles
                self.articles_by_language[lang_code] = df
                print(f"  Chargé {len(df)} articles pour {lang_code}")

            except Exception as e:
                print(f"Erreur lors du chargement de {file}: {e}")
                continue

        if not self.articles_by_language:
            raise ValueError("Aucun fichier CSV valide n'a pu être chargé")

        print(f"Données chargées pour {len(self.articles_by_language)} langues")

    def build_language_models(self):
        """Construit des modèles statistiques pour chaque langue

        Crée des modèles de n-grammes de caractères et de mots pour chaque
        langue chargée. Ces modèles servent de base pour la génération
        de texte synthétique.

        Les modèles incluent :
            - n-grammes de caractères (bi, tri, quadri-grammes)
            - n-grammes de mots (uni, bi, tri-grammes)

        Raises:
            ValueError: si aucune donnée n'a été chargée avant l'appel

        Note:
            En cas d'erreur sur une langue spécifique, un modèle vide
            est créé et le traitement continue pour les autres langues
        """
        print("Construction des modèles de langue...")

        for lang, articles_df in self.articles_by_language.items():
            print(f"  Modélisation de la langue {lang}...")

            try:
                # Concaténer tous les textes
                all_text = " ".join(articles_df["text"].fillna(""))

                if not all_text.strip():
                    print(f"    Aucun texte disponible pour {lang}")
                    continue

                # Modèle de n-grammes de caractères
                # (pour capturer les schémas orthographiques)
                char_ngram_dict = {}
                for n in range(2, 5):  # bigrammes, trigrammes et quadrigrammes
                    char_ngram_dict[n] = Counter()
                    for i in range(len(all_text) - n + 1):
                        ngram = all_text[i: i + n]
                        char_ngram_dict[n][ngram] += 1

                self.char_ngram_models[lang] = char_ngram_dict

                # Modèle de n-grammes de mots (pour la structure syntaxique)
                try:
                    # Tokeniser par mots (approximation)
                    words = re.findall(r"\b\w+\b", all_text.lower())

                    word_ngram_dict = {}
                    for n in range(1, 4):  # unigrammes, bigrammes et trigrammes
                        word_ngram_dict[n] = Counter(ngrams(words, n))

                    self.word_ngram_models[lang] = word_ngram_dict
                    print(f"    Modèle construit avec {len(words)} mots")

                except Exception as e:
                    print(
                        f"    Erreur lors de la construction du modèle de mots pour {lang}: {e}"
                    )
                    # Créer un modèle vide en cas d'erreur
                    self.word_ngram_models[lang] = {
                        n: Counter() for n in range(1, 4)
                    }

            except Exception as e:
                print(f"    Erreur lors de la modélisation pour {lang}: {e}")
                # Créer des modèles vides en cas d'erreur
                self.char_ngram_models[lang] = {
                    n: Counter() for n in range(2, 5)
                }
                self.word_ngram_models[lang] = {
                    n: Counter() for n in range(1, 4)
                }

        print("Tous les modèles de langue ont été construits!")

    def generate_synthetic_text(self, lang, length=250):
        """Génère un texte synthétique basé sur le modèle de langue

        Crée un nouveau texte en combinant des segments d'articles existants
        et en appliquant optionnellement des modèles n-grammes pour les
        langues sous-représentées.

        Args:
            lang (str): code de la langue pour laquelle générer le texte
            length (int, optional): longueur cible en mots (par défaut: 250)

        Returns:
            str: texte synthétique généré, ou chaîne vide en cas d'échec

        Note:
            Pour les langues des groupes C et D (langues très minoritaires),
            utilise également un modèle de trigrammes de caractères pour
            augmenter la diversité du texte généré
        """
        # Vérifier que les modèles et articles existent
        if (
            lang not in self.char_ngram_models
            or lang not in self.word_ngram_models
            or lang not in self.articles_by_language
        ):
            print(f"Pas de modèle disponible pour la langue {lang}")
            return ""

        # Approche 1: utiliser des segments de textes existants
        original_articles = self.articles_by_language[lang]["text"].tolist()

        if not original_articles:
            return ""

        # Sélectionner quelques articles aléatoires
        selected_articles = random.sample(
            original_articles, min(3, len(original_articles))
        )

        # Découper les articles en segments plus courts
        segments = []
        for article in selected_articles:
            article_segments = create_text_segments(article)
            segments.extend(article_segments)

        # Si pas assez de segments, utiliser l'article complet
        if len(segments) < 3:
            segments = selected_articles

        # Approche 2: pour les langues minoritaires,
        # utiliser aussi le modèle n-grammes
        if lang in MINORITY_LANGUAGES:
            try:
                # Utiliser le modèle de trigrammes
                # pour générer du texte à partir de zéro
                char_model = {}
                all_text = " ".join(original_articles)

                # Construire un modèle de trigrammes de caractères
                for i in range(len(all_text) - 3):
                    trigram = all_text[i:i+3]
                    next_char = all_text[i + 3]
                    if trigram not in char_model:
                        char_model[trigram] = []
                    char_model[trigram].append(next_char)

                # Générer du texte avec ce modèle
                if char_model:
                    try:
                        # Sélectionner un trigramme aléatoire pour commencer
                        current = random.choice(list(char_model.keys()))
                        generated_text = current

                        # Générer une centaine de caractères
                        for _ in range(NGRAM_TEXT_LENGTH):
                            if current in char_model and char_model[current]:
                                next_char = random.choice(char_model[current])
                                generated_text += next_char
                                current = current[1:] + next_char
                            else:
                                # Si le trigramme n'existe pas, prendre un autre au hasard
                                if char_model:
                                    current = random.choice(
                                        list(char_model.keys())
                                    )

                        # Ajouter ce texte généré comme segment supplémentaire
                        segments.append(generated_text)
                    except (IndexError, KeyError) as e:
                        # En cas d'erreur, ignorer la génération par n-grammes
                        print(f"    Erreur n-gramme pour {lang}: {e}")
                        pass
            except Exception as e:
                print(f"    Erreur lors de la génération n-gramme pour {lang}: {e}")

        # Mélanger et combiner des segments
        random.shuffle(segments)

        # Prendre suffisamment de segments pour atteindre la longueur cible
        synthetic_text = " ".join(
            segments[: min(MAX_SEGMENTS_PER_ARTICLE, len(segments))]
        )

        # Ajuster pour obtenir une longueur approximative
        words = synthetic_text.split()
        if len(words) > length:
            synthetic_text = " ".join(words[:length])

        # Répéter pour atteindre au moins 80% de la longueur cible
        attempts = 0
        max_attempts = 3  # éviter les boucles infinies
        while (
            len(synthetic_text.split()) < 0.8 * length
            and len(segments) > 5
            and attempts < max_attempts
        ):
            random.shuffle(segments)
            additional_text = " ".join(
                segments[:3]
            )  # prendre 3 segments supplémentaires
            synthetic_text += " " + additional_text

            # Revérifier qu'on ne dépasse pas la longueur cible
            words = synthetic_text.split()
            if len(words) > length:
                synthetic_text = " ".join(words[:length])
                break

            attempts += 1

        return synthetic_text

    def generate_synthetic_dataset(self, count_per_language=20):
        """Génère un ensemble de données synthétiques pour toutes les langues

        Crée des articles synthétiques pour chaque langue chargée en utilisant
        des modèles de langue et des techniques de recombinaison de segments.

        Args:
            count_per_language (int, optional): nb d'articles à générer
                par langue (par défaut: 20)

        Returns:
            pandas.DataFrame: DataFrame contenant
                tous les articles synthétiques avec les colonnes
                'language', 'title', 'text', 'token_count',
                'category', 'source'

        Note:
            La longueur cible des articles est ajustée selon la langue:
                - langues du groupe A: 85% de la longueur moyenne originale
                - langues du groupe B: 90%
                - langues des groupes C et D: 95%
        """
        print(f"Génération de {count_per_language} articles synthétiques par langue...")

        all_synthetic_articles = []

        for lang in self.articles_by_language.keys():
            print(f"  Génération pour {lang}...")

            try:
                # Calculer la longueur moyenne des articles originaux
                original_lengths = self.articles_by_language[lang][
                    "token_count"
                ].tolist()
                avg_length = (
                    int(np.mean(original_lengths)) if original_lengths else 200
                )

                # Utiliser notre fonction utilitaire pour le ratio de longueur
                target_ratio = get_language_target_ratio(lang)

                # Générer des articles synthétiques
                articles_created = 0
                for i in range(count_per_language):
                    # Varier un peu la longueur
                    target_length = int(
                        avg_length * target_ratio * random.uniform(0.8, 1.2)
                    )

                    # Générer le texte
                    synthetic_text = self.generate_synthetic_text(
                        lang, length=target_length
                    )

                    if synthetic_text:
                        # Créer un article synthétique
                        synthetic_article = {
                            "language": lang,
                            "title": f"Synthetic_{lang}_{i+1}",
                            "text": synthetic_text,
                            "token_count": len(synthetic_text.split()),
                            "category": "Synthetic",
                            "source": "data_augmentation",
                        }

                        all_synthetic_articles.append(synthetic_article)
                        articles_created += 1

                print(f"    {articles_created} articles générés")

            except Exception as e:
                print(f"Erreur lors de la génération pour {lang}: {e}")
                continue

        # Créer un DataFrame avec tous les articles synthétiques
        synthetic_df = pd.DataFrame(all_synthetic_articles)

        # Sauvegarder les données synthétiques
        try:
            output_file = f"{self.output_dir}/synthetic_articles.csv"
            synthetic_df.to_csv(output_file, index=False)
            print(f"Dataset synthétique créé avec {len(synthetic_df)} articles")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde: {e}")

        return synthetic_df

    def generate_cross_language_articles(self, pairs=None, count_per_pair=5):
        """Génère des articles synthétiques en mélangeant 2 langues proches

        Crée des articles multilingues en combinant des phrases de langues
        linguistiquement proches pour simuler des phénomènes de code-switching
        ou de multilinguisme.

        Args:
            pairs (list of tuples, optional): paires de langues à mélanger -
                Si None, utilise des paires prédéfinies basées sur la proximité
                linguistique (par défaut: None)
            count_per_pair (int, optional): nb d'articles à générer par
                paire de langues - les paires prioritaires (groupes C et D)
                génèrent le double (par défaut: 5)

        Returns:
            pandas.DataFrame: DataFrame contenant les articles multilingues
                avec colonne supplémentaire 'mixing_strategy'

        Note:
            2 stratégies de mélange sont utilisées:
                - Alternance: phrases alternées entre les 2 langues
                - Blocs: blocs consécutifs de phrases par langue
        """
        # Si aucune paire n'est fournie
        if pairs is None:
            pairs = LANGUAGE_PAIRS

        print("Génération d'articles synthétiques par mélange de langues...")
        mixed_articles = []

        for lang1, lang2 in pairs:
            # Vérifier si les 2 langues sont disponibles
            if (
                lang1 not in self.articles_by_language
                or lang2 not in self.articles_by_language
            ):
                print(f"  Paire {lang1}-{lang2} ignorée: une des langues manque")
                continue

            # Déterminer combien d'articles générer pour cette paire
            if (
                (lang1, lang2) in PRIORITY_PAIRS or
                (lang2, lang1) in PRIORITY_PAIRS
            ):
                pair_count = (
                    count_per_pair * 2
                )  # double d'articles pour les paires prioritaires
            else:
                pair_count = count_per_pair

            print(
                f"  Mélange des langues {lang1} et {lang2} (objectif: {pair_count} articles)..."
            )

            try:
                # Récupérer des articles pour les 2 langues
                articles1 = self.articles_by_language[lang1]["text"].tolist()
                articles2 = self.articles_by_language[lang2]["text"].tolist()

                if not articles1 or not articles2:
                    continue

                # Générer des articles mélangés
                articles_created = 0
                max_attempts = pair_count * 2  # permettre plus de tentatives

                for i in range(max_attempts):
                    if articles_created >= pair_count:
                        break

                    try:
                        # Sélectionner des articles aléatoires
                        article1 = random.choice(articles1)
                        article2 = random.choice(articles2)

                        # Découper en phrases
                        sentences1 = split_text_into_sentences(article1)
                        sentences2 = split_text_into_sentences(article2)

                        if not sentences1 or not sentences2:
                            continue

                        # Mélanger des phrases des 2 langues (2 stratégies)
                        mixed_sentences = []

                        # Stratégie 1: alternance simple (articles pairs)
                        if i % 2 == 0:
                            for j in range(
                                min(10, max(len(sentences1), len(sentences2)))
                            ):
                                if j % 2 == 0 and j // 2 < len(sentences1):
                                    mixed_sentences.append(sentences1[j // 2])
                                elif j // 2 < len(sentences2):
                                    mixed_sentences.append(sentences2[j // 2])

                        # Stratégie 2: blocs de phrases (articles impairs)
                        else:
                            # 1er bloc: langue 1
                            start_idx = random.randint(
                                0, max(0, len(sentences1) - 3)
                            )
                            end_idx = min(start_idx + 3, len(sentences1))
                            mixed_sentences.extend(
                                sentences1[start_idx:end_idx]
                            )

                            # 2ème bloc: langue 2
                            start_idx = random.randint(
                                0, max(0, len(sentences2) - 3)
                            )
                            end_idx = min(start_idx + 3, len(sentences2))
                            mixed_sentences.extend(
                                sentences2[start_idx:end_idx]
                            )

                            # 3ème bloc: langue 1
                            if len(sentences1) > end_idx + 3:
                                mixed_sentences.extend(
                                    sentences1[end_idx:end_idx+2]
                                )

                        # Créer le texte mélangé
                        mixed_text = ". ".join(mixed_sentences)

                        # S'assurer que le texte est suffisamment long
                        if len(mixed_text.split()) < MIN_MIXED_TEXT_LENGTH:
                            continue

                        # Ajouter l'article mélangé
                        mixed_article = {
                            "language": f"{lang1}_{lang2}_mix",
                            "title": f"Mixed_{lang1}_{lang2}_{articles_created+1}",
                            "text": mixed_text,
                            "token_count": len(mixed_text.split()),
                            "category": "Mixed_Language",
                            "source": "cross_language_augmentation",
                            "mixing_strategy": (
                                "alternating" if i % 2 == 0 else "blocks"
                            ),
                        }

                        mixed_articles.append(mixed_article)
                        articles_created += 1

                    except Exception as e:
                        print(
                            f"    Erreur lors de la création d'un article mélangé: {e}"
                        )
                        continue

                print(f"    {articles_created} articles mélangés créés")

            except Exception as e:
                print(f"Erreur lors du traitement de la paire {lang1}-{lang2}: {e}")
                continue

        # Créer un DataFrame avec les articles mélangés
        mixed_df = pd.DataFrame(mixed_articles)

        # Sauvegarder les données
        if not mixed_df.empty:
            try:
                output_file = f"{self.output_dir}/mixed_language_articles.csv"
                mixed_df.to_csv(output_file, index=False)
                print(f"  {len(mixed_df)} articles de langues mélangées créés au total")
            except Exception as e:
                print(f"Erreur lors de la sauvegarde: {e}")
        else:
            print("  Aucun article de langues mélangées n'a pu être créé")

        return mixed_df

    def data_perturbation(self, character_swap_prob=0.01, deletion_prob=0.01):
        """
        Crée des variations des articles existants par perturbation légère du texte.

        Simule des erreurs typographiques et des variations dialectales en
        appliquant des substitutions et suppressions de caractères contrôlées.

        Args:
            character_swap_prob (float, optional): probabilité de remplacer
                un caractère par un équivalent linguistique (par défaut: 0.01)
            deletion_prob (float, optional): probabilité de supprimer un
                caractère (par défaut: 0.01)

        Returns:
            pandas.DataFrame: DataFrame contenant les articles perturbés
                avec colonne supplémentaire 'original_id'

        Note:
            L'intensité de perturbation est triplée pour les langues des
            groupes C et D (ab, kbd, koi, kv, mhr) pour compenser leur
            sous-représentation dans le corpus original.

            Les substitutions de caractères sont linguistiquement motivées
            (ex: variations orthographiques entre langues proches).
        """
        print("Génération d'articles par perturbation de texte...")

        all_perturbed_articles = []

        for lang, articles_df in self.articles_by_language.items():
            print(f"  Perturbation pour {lang}...")

            try:
                # Définir des substitutions spécifiques à cette langue
                lang_subs = CHAR_SUBSTITUTIONS.get(
                    lang, CHAR_SUBSTITUTIONS["default"]
                )

                # Intensité de perturbation pour la langue
                intensity = PERTURBATION_INTENSITY.get(
                    lang, PERTURBATION_INTENSITY["default"]
                )

                # Ajuster les probabilités selon l'intensité
                adjusted_swap_prob = character_swap_prob * intensity
                adjusted_deletion_prob = deletion_prob * intensity

                # Nb d'articles à perturber (plus pour les langues prioritaires)
                if lang in MINORITY_LANGUAGES:
                    sample_size = min(20, len(articles_df))
                else:
                    sample_size = min(10, len(articles_df))

                # Pour chaque article, créer une version perturbée
                for _, article in articles_df.sample(sample_size).iterrows():
                    try:
                        text = article["text"]

                        if not isinstance(text, str) or not text:
                            continue

                        perturbed_text = ""
                        for char in text:
                            # Possibilité de supprimer un caractère
                            if random.random() < adjusted_deletion_prob:
                                continue

                            # Possibilité de remplacer un caractère
                            if random.random() < adjusted_swap_prob:
                                if char.lower() in lang_subs:
                                    replacement = lang_subs[char.lower()]
                                    # Préserver la casse
                                    if char.isupper():
                                        replacement = replacement.upper()
                                    char = replacement

                            perturbed_text += char

                        # S'assurer que le texte perturbé n'est pas vide
                        if not perturbed_text.strip():
                            continue

                        # Créer l'article perturbé
                        perturbed_article = {
                            "language": lang,
                            "title": f"Perturbed_{article.get('title', 'Unknown')}",
                            "text": perturbed_text,
                            "token_count": len(perturbed_text.split()),
                            "category": "Perturbed",
                            "source": "data_perturbation",
                            "original_id": article.get("id", ""),
                        }

                        all_perturbed_articles.append(perturbed_article)

                    except Exception as e:
                        print(f"    Erreur lors de la perturbation d'un article: {e}")
                        continue

                articles_for_lang = [
                    a for a in all_perturbed_articles if a["language"] == lang
                ]
                print(
                    f"    {len(articles_for_lang)} articles perturbés créés pour {lang}"
                )

            except Exception as e:
                print(f"Erreur lors de la perturbation pour {lang}: {e}")
                continue

        # Créer un DataFrame avec tous les articles perturbés
        perturbed_df = pd.DataFrame(all_perturbed_articles)

        # Sauvegarder les données
        if not perturbed_df.empty:
            try:
                output_file = f"{self.output_dir}/perturbed_articles.csv"
                perturbed_df.to_csv(output_file, index=False)
                print(
                    f"Dataset d'articles perturbés créé avec {len(perturbed_df)} articles"
                )
            except Exception as e:
                print(f"Erreur lors de la sauvegarde: {e}")
        else:
            print("Aucun article perturbé n'a pu être créé")

        return perturbed_df

    def combine_augmented_datasets(self, synthetic_df, mixed_df, perturbed_df):
        """
        Combine tous les datasets augmentés en un seul.

        Fusionne les DataFrames des différentes stratégies d'augmentation
        (synthétique, multilingue, perturbé) en un corpus unifié.

        Args:
            synthetic_df (pandas.DataFrame): articles synthétiques
            mixed_df (pandas.DataFrame): articles multilingues
            perturbed_df (pandas.DataFrame): articles perturbés

        Returns:
            pandas.DataFrame: corpus augmenté complet combinant toutes
                les stratégies d'augmentation (ou DataFrame vide si aucune
                donnée d'augmentation n'est disponible)

        Note:
            Le DataFrame résultant est automatiquement sauvegardé sous
            le nom "all_augmented_articles.csv" dans le répertoire de sortie
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
            try:
                combined_df = pd.concat(all_dfs, ignore_index=True)
                output_file = f"{self.output_dir}/all_augmented_articles.csv"
                combined_df.to_csv(output_file, index=False)
                print(f"Dataset augmenté complet créé avec {len(combined_df)} articles")
                return combined_df
            except Exception as e:
                print(f"Erreur lors de la combinaison: {e}")
                return pd.DataFrame()
        else:
            print("Aucun article augmenté n'a été créé")
            return pd.DataFrame()

    def augment_data(
            self,
            synthetic_count=20,
            mixed_pairs=None,
            mixed_count=5
    ):
        """
        Exécute le processus complet d'augmentation des données.

        Lance séquentiellement toutes les étapes d'augmentation:
        chargement des données, construction des modèles,
        génération synthétique, création d'articles multilingues,
        perturbation, et combinaison finale.

        Args:
            synthetic_count (int, optional): nb d'articles synthétiques
                à générer par langue (par défaut: 20)
            mixed_pairs (list of tuples, optional): paires de langues pour
                la génération multilingue - si None, utilise des paires
                prédéfinies (par défaut: None)
            mixed_count (int, optional): nb d'articles multilingues
                à générer par paire de langues (par défaut: 5)

        Returns:
            pandas.DataFrame: corpus augmenté complet combinant toutes
                les stratégies d'augmentation

        Raises:
            FileNotFoundError: si le dossier d'entrée n'existe pas
            ValueError: si aucune donnée valide n'est trouvée

        Exemples:
            Augmentation avec paramètres par défaut :
                >>> augmenter = CyrillicDataAugmenter()
                >>> corpus = augmenter.augment_data()

            Augmentation personnalisée :
                >>> corpus = augmenter.augment_data(
                ...     synthetic_count=30,
                ...     mixed_count=10
                ... )
        """
        try:
            print("=== Début du processus d'augmentation des données ===")

            # 1. Charger les données
            print("\n1. Chargement des données...")
            self.load_data()

            # 2. Construire les modèles de langue
            print("\n2. Construction des modèles de langue...")
            self.build_language_models()

            # 3. Générer des articles synthétiques
            print("\n3. Génération d'articles synthétiques...")
            synthetic_df = self.generate_synthetic_dataset(
                count_per_language=synthetic_count
            )

            # 4. Générer des articles de langues mélangées
            print("\n4. Génération d'articles multilingues...")
            mixed_df = self.generate_cross_language_articles(
                pairs=mixed_pairs, count_per_pair=mixed_count
            )

            # 5. Générer des articles perturbés
            print("\n5. Génération d'articles perturbés...")
            perturbed_df = self.data_perturbation()

            # 6. Combiner tous les datasets
            print("\n6. Combinaison des datasets...")
            combined_df = self.combine_augmented_datasets(
                synthetic_df, mixed_df, perturbed_df
            )

            # 7. Afficher un résumé
            self._print_summary(
                synthetic_df,
                mixed_df,
                perturbed_df,
                combined_df
            )

            print("\n=== Processus d'augmentation terminé avec succès ===")
            return combined_df

        except Exception as e:
            print(f"\nErreur lors du processus d'augmentation: {e}")
            raise

    def _print_summary(self, synthetic_df, mixed_df, perturbed_df, combined_df):
        """Affiche un résumé détaillé de l'augmentation"""
        print("\n" + "=" * 50)
        print("RÉSUMÉ DE L'AUGMENTATION DES DONNÉES")
        print("=" * 50)

        synthetic_count = len(synthetic_df) if synthetic_df is not None else 0
        mixed_count = len(mixed_df) if mixed_df is not None else 0
        perturbed_count = len(perturbed_df) if perturbed_df is not None else 0
        total_count = len(combined_df) if combined_df is not None else 0

        print(f"Articles synthétiques générés     : {synthetic_count:,}")
        print(f"Articles multilingues créés       : {mixed_count:,}")
        print(f"Articles perturbés créés          : {perturbed_count:,}")
        print("-" * 50)
        print(f"TOTAL des articles augmentés      : {total_count:,}")

        if combined_df is not None and not combined_df.empty:
            # Statistiques par langue si disponibles
            if "language" in combined_df.columns:
                lang_counts = combined_df["language"].value_counts()
                print(f"\nNombre de langues augmentées      : {len(lang_counts)}")
                print("\nTop 5 des langues les plus augmentées :")
                for lang, count in lang_counts.head().items():
                    print(f"  {lang}: {count:,} articles")

            # Statistiques par méthode si disponibles
            if "source" in combined_df.columns:
                source_counts = combined_df["source"].value_counts()
                print("\nDistribution par méthode d'augmentation :")
                for source, count in source_counts.items():
                    percentage = (count / total_count) * 100
                    print(f"  {source}: {count:,} articles ({percentage:.1f}%)")


# Point d'entrée principal du script
if __name__ == "__main__":
    """Point d'entrée principal du script avec gestion d'erreurs

    Exécute l'augmentation de données avec les paramètres par défaut
    si le module est lancé directement depuis la ligne de commande.

    Usage:
        python augment_corpus.py
    """
    try:
        print("Lancement de l'augmentation de corpus cyrillique...")

        augmenter = CyrillicDataAugmenter(
            input_dir="data/processed/merged",
            output_dir="data/processed/augmented"
        )

        corpus_augmente = augmenter.augment_data()

        print("\n✅ Augmentation terminée avec succès !")
        print(f"{len(corpus_augmente)} articles augmentés générés")
        print("📁 Résultats sauvegardés dans data/processed/augmented/")

    except KeyboardInterrupt:
        print("\n❌ Processus interrompu par l'utilisateur")
        exit(1)
    except FileNotFoundError as e:
        print(f"\n❌ Fichier/dossier non trouvé: {e}")
        print(
            "💡 Vérifiez que le répertoire d'entrée existe et contient des fichiers CSV"
        )
        exit(1)
    except ValueError as e:
        print(f"\n❌ Erreur de données: {e}")
        print("💡 Vérifiez le format de vos fichiers CSV")
        exit(1)
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        print("💡 Consultez la documentation")
        exit(1)
