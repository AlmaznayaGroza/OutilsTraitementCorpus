"""Module de configuration pour la collecte de corpus Wikipédia.

Ce module contient tous les paramètres de configuration pour la collecte
d'articles Wikipedia multilingues. Il définit les groupes de langues,
les paramètres adaptatifs, les traductions de catégories et les fonctions
utilitaires pour l'optimisation de la collecte de corpus.

Composants principaux:
- Définition des groupes de langues basées sur la disponibilité des ressources
- Paramètres de collecte adaptatifs pour différents groupes de langues
- Traductions de catégories pour la collecte multilingue
- Calculs d'objectifs de tokens et fonctions de récupération de paramètres

La configuration utilise des stratégies adaptatives où les langues bien dotées
(Groupe A) utilisent des paramètres de collecte différents des langues moins
dotées (Groupes B, C, D).
"""

import logging
from typing import Dict, List, Optional


# === CONSTANTES GÉNÉRALES ===

# Temps limite pour éviter les exécutions trop longues (secondes)
TIME_LIMIT = 21600  # 6 heures

# Objectif spécial pour le bélarussien
# (fusion postérieure des 2 variantes)
BELARUSIAN_TARGET = 60000

# Valeurs par défaut pour les cas d'erreur
DEFAULT_LANGUAGE_GROUP = "C"
DEFAULT_MIN_CHAR_LENGTH = 400
DEFAULT_MAX_TOKEN_LENGTH = 1000
DEFAULT_TOKEN_TARGET = 80000


# === GROUPES DE LANGUES ===

# Définition des groupes basés sur la richesse des ressources Wikipedia
# (déterminée après plusieurs tests)
LANGUAGE_GROUPS: Dict[str, List[str]] = {
    "A": [  # langues très bien dotées
        "ba",
        "be",
        "be-tarask",
        "bg",
        "bxr",
        "cv",
        "kk",
        "ky",
        "mk",
        "mn",
        "ru",
        "rue",
        "sah",
        "sr",
        "tg",
        "tt",
        "tyv",
        "uk",
    ],
    "B": ["ce", "os", "udm"],  # langues bien dotées
    "C": ["ab", "koi", "myv"], # langues peu dotées
    "D": ["kbd", "kv", "mhr"], # Langues très peu dotées
}

# Liste complète des langues prises en charge
LANGUAGES = []
for group_languages in LANGUAGE_GROUPS.values():
    LANGUAGES.extend(group_languages)


# === PARAMÈTRES ADAPTATIFS PAR GROUPE ===

# Objectifs de tokens adaptés à chaque groupe
TARGET_TOKENS_BY_GROUP: Dict[str, int] = {
    "A": 100000, # objectif élevé pour les langues bien dotées
    "B": 90000,  # objectif modéré
    "C": 80000,  # objectif réduit
    "D": 75000,  # objectif minimal
}

# Profondeur d'exploration des sous-catégories par groupe
MAX_DEPTHS_BY_GROUP: Dict[str, int] = {
    "A": 3,  # exploration superficielle (ressources abondantes)
    "B": 6,  # exploration intermédiaire
    "C": 9,  # exploration plus profonde nécessaire
    "D": 12, # exploration très profonde
}

# Paramètres de collecte adaptatifs
ADAPTIVE_PARAMS: Dict[str, Dict[str, float]] = {
    "A": {
        "min_char_length": 500,        # seuil de qualité élevé
        "max_token_length": 1000,      # limite standard
        "main_category_ratio": 0.25,   # 25% catégories principales
        "subcategory_ratio": 0.35,     # 35% sous-catégories
        "random_ratio": 0.4,           # 40% articles aléatoires
        "fixed_selection_ratio": 0.34, # 34% sélection ordonnée
    },
    "B": {
        "min_char_length": 450,
        "max_token_length": 1000,
        "main_category_ratio": 0.1,    # moins de catégories principales
        "subcategory_ratio": 0.45,     # plus de sous-catégories
        "random_ratio": 0.45,          # plus d'articles aléatoires
        "fixed_selection_ratio": 0.5,
    },
    "C": {
        "min_char_length": 400,
        "max_token_length": 1250,      # limite plus élevée
        "main_category_ratio": 0.1,
        "subcategory_ratio": 0.4,
        "random_ratio": 0.5,           # majorité d'articles aléatoires
        "fixed_selection_ratio": 0.5,
    },
    "D": {
        "min_char_length": 300,        # seuil plus bas par nécessité
        "max_token_length": 1500,      # limite la plus élevée
        "main_category_ratio": 0.1,
        "subcategory_ratio": 0.4,
        "random_ratio": 0.5,
        "fixed_selection_ratio": 0.67, # plus de sélection ordonnée
    },
}


# === PRÉFIXES ET TRADUCTIONS ===

# Catégories thématiques universelles
ALL_CATEGORIES: List[str] = [
    "Culture",
    "History",
    "Geography",
    "Politics",
    "People",
    "Science",
    "Sports",
]

# Préfixes de catégorie selon la langue Wikipedia
CATEGORY_PREFIXES: Dict[str, str] = {
    "ru": "Категория:",         # russe
    "uk": "Категорія:",         # ukrainien
    "be": "Катэгорыя:",         # bélarussien officiel
    "be-tarask": "Катэгорыя:",  # bélarussien classique
    "bg": "Категория:",         # bulgare
    "sr": "Категорија:",        # serbe
    "mk": "Категорија:",        # macédonien
    "mn": "Ангилал:",           # mongol
    "kk": "Санат:",             # kazakh
    "ky": "Категория:",         # kirghize
    "tg": "Гурӯҳ:",             # tadjik
    "tt": "Төркем:",            # tatar
    "ba": "Категория:",         # bachkir
    "cv": "Категори:",          # tchouvache
    "rue": "Катеґорія:",        # rusyn
    "ce": "Категори:",          # tchétchène
    "os": "Категори:",          # ossète
    "sah": "Категория:",        # iakoute (sakha)
    "mhr": "Категорий:",        # mari des prairies
    "myv": "Категория:",        # erzya
    "koi": "Категория:",        # komi-permyak
    "kv": "Категория:",         # komi
    "udm": "Категория:",        # oudmourte
    "kbd": "Категориэ:",        # kabarde (circassien oriental)
    "ab": "Акатегориа:",        # abkhaze
    "bxr": "Категори:",         # bouriate
    "tyv": "Аңгылал:",          # touvin
}

# Traductions des catégories par langue
CATEGORY_TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "Culture": {
        "ru": "Культура",
        "uk": "Культура",
        "be": "Культура",
        "be-tarask": "Культура",
        "bg": "Култура",
        "sr": "Култура",
        "mk": "Култура",
        "mn": "Соёл",
        "kk": "Мәдениет",
        "ky": "Маданият",
        "tg": "Фарҳанг",
        "tt": "Мәдәният",
        "ba": "Мәҙәниәт",
        "cv": "Культура",
        "rue": "Култура",
        "ce": "Оьздангалла",
        "os": "Культурæ",
        "sah": "Култуура",
        "mhr": "Тӱвыра",
        "myv": "Культура",
        "koi": "Культура",
        "kv": "Культура",
        "udm": "Культура",
        "kbd": "Културэ",
        "ab": "Акультура",
        "bxr": "Соёл",
        "tyv": "Культура",
    },
    "History": {
        "ru": "История",
        "uk": "Історія",
        "be": "Гісторыя",
        "be-tarask": "Гісторыя",
        "bg": "История",
        "sr": "Историја",
        "mk": "Историја",
        "mn": "Түүх",
        "kk": "Тарих",
        "ky": "Тарых",
        "tg": "Таърих",
        "tt": "Тарих",
        "ba": "Тарих",
        "cv": "Истори",
        "rue": "Історія",
        "ce": "Истори",
        "os": "Истори",
        "sah": "История",
        "mhr": "Эртымгорно",
        "myv": "История",
        "koi": "История",
        "kv": "Важвылӧм",
        "udm": "История",
        "kbd": "Тхыдэ",
        "ab": "Аҭоурых",
        "bxr": "Түүхэ",
        "tyv": "Төөгү",
    },
    "Geography": {
        "ru": "География",
        "uk": "Географія",
        "be": "Геаграфія",
        "be-tarask": "Геаграфія",
        "bg": "География",
        "sr": "Географија",
        "mk": "Географија",
        "mn": "Газар_зүй",
        "kk": "География",
        "ky": "География",
        "tg": "Ҷуғрофиё",
        "tt": "География",
        "ba": "География",
        "cv": "Географи",
        "rue": "Ґеоґрафія",
        "ce": "Географи",
        "os": "Географи",
        "sah": "География",
        "mhr": "Географий",
        "myv": "География",
        "koi": "География",
        "kv": "География",
        "udm": "География",
        "kbd": "Хэкумэтх",
        "ab": "Агеографиа",
        "bxr": "Газар_зүй",
        "tyv": "География",
    },
    "Politics": {
        "ru": "Политика",
        "uk": "Політика",
        "be": "Палітыка",
        "be-tarask": "Палітыка",
        "bg": "Политика",
        "sr": "Политика",
        "mk": "Политика",
        "mn": "Улс_төр",
        "kk": "Саясат",
        "ky": "Саясат",
        "tg": "Сиёсат",
        "tt": "Сәясәт",
        "ba": "Сәйәсәт",
        "cv": "Политика",
        "rue": "Політіка",
        "ce": "Политика",
        "os": "Политикæ",
        "sah": "Политика",
        "mhr": "Политике",
        "myv": "Политикась",
        "koi": "Политика",
        "kv": "Политика",
        "udm": "Политика",
        "kbd": "Политикэ",
        "ab": "Аполитика",
        "bxr": "Улас_түрэ",
    },
    "People": {
        "ru": "Люди",
        "uk": "Персоналії",
        "be": "Асобы",
        "be-tarask": "Асобы",
        "bg": "Хора",
        "sr": "Људи",
        "mk": "Луѓе",
        "mn": "Хүн",
        "kk": "Тұлғалар",
        "ky": "Адамдар",
        "tg": "Одамон",
        "tt": "Шәхесләр",
        "ba": "Кешеләр",
        "cv": "Çынсем",
        "rue": "Люде",
        "ce": "Нах",
        "os": "Зындгонд_адæм",
        "sah": "Дьон",
        "mhr": "Еҥ-влак",
        "myv": "Ломанть",
        "koi": "Персоналияэз",
        "udm": "Адямиос",
        "kbd": "Персонэхэр",
        "ab": "Ауаа",
        "bxr": "Хүнүүд",
    },
    "Science": {
        "ru": "Наука",
        "uk": "Наука",
        "be": "Навука",
        "be-tarask": "Навука",
        "bg": "Наука",
        "sr": "Наука",
        "mk": "Наука",
        "mn": "Шинжлэх_ухаан",
        "kk": "Ғылым",
        "ky": "Илим",
        "tg": "Илм",
        "tt": "Фән",
        "ba": "Фән",
        "cv": "Ăслăх",
        "rue": "Наука",
        "ce": "Ӏилма",
        "os": "Зонад",
        "sah": "Үөрэх",
        "mhr": "Шанче",
        "myv": "Тона",
        "koi": "Тӧдмалан",
        "udm": "Наука",
        "ab": "Аҭҵаарадырра",
        "bxr": "Шэнжэлхэ_ухаан",
    },
    "Sports": {
        "ru": "Спорт",
        "uk": "Спорт",
        "be": "Спорт",
        "be-tarask": "Спорт",
        "bg": "Спорт",
        "sr": "Спорт",
        "mk": "Спорт",
        "mn": "Спорт",
        "kk": "Спорт",
        "ky": "Спорт",
        "tg": "Варзиш",
        "tt": "Спорт",
        "ba": "Спорт",
        "cv": "Спорт",
        "rue": "Шпорт",
        "ce": "Спорт",
        "os": "Спорт",
        "sah": "Спорт",
        "mhr": "Спорт",
        "myv": "Спорт",
        "kv": "Спорт",
        "udm": "Спорт",
        "kbd": "Спорт",
        "ab": "Аспорт",
        "bxr": "Спорт",
        "tyv": "Спорт",
    },
}


# === FONCTIONS UTILITAIRES ===

def get_language_group(language_code: str) -> str:
    """
    Détermine le groupe d'une langue donnée.

    Args:
        language_code: code de la langue
            (ex: 'ru', 'uk', 'be')

    Returns:
        identifiant du groupe ('A', 'B', 'C', ou 'D')

    Raises:
        ValueError: si le code de langue est vide ou None
    """
    if not language_code:
        raise ValueError("Le code de langue ne peut pas être vide ou None")

    for group, languages in LANGUAGE_GROUPS.items():
        if language_code in languages:
            logging.debug(f"Langue '{language_code}' assignée au groupe {group}")
            return group

    # Groupe par défaut pour les langues non reconnues
    logging.warning(
        f"Langue '{language_code}' non reconnue, "
        f"assignation au groupe par défaut '{DEFAULT_LANGUAGE_GROUP}'"
    )
    return DEFAULT_LANGUAGE_GROUP


def get_adaptive_params(language_code: str) -> Dict[str, float]:
    """
    Récupère les paramètres adaptatifs pour une langue donnée.

    Les paramètres sont adaptés selon le groupe de la langue:
    - Groupe A: paramètres optimisés pour les langues bien dotées
    - Groupe B : paramètres intermédiaires
    - Groupe C: paramètres pour langues peu dotées
    - Groupe D: paramètres pour langues très peu dotées

    Args:
        language_code: code de la langue

    Returns:
        Dictionnaire des paramètres adaptatifs contenant:
        - min_char_length: longueur minimale des articles
        - max_token_length: longueur maximale (en tokens)
        - main_category_ratio: proportion de catégories principales
        - subcategory_ratio: proportion de sous-catégories
        - random_ratio: proportion d'articles aléatoires
        - fixed_selection_ratio: proportion de sélection ordonnée

    Raises:
        ValueError: si le code de langue est invalide
    """
    if not language_code:
        raise ValueError("Le code de langue ne peut pas être vide ou None")

    try:
        group = get_language_group(language_code)
        params = ADAPTIVE_PARAMS[group].copy()

        logging.debug(
            f"Paramètres adaptatifs récupérés pour '{language_code}'"
            f" (groupe {group})"
        )
        return params

    except KeyError as e:
        logging.error(f"Groupe '{group}' non trouvé dans ADAPTIVE_PARAMS: {e}")
        # Renvoyer des paramètres par défaut sécurisés
        return {
            "min_char_length": DEFAULT_MIN_CHAR_LENGTH,
            "max_token_length": DEFAULT_MAX_TOKEN_LENGTH,
            "main_category_ratio": 0.1,
            "subcategory_ratio": 0.4,
            "random_ratio": 0.5,
            "fixed_selection_ratio": 0.5,
        }


def get_target_for_language(language_code: str) -> int:
    """
    Détermine l'objectif de tokens pour une langue donnée.

    Gère le cas spécial du bélarussien (fusion des 2 variantes) et
    applique l'objectif standard du groupe pour les autres langues.

    Args:
        language_code: code de la langue

    Returns:
        Nombre de tokens cible pour cette langue

    Raises:
        ValueError: si le code de langue est invalide
    """
    if not language_code:
        raise ValueError("Le code de langue ne peut pas être vide ou None")

    # Cas spécial: variantes du bélarussien
    if language_code in ["be", "be-tarask"]:
        logging.info(
            f"Objectif spécial bélarussien appliqué pour '{language_code}':"
            f" {BELARUSIAN_TARGET}"
        )
        return BELARUSIAN_TARGET

    # Cas général: objectif basé sur le groupe
    try:
        group = get_language_group(language_code)
        target = TARGET_TOKENS_BY_GROUP[group]

        logging.debug(
            f"Objectif de tokens pour '{language_code}' "
            f"(groupe {group}): {target}"
        )
        return target

    except KeyError as e:
        logging.error(
            f"Groupe '{group}' non trouvé "
            f"dans TARGET_TOKENS_BY_GROUP: {e}"
        )
        logging.warning(
            f"Application de l'objectif par défaut: {DEFAULT_TOKEN_TARGET}"
        )
        return DEFAULT_TOKEN_TARGET


def validate_language_code(language_code: str) -> bool:
    """
    Valide qu'un code de langue est supporté par le système.

    Args:
        language_code: code de la langue à valider

    Returns:
        True si la langue est supportée, False sinon

    Examples:
        >>> validate_language_code('ru')
        True
        >>> validate_language_code('xyz')
        False
    """
    if not language_code:
        return False

    is_valid = language_code in LANGUAGES

    if not is_valid:
        logging.warning(f"Code de langue non supporté: '{language_code}'")

    return is_valid


def get_available_categories_for_language(language_code: str) -> List[str]:
    """
    Récupère la liste des catégories disponibles pour une langue donnée.

    Args:
        language_code: code de la langue

    Returns:
        liste des catégories disponibles (en anglais)

    Raises:
        ValueError: si le code de langue n'est pas supporté
    """
    if not validate_language_code(language_code):
        raise ValueError(f"Code de langue non pris en charge: '{language_code}'")

    available_categories = []

    for category in ALL_CATEGORIES:
        if language_code in CATEGORY_TRANSLATIONS.get(category, {}):
            available_categories.append(category)

    logging.debug(
        f"Catégories disponibles pour '{language_code}': {available_categories}"
    )
    return available_categories


def get_category_translation(category: str, language_code: str) -> Optional[str]:
    """
    Récupère la traduction d'une catégorie dans une langue donnée.

    Args:
        category: nom de la catégorie en anglais
        language_code: code de la langue cible

    Returns:
        traduction de la catégorie, ou None si non trouvée
    """
    if not category or not language_code:
        return None

    translation = CATEGORY_TRANSLATIONS.get(category, {}).get(language_code)

    if not translation:
        logging.warning(
            f"Traduction non trouvée pour '{category}' en '{language_code}'"
        )

    return translation
