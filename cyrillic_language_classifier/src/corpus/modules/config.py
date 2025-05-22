# === CONFIGURATION GÉNÉRALE ===

# Limiter le temps à ~6h par langue (script déjà très long)
TIME_LIMIT = 22000

# Objectif spécial pour le bélarussien (car les 2 variantes de la langue seront ensuite fusionnées)
BELARUSIAN_TARGET = 60000


# === GROUPES DE LANGUES ===

# Définition des groupes de langues basés sur les résultats de plusieurs tests préalables
LANGUAGE_GROUPS = {
    'A': [
        'ba', 'be', 'be-tarask', 'bg', 'bxr', 'cv', 'kk', 'ky', 'mk', 'mn',
        'ru', 'rue', 'sah', 'sr', 'tg', 'tt', 'tyv', 'uk'], # langues très bien dotées
    'B': ['ce', 'os', 'udm'],  # moins dotées
    'C': ['ab', 'koi', 'myv'], # encore moins dotées
    'D': ['kbd', 'kv', 'mhr']  # très peu dotées
}

LANGUAGES = []
for group, languages in LANGUAGE_GROUPS.items():
    LANGUAGES.extend(languages)


# === PARAMÈTRES ADAPTATIFS ===

TARGET_TOKENS_BY_GROUP = {
    'A': 100000,
    'B': 90000,
    'C': 80000,
    'D': 75000
}

# Profondeur d'exploration adaptative par groupe
MAX_DEPTHS_BY_GROUP = {
    'A': 3,  # une exploration superficielle suffit (plus rapide)
    'B': 6,  # profondeur intermédiaire
    'C': 9,  # exploration plus profonde nécessaire
    'D': 12
}

# Paramètres adaptatifs par groupe
ADAPTIVE_PARAMS = {
    'A': {
        'min_char_length': 500,         # nb minimum de caractères qu'un article doit avoir pour être retenu
        'max_token_length': 1000,       # nb maximum de tokens à conserver pour chaque article
        'main_category_ratio': 0.25,    # 30% catégories principales
        'subcategory_ratio': 0.35,      # 30% sous-catégories
        'random_ratio': 0.4,            # 40% articles aléatoires
        'fixed_selection_ratio': 0.34   # 60% des articles par ordre 'naturel', 40% aléatoire
    },
    'B': {
        'min_char_length': 450,
        'max_token_length': 1000,
        'main_category_ratio': 0.1,     # 10% catégories principales
        'subcategory_ratio': 0.45,      # 45% sous-catégories
        'random_ratio': 0.45,           # 45% articles aléatoires
        'fixed_selection_ratio': 0.5    # 50% des articles par ordre 'naturel', 60% aléatoire
    },
    'C': {
        'min_char_length': 400,
        'max_token_length': 1250,
        'main_category_ratio': 0.1,     # 10% catégories principales
        'subcategory_ratio': 0.4,       # 40% sous-catégories
        'random_ratio': 0.5,            # 50% articles aléatoires
        'fixed_selection_ratio': 0.5    # 50% des articles par ordre naturel, 65% aléatoire
    },
    'D': {
        'min_char_length': 300,
        'max_token_length': 1500,
        'main_category_ratio': 0.1,     # 10% catégories principales
        'subcategory_ratio': 0.4,       # 40% sous-catégories
        'random_ratio': 0.5,            # 50% articles aléatoires
        'fixed_selection_ratio': 0.67   # 67% des articles par ordre naturel, 75% aléatoire
    }
}


# === TRADUCTIONS ET PRÉFIXES ===

# Catégories thématiques
ALL_CATEGORIES = [
    "Culture", "History", "Geography", "Politics", 
    "People", "Science", "Sports"
]

# Préfixe de catégorie selon la langue
CATEGORY_PREFIXES = {
    'ru': 'Категория:',        # russe
    'uk': 'Категорія:',        # ukrainien
    'be': 'Катэгорыя:',        # bélarussien moderne
    'be-tarask': 'Катэгорыя:', # bélarussien classique
    'bg': 'Категория:',        # bulgare
    'sr': 'Категорија:',       # serbe
    'mk': 'Категорија:',       # macédonien
    'mn': 'Ангилал:',          # mongol
    'kk': 'Санат:',            # kazakh
    'ky': 'Категория:',        # kirghize
    'tg': 'Гурӯҳ:',            # tadjik
    'tt': 'Төркем:',           # tatar
    'ba': 'Категория:',        # bachkir
    'cv': 'Категори:',         # tchouvache
    'rue': 'Катеґорія:',       # rusyn
    'ce': 'Категори:',         # tchétchène
    'os': 'Категори:',         # ossète
    'sah': 'Категория:',       # iakoute (aka sakha)
    'mhr': 'Категорий:',       # mari des prairies
    'myv': 'Категория:',       # erzya
    'koi': 'Категория:',       # komi-permyak
    'kv': 'Категория:',        # komi
    'udm': 'Категория:',       # oudmourte
    'kbd': 'Категориэ:',       # kabarde (aka circassien oriental)
    'ab': 'Акатегориа:',       # abkhaze
    'bxr': 'Категори:',        # bouriate
    'tyv': 'Аңгылал:',         # touvin
}

# Catégories sélectionnées
CATEGORY_TRANSLATIONS = {
    'Culture': {
        'ru': 'Культура',
        'uk': 'Культура',
        'be': 'Культура',
        'be-tarask': 'Культура',
        'bg': 'Култура',
        'sr': 'Култура',
        'mk': 'Култура',
        'mn': 'Соёл',
        'kk': 'Мәдениет',
        'ky': 'Маданият',
        'tg': 'Фарҳанг',
        'tt': 'Мәдәният',
        'ba': 'Мәҙәниәт',
        'cv': 'Культура',
        'rue': 'Култура',
        'ce': 'Оьздангалла',
        'os': 'Культурæ',
        'sah': 'Култуура',
        'mhr': 'Тӱвыра',
        'myv': 'Культура',
        'koi': 'Культура',
        'kv': 'Культура',
        'udm': 'Культура',
        'kbd': 'Културэ',
        'ab': 'Акультура',
        'bxr': 'Соёл',
        'tyv': 'Культура',
    },
    'History': {
        'ru': 'История',
        'uk': 'Історія',
        'be': 'Гісторыя',
        'be-tarask': 'Гісторыя',
        'bg': 'История',
        'sr': 'Историја',
        'mk': 'Историја',
        'mn': 'Түүх',
        'kk': 'Тарих',
        'ky': 'Тарых',
        'tg': 'Таърих',
        'tt': 'Тарих',
        'ba': 'Тарих',
        'cv': 'Истори',
        'rue': 'Історія',
        'ce': 'Истори',
        'os': 'Истори',
        'sah': 'История',
        'mhr': 'Эртымгорно',
        'myv': 'История',
        'koi': 'История',
        'kv': 'Важвылӧм',
        'udm': 'История',
        'kbd': 'Тхыдэ',
        'ab': 'Аҭоурых',
        'bxr': 'Түүхэ',
        'tyv': 'Төөгү',
    },
    'Geography': {
        'ru': 'География',
        'uk': 'Географія',
        'be': 'Геаграфія',
        'be-tarask': 'Геаграфія',
        'bg': 'География',
        'sr': 'Географија',
        'mk': 'Географија',
        'mn': 'Газар_зүй',
        'kk': 'География',
        'ky': 'География',
        'tg': 'Ҷуғрофиё',
        'tt': 'География',
        'ba': 'География',
        'cv': 'Географи',
        'rue': 'Ґеоґрафія',
        'ce': 'Географи',
        'os': 'Географи',
        'sah': 'География',
        'mhr': 'Географий',
        'myv': 'География',
        'koi': 'География',
        'kv': 'География',
        'udm': 'География',
        'kbd': 'Хэкумэтх',
        'ab': 'Агеографиа',
        'bxr': 'Газар_зүй',
        'tyv': 'География',
    },
    'Politics': {
        'ru': 'Политика',
        'uk': 'Політика',
        'be': 'Палітыка',
        'be-tarask': 'Палітыка',
        'bg': 'Политика',
        'sr': 'Политика',
        'mk': 'Политика',
        'mn': 'Улс_төр',
        'kk': 'Саясат',
        'ky': 'Саясат',
        'tg': 'Сиёсат',
        'tt': 'Сәясәт',
        'ba': 'Сәйәсәт',
        'cv': 'Политика',
        'rue': 'Політіка',
        'ce': 'Политика',
        'os': 'Политикæ',
        'sah': 'Политика',
        'mhr': 'Политике',
        'myv': 'Политикась',
        'koi': 'Политика',
        'kv': 'Политика',
        'udm': 'Политика',
        'kbd': 'Политикэ',
        'ab': 'Аполитика',
        'bxr': 'Улас_түрэ',
    },
    'People': {
        'ru': 'Люди',
        'uk': 'Персоналії',
        'be': 'Асобы',
        'be-tarask': 'Асобы',
        'bg': 'Хора',
        'sr': 'Људи',
        'mk': 'Луѓе',
        'mn': 'Хүн',
        'kk': 'Тұлғалар',
        'ky': 'Адамдар',
        'tg': 'Одамон',
        'tt': 'Шәхесләр',
        'ba': 'Кешеләр',
        'cv': 'Çынсем',
        'rue': 'Люде',
        'ce': 'Нах',
        'os': 'Зындгонд_адæм',
        'sah': 'Дьон',
        'mhr': 'Еҥ-влак',
        'myv': 'Ломанть',
        'koi': 'Персоналияэз',
        'udm': 'Адямиос',
        'kbd': 'Персонэхэр',
        'ab': 'Ауаа',
        'bxr': 'Хүнүүд',
    },
    'Science': {
        'ru': 'Наука',
        'uk': 'Наука',
        'be': 'Навука',
        'be-tarask': 'Навука',
        'bg': 'Наука',
        'sr': 'Наука',
        'mk': 'Наука',
        'mn': 'Шинжлэх_ухаан',
        'kk': 'Ғылым',
        'ky': 'Илим',
        'tg': 'Илм',
        'tt': 'Фән',
        'ba': 'Фән',
        'cv': 'Ăслăх',
        'rue': 'Наука',
        'ce': 'Ӏилма',
        'os': 'Зонад',
        'sah': 'Үөрэх',
        'mhr': 'Шанче',
        'myv': 'Тона',
        'koi': 'Тӧдмалан',
        'udm': 'Наука',
        'ab': 'Аҭҵаарадырра',
        'bxr': 'Шэнжэлхэ_ухаан',
    },
    'Sports': {
        'ru': 'Спорт',
        'uk': 'Спорт',
        'be': 'Спорт',
        'be-tarask': 'Спорт',
        'bg': 'Спорт',
        'sr': 'Спорт',
        'mk': 'Спорт',
        'mn': 'Спорт',
        'kk': 'Спорт',
        'ky': 'Спорт',
        'tg': 'Варзиш',
        'tt': 'Спорт',
        'ba': 'Спорт',
        'cv': 'Спорт',
        'rue': 'Шпорт',
        'ce': 'Спорт',
        'os': 'Спорт',
        'sah': 'Спорт',
        'mhr': 'Спорт',
        'myv': 'Спорт',
        'kv': 'Спорт',
        'udm': 'Спорт',
        'kbd': 'Спорт',
        'ab': 'Аспорт',
        'bxr': 'Спорт',
        'tyv': 'Спорт',
    }
}


def get_language_group(language_code):
    """Détermine le groupe d'une langue"""
    for group, languages in LANGUAGE_GROUPS.items():
        if language_code in languages:
            return group
    return 'C'  # par défaut, considérer comme groupe C (moins doté)


def get_adaptive_params(language_code):
    """Récupère les paramètres adaptatifs pour une langue donnée."""
    group = get_language_group(language_code)
    return ADAPTIVE_PARAMS[group]


def get_target_for_language(language_code):
    # Exception pour les variantes du bélarussien
    if language_code in ["be", "be-tarask"]:
        return BELARUSIAN_TARGET
        
    # Sinon, utiliser l'objectif standard du groupe
    group = get_language_group(language_code)
    return TARGET_TOKENS_BY_GROUP[group]