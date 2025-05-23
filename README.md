# Outils de Traitement de Corpus

## Documentation du projet de classification de langues cyrilliques

### Vue d'ensemble

Ce projet vise à développer un système de classification automatique capable d'identifier la langue d'un texte parmi plusieurs langues utilisant l'alphabet cyrillique. L'objectif est d'inclure un éventail de langues cyrilliques aussi large que possible, y compris des langues moins dotées en ressources, afin de contribuer à leur préservation linguistique et d'améliorer les outils de traitement automatique pour ces langues.

### Contexte et objectifs du projet

#### Motivation et enjeux sociétaux

Ce projet s'inscrit dans une démarche de préservation linguistico-culturelle et de développement d'outils de traitement automatique pour les langues utilisant l'alphabet cyrillique. Il répond à un double besoin:

1. **Préservation et représentation** des langues moins dotées en ressources, parfois menacées pour des raisons politiques
2. **Amélioration des outils de TAL** pour l'ensemble de l'espace linguistique cyrillique

L'objectif est de développer un modèle capable de différencier des langues très diverses, mais ayant pour point commun d'utiliser un même système d'écriture, avec une volonté d'inclusion maximale selon les ressources disponibles.

#### Applications pratiques

Ce projet répond également au besoin croissant d'outils de traitement automatique de contenus multilingues:
- **Systèmes de traduction automatique** nécessitant une identification préalable de la langue source
- **Modération de contenu** sur les plateformes numériques multilingues
- **Analyse de données** provenant de sources cyrilliques diverses

#### Tâche et approche

**Type de tâche**: classification automatique multiclasse pour l'identification de langues cyrilliques

**Corpus**: contrairement aux travaux existants centrés sur les langues majoritaires ou des paires spécifiques, ce projet vise une couverture large et inclusive de l'espace linguistique cyrillique.

**Données**: textes provenant d'articles Wikipédia, sélectionnés selon une stratégie adaptative pour garantir diversité thématique et équilibre entre langues.

### Couverture linguistique

Le projet vise à inclure un maximum de langues de types et de famille divers, et pour lesquelles le cyrillique est le système d'écriture officiel ou co-officiel.

Après des test préalables, 26 langues ont finalement été incluses dans le corpus:
- *slaves*: russe (ru), ukrainien (uk), bélarussien (be) dans ses 2 variantes (officielle et classqiue), bulgare (bg), macédonien (mk), serbe (sr), rusyn (rue);
- *turciques*: kazakh (kk), kirghize (ky), tatar (tt), bachkir (ba), tchouvache (cv), iaoukate (sah), touvin (tyv);
- *iraniennes*: tadjik (tg), ossète - dialecte iron (os);
- *ouraliennes (finno-permiennes)*: oudmourte (udm), komi (kv), komi-permyak (koi), erzya (langue mordve) (myv);
- *mongoles*: mongol (mn), bouriate (bxr);
- *caucasiennes*: tchétchène (ce), kabarde (kbd), abkhaze (ab).

Le corpus final comporte **43 langues** au total :
- **26 langues individuelles**
- **17 variantes mixtes** créées par augmentation de données pour améliorer la robustesse du modèle (notamment pour les langues moins dotées)

Cette approche permet de traiter à la fois des langues très bien dotées (russe, ukrainien, bulgare) et des langues moins représentées (komi-permyak, abkhaze, mari des prairies), contribuant ainsi à la diversité et à l'inclusivité du système.

### Choix méthodologiques spécifiques

#### Fusion des variantes bélarussiennes

Le bélarussien présente une situation sociolinguistique unique avec 2 variantes utilisés sur Wikipédia:

- `be` (**narkamoŭka** / наркамоўка): variante officielle standardisée
- `be-tarask` (**taraškevica** / тарашкевіца): variante historique pré-soviétique

##### Justification de la fusion*

Cette dualité, qui est principalement orthographique et parfois aussi lexicale, ne reflète pas une différence linguistique fondamentale mais plutôt des choix politiques et culturels distincts. La fusion des 2 variantes dans le corpus final se justifie par plusieurs considérations méthodologiques:

1. **Unité linguistique**: les deux variantes représentent la même langue avec des conventions orthographiques différentes, comparable aux variations entre l'anglais britannique et américain

2. **Robustesse statistique**: maintenir une séparation avec les mêmes objectifs quantitatifs aurait doublé la taille des échantillons pour le bélarussien, compromettant la significativité statistique des analyses

3. **Applications pratiques**: cette unification permet de créer des modèles plus robustes capables de gérer les variations orthographiques naturelles du bélarussien

4. **Représentativité**: l'unification offre une représentation plus complète et équilibrée de la langue bélarussienne dans ses usages contemporains

Cette approche respecte la diversité linguistique tout en optimisant l'efficacité du modèle de classification pour les applications pratiques.

### Structure du projet

```
cyrillic_language_classifier/
├── data/                                         # Données utilisées et générées par le projet
│   ├── raw/                                      # Corpus bruts collectés depuis Wikipédia
│   │   ├── final_corpus/                         # Corpus final par langue (27 langues)
│   │   ├── intermediate_articles/                # Articles intermédiaires de collecte
│   │   ├── temp_collection/                      # Collections temporaires en cours
│   │   ├── temp_collection_final/                # Collections finales temporaires
│   │   └── direct_scraping/                      # Données de scraping direct
│   ├── processed/                                # Données prétraitées et transformées
│   │   ├── cleaned/                              # Articles nettoyés par langue
│   │   ├── merged/                               # Articles fusionnés (be + be-tarask)
│   │   └── augmented/                            # Données augmentées pour l'entraînement
│   │       ├── all_augmented_articles.csv
│   │       ├── mixed_language_articles.csv
│   │       ├── perturbed_articles.csv
│   │       └── synthetic_articles.csv
│   ├── final/                                    # Ensembles finaux prêts pour l'entraînement
│   │   ├── train/                                # Données d'entraînement
│   │   ├── validation/                           # Données de validation
│   │   └── test/                                 # Données de test
│   └── final.zip                                 # Archive du corpus final
├── src/                                          # Code source du projet
│   ├── corpus/                                   # Module de collecte et préparation du corpus
│   │   ├── modules/                              # Modules refactorisés
│   │   │   ├── config.py                         # Configuration adaptative par langue
│   │   │   ├── api_utils.py                      # Utilitaires API Wikipedia
│   │   │   ├── text_processing.py                # Traitement et validation de textes
│   │   │   ├── article_collector.py              # Collecteur principal avec stratégies
│   │   │   ├── data_manager.py                   # Gestion et fusion de données
│   │   │   ├── stat_manager.py                   # Statistiques et métriques
│   │   │   └── cache_manager.py                  # Cache pour optimiser les performances
│   │   └── scripts/                              # Scripts de traitement de corpus
│   │       ├── create_corpus.py                  # Script principal de collecte
│   │       ├── collect_missing_langs.py          # Collecte complémentaire
│   │       ├── clean_corpus.py                   # Nettoyage des données
│   │       ├── augment_corpus.py                 # Augmentation de données
│   │       ├── merge_belarusian.py               # Fusion des variantes bélarusses
│   │       ├── consolidate_data.py               # Consolidation finale
│   │       └── split_datasets.py                 # Division train/val/test
│   ├── models/                                   # Entraînement et évaluation des modèles
│   │   └── language_detection.ipynb              # Notebook principal d'entraînement
│   └── visualization/                            # Scripts de visualisation et inspection
│       ├── visualize_corpus.py                   # Visualisations de corpus
│       └── inspect_aug_data.py                   # Inspection des données augmentées
├── results/                                      # Résultats d'analyse et d'évaluation
│   ├── figures/                                  # Visualisations et graphiques
│   │   ├── corpus_analysis/                      # Analyses du corpus
│   │   │   ├── distribution/                     # Distribution des langues et tokens
│   │   │   ├── cleaning/                         # Impact du nettoyage
│   │   │   └── augmentation/                     # Analyse de l'augmentation
│   │   └── model_evaluation/                     # Évaluation des modèles
│   ├── metrics/                                  # Métriques d'évaluation détaillées
│   │   ├── collection/                           # Métriques de collecte
│   │   │   ├── global/                           # Statistiques globales
│   │   │   ├── language/                         # Statistiques par langue
│   │   │   └── languages/                        # Analyses linguistiques
│   │   ├── corpus_analysis/                      # Analyses approfondies du corpus
│   │   │   ├── cleaning/                         # Métriques de nettoyage
│   │   │   └── augmentation/                     # Métriques d'augmentation
│   │   └── model_evaluation/                     # Évaluation des performances modèles
│   └── models/                                   # Modèles sauvegardés et checkpoints
│       ├── language-detection-20250521_224326/   # Modèle avec checkpoints
│       └── language-detection-final/             # Modèle final optimisé
├── logs/                                         # Journaux d'exécution
│   ├── cyrillique_collecte.log                   # Logs de collecte
│   └── training_session_*.log                    # Logs d'entraînement
├── requirements.txt                              # Dépendances du projet
└── resume_*.json                                 # États de reprise de collecte
```


### Collecte du corpus

La collecte du corpus est réalisée via l'API de Wikipédia, en utilisant différentes stratégies pour garantir un corpus équilibré et représentatif:
- articles issus de catégories principales (choisis par ordre d'apparition, puis aléatoirement);
- articles sélectionnés aléatoirement depuis les sous-catégories (avec plusieurs couches d'exploration);
- articles sélectionnés complètement aléatoirement, sans considération de thématique.

Concernant la collecte de données, je souhaite apporter une clarification importante: le corpus sur lequel repose mon projet, qui comprend près de 2 millions de tokens, a été constitué en utilisant l'API de Wikipédia avant que la restriction sur l'utilisation des APIs ne soit explicitement mentionnée dans les consignes.
Lorsque j'ai réalisé qu'on n'était pas censé utiliser d'API, j'avais déjà investi plusieurs dizaines d'heures dans le développement et l'optimisation de mon script de collecte (`create_corpus.py`), ainsi que dans le nettoyage et la préparation des données. Recommencer entièrement ce processus aurait compromis la qualité et l'ampleur de mon projet (j'avais pu constituer un corpus d'une qualité et d'une diversité vraiment intéressantes), particulièrement pour les langues cyrilliques moins représentées qui nécessitaient une stratégie de collecte spécifique. Par ailleurs, j'ai veillé lors de cette approche à bien respecter les bonnes pratiques d'extraction de données présentées en cours (respect des limitations de requêtes, identification appropriée, etc.).
Néanmoins, j'ai développé après-coup un script complémentaire (`direct_scraping.py`MediaWiki), qui illustre comment on pourrait collecter des textes cyrilliques sans passer par une API. Ce script, bien que moins exhaustif que ma méthode principale, implémente les techniques de web scraping requises.

### Pipeline du projet

Le traitement des données suit un pipeline en 7 étapes, conçu pour transformer des articles Wikipédia bruts en un corpus équilibré et prêt pour l'entraînement d'un modèle de classification:

#### 1. Collecte des données

##### Approche principale: API MediaWiki

**Scripts impliqués**:
- `create_corpus.py`: script principal de collecte, implémentant la stratégie adaptative selon les groupes de langues et les 3 méthodes de collecte définies (catégories principales, sous-catégories, articles aléatoires)
- `collect_missing_langs.py`: script de collecte ciblée pour les langues où la collecte initiale avait échoué ou était incomplète 
- `consolidate_data.py`: fusion et déduplication des données provenant des différentes sessions de collecte

**Objectif** : constituer un corpus équilibré de textes en langues cyrilliques à partir d'articles Wikipédia

**Processus** :
- Collecte adaptative selon 4 groupes de langues (A, B, C, D) avec paramètres spécifiques
- Stratégies multiples: catégories principales, sous-catégories, articles aléatoires
- Consolidation finale pour éviter les doublons entre collectes successives
- Gestion des reprises ciblées pour optimiser le temps de collecte

**Résultat**: ~1,9 millions de tokens sur 26 langues
**Sortie**: `data/raw/final_corpus/`

###### Gestion de la collecte longue durée
- Fichiers de reprise (`resume_*_state.json`): permettent de reprendre la collecte après interruption
- Collectes ciblées: optimisation pour les langues ayant rencontré des difficultés initiales

##### Approche alternative: Web scraping direct
- **Script** : `direct_scraping.py`
- **Techniques**: BeautifulSoup, gestion des headers, respect des robots.txt
- **Objectif**: démonstration des techniques sans API

#### 2. Nettoyage et préparation des données

##### Nettoyage du corpus brut

**Script**: `clean_corpus.py`
**Objectif**: éliminer les doublons, outliers et artifacts pour obtenir un corpus propre

**Opérations**:
- détection et suppression de doublons exacts et similaires (hash MD5)
- identification et suppression des outliers par analyse IQR par langue
- nettoyage des balises wiki résiduelles
- normalisation des caractères spéciaux

**Résultat**: corpus nettoyé avec statistiques de qualité
**Sortie**: `data/processed/cleaned/`

##### Fusion des variantes linguistiques
- **Script**: `merge_belarusian.py`
- **Objectif**: fusion des variantes standard (be) et classique (be-tarask)
- **Sortie**: `data/processed/merged/`

#### 3. Analyse et visualisation du corpus

**Objectif**: comprendre les caractéristiques du corpus pour en valider la qualité et guider les étapes suivantes

##### Analyse générale
**Script**: `visualize_corpus.py`
**Visualisations générées**:
  - distribution des tokens par langue et par groupe
  - analyse de la loi de Zipf pour chaque langue
  - identification des caractères cyrilliques distinctifs
  - corrélations entre longueur et type de source

##### Inspection des données
- **Script**: `inspect_aug_data.py`
- **Objectif**: validation de la qualité des données augmentées et analyse comparative avec le corpus original
- **Métriques**: entropie, équilibre des distributions
- **Sortie**: `results/figures/distribution/`

#### 4. Augmentation des données synthétiques

##### Stratégies d'augmentation
**Script**: `augment_corpus.py`
**3 méthodes**:
  1. **Génération synthétique**: modèles n-grammes pour recombiner des segments
  2. **Mélange inter-linguistique**: combinaison d'articles de langues proches
  3. **Perturbation**: variations orthographiques simulant des dialectes ou des erreurs typographiques
**Rationale**: ces 3 approches complémentaires permettent d'enrichir le corpus tout en préservant les caractéristiques linguistiques spécifiques à chaque langue

##### Priorité aux langues peu dotées
- Focus sur les groupes C et D (ab, kbd, koi, kv, mhr)
- Paramètres adaptatifs selon la disponibilité des données
- **Sortie**: `data/processed/augmented/`

#### 5. Préparation des ensembles finaux

##### Division stratifiée
- **Script**: `split_datasets.py`
- **Stratégie**: Division par langue et par source (original/augmenté)
- **Ratios**: 80% train / 10% validation / 10% test
- **Sortie**: `data/final/` (train/, validation/, test/)

#### 6. Adaptation d'un modèle Transformer

##### Choix du modèle
- **Modèle de base**: XLM-RoBERTa pré-entraîné pour la détection de langues
- **Pourquoi ce modèle?**: XLM-RoBERTa offre une base multilingue robuste et une architecture adaptée aux langues utilisant des scripts non-latins
- **Adaptation**: fine-tuning de la couche de classification pour 43 langues

##### Entraînement
- **Script**: `language_detection_training.ipynb` (Colab)
- **Hyperparamètres**:
      12 époques
      batch size: 32 pour l'entraînement, 64 pour l'évaluation
      learning rate: 8e-5
      warmup steps: 200
      weight decay: 0.01
- **Tokenisation**: séquences de 64 tokens maximum
- **Optimisations**: précision mixte (fp16), warmup progressif, régularisation par weight decay

##### Architecture finale
- Tokenizer XLM-RoBERTa + couche de classification adaptée
- Prise en charge de 43 langues cyrilliques (dont 26 naturelles)

#### 7. Évaluation du modèle

**Approche d'évaluation**: évaluation multi-facettes pour évaluer tant la performance brute que la robustesse et la généralisation du modèle

##### Métriques intrinsèques
- **Accuracy globale**: 99,46% sur 2043 exemples de test
- **F1-scores** par langue et par groupe linguistique
- **Matrices de confusion** détaillées
- **Top-3 accuracy**: 99,76%

##### Métriques extrinsèques
- **Analyse des erreurs**: confusions intra vs. inter-groupes (9,1% d'erreurs intra-groupe)
- **Robustesse**: performance quasi-constante selon la longueur du texte
- **Comparaison**: Transformer vs. modèles classiques (TF-IDF + SVM)

##### Analyses avancées
- Calibration exceptionnelle (ECE = 0,0029)
- 20 langues avec performance parfaite (exactitude et F1-score)
- Confiance moyenne du modèle: ???%

### Résultats principaux

#### Performance globale
- **99,46% de précision** sur 43 langues
- **Taux d'erreur** minimal: seulement 11 erreurs sur 2043 exemples de test (soit 0,54%)
- **Calibration quasi-parfaite**: ECE de 0,0029
- **Robustesse** démontrée sur différentes longueurs de texte

#### Analyse des erreurs

- Total de 11 erreurs sur 2043 exemples (0,54%)
- Confusions les plus fréquentes: bouriate (bxr) → iakoute (sah) et oudmourte (udm) → mélange kabarde-tchétchène (kbd_ce_mix), avec 2 erreurs chacune
- 45.5% des erreurs sont linguistiquement justifiées (toutes des confusions avec des mélanges artificiels contenant la langue prédite)
- Seulement 9,1% d'erreurs intra-famille, démontrant une très bonne discrimination entre langues apparentées
- Métacognition sophistiquée: le modèle démontre une bonne auto-évaluation avec des erreurs concentrées dans les zones de confiance modérée (50-80%) et une calibration quasi-parfaite (ECE = 0.0029) pour les prédictions hautement confiantes.

#### Contributions principales
1. **Corpus multilingue**: corpus équilibré de grande ampleur pour les langues cyrilliques
2. **Stratégies d'augmentation** de diverses sortes pour les langues peu dotées
3. **Évaluation complète**: analyses détaillées des performances par groupes linguistiques
4. **Performances très satisfaisantes**: précision de 99,46% sur un ensemble diversifié de 43 langues

Le corpus obtenu étant relativement original et s'avérant aussi qualitatif, il a été publié sur Hugging Face pour être mis à la disposition de la communauté de recherche en TAL. En effet, ce corpus se distingue par son approche inclusive des langues cyrilliques, incluant de nombreuses langues minoritaires rarement représentées dans les ressources numériques existantes, et constitue ainsi une contribution pour le développement d'outils de TAL équitables et la préservation du patrimoine linguistique cyrillique.
[Accès au corpus](https://huggingface.co/datasets/AlmaznayaGroza/cyrillic-language-classification)

### Installation et utilisation

#### Pré-requis

**Versions Python recommandées**: 3.11 ou 3.12
> Note: en raison de problèmes de compatibilité avec certaines bibliothèques (notamment spaCy), Python 3.13 n'est pas encore pleinement supporté.

#### Installation de base

Ce projet nécessite plusieurs bibliothèques Python:

```bash
pip install -r requirements.txt
```

#### Reproduction du pipeline complet

**1. Collecte** (via l'API)
```bash
PYTHONPATH=src python src/corpus/scripts/create_corpus.py
``

**2. Nettoyage**
`python src/corpus/clean_corpus.py`

**3. Fusion des variantes**
`python src/corpus/merge_belarusian.py`

**4. Visualisation**
`python src/visualization/visualize_corpus.py`

**5. Augmentation**
`python src/corpus/augment_corpus.py`

**6. Division finale**
`python src/corpus/split_datasets.py`

**7. Entraînement** (sur Colab avec GPU)
Voir le notebook `language_detection_training.ipynb`

#### Modèle entraîné

En raison des limitations de taille de GitHub, le modèle final n'est pas inclus dans ce dépôt, mais le processus complet d'entraînement est documenté à travers les visualisations et métriques incluses dans les répertoires results/figures/ et results/metrics/.

#### Conclusion

Ce travail démontre qu'une approche méthodique combinant techniques modernes et évaluation rigoureuse peut produire des systèmes atteignant un niveau de performance et de compréhension très satisfaisants pour des tâches de classification linguistique complexe.