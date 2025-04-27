# TP

## Partie 1: étude de cas CoNLL 2003

1. CoNLL 2003 a pour tâche la reconnaissance des entités nommées (NER), indépendamment de la langue.

2. La base données de CoNLL est constituée de données textuelles annotées en anglais (dépêches de presse issues du Reuters Corpus) et en allemand (issues d'articles du quotidien "Frankfurter Rundschau"). Les phrases ont été segmentées en tokens, et les tokens sont présentés dans un format tabulaire: chaque ligne représente un token, avec les annotations dans différentes colonnes (plus précisément, et dans l'ordre: étiquettes de POS, de chunk syntaxique et d'entité nommée).

3. CoNLL 2003 répond au besoin suivant: créer un système standardisé pour évaluer les systèmes de NER à travers différentes langues, en s'affranchissant des ressources linguistiques spécifiques qui limitaient la portée des premiers systèmes. Les participants au projet ont constitué des corpus d'entraînement et de test pour la NER en anglais et en allemand (venant compléter ceux pour l'allemand et le néerlandais, créés pour la CoNLL 2022), en incluant une composante d'apprentissage automatique.

4. Sur ConNLL 2003, ont été entraînés les modèles suivants:
* modèles statistiques classiques: modèles de Markov cachées, modèles basés sur le maximum d'entropie, champs aléatoires conditionnels
* modèles neuronaux: réseaux récurrents bidirectionnels (BiLSTM)
* modèles basés sur les transformers: BERT et ses variantes (RoBERTa, DistilBERT, CamemBERT...)
* modèles de NER: le framework flair, spaCy EntityRecognizer

5. C'est un corpus multilingue, bilingue plus précisément (anglais et allemand).


## Partie 2: projet

### Définition des besoins du projet

- Dans quel **besoin** le projet s'inscrit-il?
Mon projet est guidé par un désir de préservation linguistico-culturelle et de représentation et de certaines langues beaucoup moins dotées en ressources et/ou parfois menacées pour des raisons politiques, et d'amélioration des outils de TAL pour les langues qui utilisent l'alphabet cyrillique, et les langues moins dotées en particulier.
Mon objectif est de développer un modèle capable de différencier des langues très diverses, ayant pour point commun d'utiliser un même système d'écriture, avec une considération d'inclusion d'un maximum de langues où les ressources permettent de constituer un corpus suffisant pour la tâche.
Mon projet répond également au besoin croissant d'outils permettant de traiter automatiquement des contenus multilingues sur le Web, par exemple pour les systèmes de traduction automatique qui doivent d'abord identifier correctement la langue source ou bien pour la modération de contenu.

- Quel **sujet** allez-vous traiter ?
C'est un projet de classification automatique de langues utilisant l'alphabet cyrillique comme système d'écriture officiel ou co-officiel. Ces langues sont très diverses, et peuvent être proches ou éloignées les unes des autres (langues slaves - parfois 2 variétés d'une même langue -, turciques, mongoles).
Le projet vise à créer un système capable d'identifier la langue d'un texte parmi plusieurs langues utilisant.
Et, contrairement à beaucoup de travaux existants, qui se concentrent sur les langues les plus répandues ou sur des paires de langues spécifiques, ce projet vise une couverture large et aussi inclusive que possible de l'espace linguistique cyrillique.

- Quel **type de tâche** allez-vous réaliser ?
C'est donc une tâche de classification automatique multiclasse, dont on peut détailler le processus comme suit:
1. Constitution d'un corpus équilibré et représentatif d'un maximum de langues cyrilliques (avec une phase initiale de test pour déterminer quelles langues disposent de ressources suffisantes pour être incluses de manière équilibrée)
2. Prétraitement des données textuelles
3. Entraînement de modèles de classification (Naive Bayes, SVM)
4. Évaluation des performances des modèles
5. Analyse des caractéristiques discriminantes entre les langues

- Quel **type de donnée**s allez-vous exploiter ?
Il s'agit de données textuelles provenant d'articles Wikipédia dans ces différentes langues. Pour chaque langue, le corpus est constitué d'articles issus de différentes grandes catégories thématiques (Culture, Histoire, Sciences, etc...) et de leurs sous-catégories, complétés par des articles sélectionnés complètement aléatoirement. Les textes recueillis sont de longueur variable (avec un minimum de caractères défini pour que l'article soit retenu, et un maximum de tokens extraits par article en cas d'articles très longs, par souci de représentativité).
Pour maximiser la robustesse du modèle, le corpus est conçu pour assurer une diversité thématique équilibrée à travers toutes les langues, en donnant la priorité aux catégories thématiques bien représentées dans la plupart des Wikipédias en cyrillique.

- Où allez-vous récupérer vos données ?
Les données sont récupérées depuis l'API de Wikipédia. Plus précisément, on utilise des requêtes à l'API pour obtenir des articles de catégories spécifiques, explorer des sous-catégories, récupérer des articles aléatoires et extraire le contenu textuel des articles.

- Sont-elles libres d'accès ?
Oui, les données de Wikipédia sont libres d'accès et peuvent être utilisées dans le cadre de projets académiques et de recherche, sous réserve de respecter les conditions d'utilisation de l'API (notamment les limites de taux de requêtes). Le contenu de Wikipédia est généralement sous licence Creative Commons Attribution-ShareAlike, ce qui permet son utilisation pour ce type de projet.