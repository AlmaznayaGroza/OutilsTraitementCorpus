# TP1

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

* **Besoin**: comprendre l'impact psychologique et émotionnel du sexisme sur les personnes qui le subissent à travers l'analyse linguistique de témoignages, afin, idéalement, d'aider à la sensibilisation aux conséquences psychologiques de ces comportements et au développement de ressources de soutien adaptées pour les personnes affectées

* **Sujet**: l'expression linguistique des états psychologiques et émotionnels dans les témoignages de faits relevant du sexisme dans le cadre professionnel, avec un focus particulier sur les indicateurs textuels liés à la santé mentale (anxiété, stress, peur, trauma...)

* **Tâches**: plusieurs pistes d'analyses complémentaires qui pourraient être menées:
  * analyse lexicale des émotions et états psychologiques: identification et classification du vocabulaire émotionnel et psychologique trouvé dans les témoignages;
  * détection des marqueurs linguistiques de détresse: repérer des expressions, structures syntaxiques ou motifs narratifs associés à différents états de détresse psychologique;
  * analyse des stratégies d'adaptation: identifier les mécanismes d'adaptation mis en place par les victimes pour gérer l'impact psychologique de ce qu'elles ont vécu;
  * analyse des métaphores et images utilisées par les personnes pour conceptualiser leur expérience psychologique.

* **Type de données exploitées**: les témoignages textuels courts issus du corpus annoté "Paye Ton Corpus", qui recense 3021 posts pour 294954 mots au total.

* Ce corpus, qui est hébergé sur la plateforme Ortolang, est accessible gratuitement pour la recherche académique.