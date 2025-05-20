import transformers
import datasets
import evaluate
import torch
import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Désactiver wandb
os.environ["WANDB_DISABLED"] = "true"

# Imports spécifiques de Transformers
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import evaluate

# Configurer la journalisation
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Créer les répertoires nécessaires
os.makedirs('/content/results/models/language-detection', exist_ok=True)
os.makedirs('/content/logs', exist_ok=True)
os.makedirs('/content/results/metrics', exist_ok=True)
os.makedirs('/content/results/figures', exist_ok=True)

logger.info("Début du script d'entraînement avec modèle pré-entraîné pour la détection de langues")

try:
    # Charger les données avec les chemins corrects
    logger.info("Chargement des données d'entraînement...")
    train_df = pd.read_csv('/content/data/final/train/train_corpus.csv')
    logger.info(f"Données d'entraînement chargées : {len(train_df)} exemples")

    logger.info("Chargement des données de validation...")
    val_df = pd.read_csv('/content/data/final/validation/validation_corpus.csv')
    logger.info(f"Données de validation chargées : {len(val_df)} exemples")

    logger.info("Chargement des données de test...")
    test_df = pd.read_csv('/content/data/final/test/test_corpus.csv')
    logger.info(f"Données de test chargées : {len(test_df)} exemples")

    # Préparer les mappages
    logger.info("Préparation des mappages langue -> ID...")
    languages = sorted(train_df['language'].unique())
    lang_to_id = {lang: idx for idx, lang in enumerate(languages)}
    id_to_lang = {idx: lang for lang, idx in lang_to_id.items()}
    logger.info(f"Nombre de langues identifiées : {len(languages)}")

    # Sauvegarder les mappages
    with open('/content/results/metrics/lang_mappings.json', 'w') as f:
        json.dump({
            'lang_to_id': lang_to_id,
            'id_to_lang': {str(k): v for k, v in id_to_lang.items()}
        }, f)
    logger.info("Mappages sauvegardés dans results/metrics/lang_mappings.json")

    # Vérifier si un GPU est disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Utilisation de : {device}")

    # Initialiser le tokenizer et le modèle pré-entraîné pour la détection de langues
    logger.info("Chargement du tokenizer et du modèle pré-entraîné pour la détection de langues...")
    model_name = "papluca/xlm-roberta-base-language-detection"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Afficher les langues supportées par le modèle pré-entraîné
    pretrained_languages = list(model.config.id2label.values())
    logger.info(f"Langues supportées par le modèle pré-entraîné : {pretrained_languages}")

    # Adapter la couche de classification pour nos langues
    logger.info("Adaptation de la couche de classification pour nos langues...")
    model.config.id2label = id_to_lang
    model.config.label2id = lang_to_id
    model.classifier.out_proj = torch.nn.Linear(model.classifier.dense.out_features, len(languages))
    model.num_labels = len(languages)
    logger.info("Couche de classification adaptée")

    # Fonction pour préparer les datasets
    def prepare_datasets(df, tokenizer, max_length=64):
        logger.info(f"Préparation d'un dataset de {len(df)} exemples...")
        texts = df['text'].tolist()
        labels = [lang_to_id[lang] for lang in df['language']]

        logger.info("Tokenisation des textes...")
        encodings = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        logger.info("Tokenisation terminée")

        dataset = Dataset.from_dict({
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': labels
        })
        logger.info("Dataset préparé")

        return dataset

    # Préparer les datasets
    logger.info("Préparation du dataset d'entraînement...")
    train_dataset = prepare_datasets(train_df, tokenizer)
    logger.info("Préparation du dataset de validation...")
    val_dataset = prepare_datasets(val_df, tokenizer)
    logger.info("Préparation du dataset de test...")
    test_dataset = prepare_datasets(test_df, tokenizer)

    # Fonction de calcul des métriques
    def compute_metrics(pred):
        logger.info("Calcul des métriques d'évaluation...")
        metric = evaluate.load("accuracy")
        logits, labels = pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Définir les arguments d'entraînement
    logger.info("Configuration des arguments d'entraînement...")
    training_args = TrainingArguments(
        output_dir='/content/results/models/language-detection',
        num_train_epochs=12,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='/content/logs',
        logging_steps=20,
        eval_strategy="epoch",  # Utilisez "eval_strategy" au lieu de "evaluation_strategy"
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
    )
    logger.info("Arguments d'entraînement configurés")

    # Initialiser le Trainer
    logger.info("Initialisation du Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    logger.info("Trainer initialisé")

    # Lancer l'entraînement
    logger.info("Début de l'entraînement...")
    train_result = trainer.train()
    train_metrics = train_result.metrics

    epochs = list(range(1, int(training_args.num_train_epochs) + 1))
    train_loss = [train_metrics["train_loss"]] * len(epochs)  # approximation, on n'a que la valeur finale
    val_loss = []
    accuracy = []

    # Récupérer les métriques d'évaluation par époque à partir des logs
    for i in range(len(epochs)):
        eval_metrics = trainer.evaluate(eval_dataset=val_dataset)
        val_loss.append(eval_metrics["eval_loss"])
        accuracy.append(eval_metrics["eval_accuracy"])

    # Visualiser l'évolution
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Époque')
    ax1.set_ylabel('Perte')
    ax1.plot(epochs, train_loss, 'b-', label='Perte d\'entraînement')
    ax1.plot(epochs, val_loss, 'g-', label='Perte de validation')
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Précision')
    ax2.plot(epochs, accuracy, 'r-', label='Précision')
    ax2.tick_params(axis='y')

    fig.tight_layout()
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    plt.title('Évolution de l\'entraînement')
    plt.savefig('/content/results/figures/training_evolution.png')
    plt.close()
    logger.info("Entraînement terminé")

    # Sauvegarder le modèle final
    logger.info("Sauvegarde du modèle final...")
    trainer.save_model('/content/results/models/language-detection-final')
    logger.info("Modèle sauvegardé dans results/models/language-detection-final")

    # Évaluation sur l'ensemble de test
    logger.info("Évaluation du modèle sur l'ensemble de test...")
    test_results = trainer.predict(test_dataset)
    predictions = np.argmax(test_results.predictions, axis=-1)

    # Convertir les prédictions en noms de langues
    pred_langs = [id_to_lang[pred] for pred in predictions]
    true_langs = test_df['language'].tolist()

    # Générer le rapport de classification
    logger.info("Génération du rapport de classification...")
    report = classification_report(true_langs, pred_langs, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv('/content/results/metrics/classification_report.csv')

    # Générer la matrice de confusion
    logger.info("Génération de la matrice de confusion...")
    plt.figure(figsize=(15, 12))
    conf_matrix = confusion_matrix(true_langs, pred_langs, labels=sorted(test_df['language'].unique()))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
               xticklabels=sorted(test_df['language'].unique()),
               yticklabels=sorted(test_df['language'].unique()))
    plt.title('Matrice de confusion par langue')
    plt.xlabel('Langue prédite')
    plt.ylabel('Langue réelle')
    plt.tight_layout()
    plt.savefig('/content/results/figures/confusion_matrix.png')

    # Performance par type d'augmentation (si applicable)
    if 'source' in test_df.columns:
        logger.info("Analyse des performances par méthode d'augmentation...")
        aug_performance = {}
        for source in test_df['source'].unique():
            source_indices = test_df[test_df['source'] == source].index
            source_true = [true_langs[i] for i in source_indices]
            source_pred = [pred_langs[i] for i in source_indices]
            source_acc = sum(t == p for t, p in zip(source_true, source_pred)) / len(source_true)
            aug_performance[source] = source_acc

        # Visualiser les performances par type d'augmentation
        plt.figure(figsize=(10, 6))
        plt.bar(aug_performance.keys(), aug_performance.values())
        plt.title('Précision par méthode d\'augmentation')
        plt.ylabel('Précision')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('/content/results/figures/augmentation_performance.png')

        # Graphique des F1-scores par langue (pour les langues originales uniquement)
        logger.info("Génération du graphique des F1-scores par langue...")
        original_langs = [l for l in report_df.index if not ('_mix' in l) and l not in ['accuracy', 'macro avg', 'weighted avg']]
        plt.figure(figsize=(15, 8))
        scores = [report_df.loc[lang, 'f1-score'] for lang in original_langs]
        plt.bar(original_langs, scores)
        plt.title('Score F1 par langue (langues originales uniquement)')
        plt.xlabel('Langue')
        plt.ylabel('Score F1')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig('/content/results/figures/f1_scores_by_language.png')
        plt.close()

    logger.info("Script d'entraînement et d'évaluation terminé avec succès")

    # Analyse des erreurs les plus fréquentes
    logger.info("Analyse des erreurs les plus fréquentes...")
    errors = []
    for true, pred in zip(true_langs, pred_langs):
        if true != pred:
            errors.append((true, pred))

    # Comptage des erreurs par paire de langues
    error_counts = {}
    for true, pred in errors:
        key = (true, pred)
        error_counts[key] = error_counts.get(key, 0) + 1

    # Affichage des 10 erreurs les plus fréquentes
    most_common_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    logger.info("Les 10 confusions les plus fréquentes :")
    for (true, pred), count in most_common_errors:
        logger.info(f"  {true} -> {pred}: {count} occurences")

    # Visualisation des erreurs les plus fréquentes
    if most_common_errors:
        plt.figure(figsize=(12, 8))
        error_pairs = [f"{true}->{pred}" for (true, pred), _ in most_common_errors]
        error_values = [count for _, count in most_common_errors]
        plt.bar(error_pairs, error_values)
        plt.title('Les 10 confusions les plus fréquentes')
        plt.xlabel('Paire de langues (réelle->prédite)')
        plt.ylabel('Nombre d\'occurences')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('/content/results/figures/most_common_errors.png')
        plt.close()

    # Analyse par groupes linguistiques
    logger.info("Analyse par groupes linguistiques...")
    language_groups = {
        'Langues slaves orientales': ['ru', 'uk', 'be', 'rue'],
        'Langues slaves méridionales': ['bg', 'mk', 'sr'],
        'Langues turciques': ['tt', 'ba', 'kk', 'ky'],
        'Langues finno-ougriennes': ['koi', 'kv', 'udm', 'mhr', 'myv'],
        'Langues caucasiennes': ['ab', 'kbd', 'ce'],
        'Autres langues': ['bxr', 'cv', 'mn', 'os', 'sah', 'tg', 'tyv']
    }

    # Calculer les performances par groupe
    group_performances = {}
    for group_name, langs in language_groups.items():
        group_indices = [i for i, lang in enumerate(true_langs) if lang in langs]
        if group_indices:
            group_true = [true_langs[i] for i in group_indices]
            group_pred = [pred_langs[i] for i in group_indices]
            group_acc = sum(t == p for t, p in zip(group_true, group_pred)) / len(group_true)
            group_performances[group_name] = group_acc

    # Visualiser les performances par groupe
    plt.figure(figsize=(12, 6))
    plt.bar(group_performances.keys(), group_performances.values())
    plt.title('Précision par groupe linguistique')
    plt.ylabel('Précision')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('/content/results/figures/language_group_performance.png')
    plt.close()

    # Afficher les résultats par groupe
    logger.info("Performances par groupe linguistique:")
    for group, acc in group_performances.items():
        logger.info(f"  {group}: {acc:.4f}")

    # Comparaison avec un modèle TF-IDF + SVM
    logger.info("Entraînement d'un modèle TF-IDF + SVM pour comparaison...")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import Pipeline

    # Combiner train et validation pour l'entraînement
    train_val_df = pd.concat([train_df, val_df])

    # Créer un pipeline simple
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 3), analyzer='char', min_df=2, max_features=20000)),
        ('classifier', LinearSVC(C=10, dual=False, max_iter=5000))
    ])

    # Entraîner le modèle
    pipeline.fit(train_val_df['text'], train_val_df['language'])

    # Évaluer sur l'ensemble de test
    y_pred = pipeline.predict(test_df['text'])
    svm_report = classification_report(test_df['language'], y_pred, output_dict=True)
    svm_accuracy = svm_report['accuracy']

    logger.info(f"Exactitude du modèle TF-IDF + SVM: {svm_accuracy:.4f}")
    logger.info(f"Comparaison: Transformer ({report['accuracy']:.4f}) vs TF-IDF+SVM ({svm_accuracy:.4f})")

    # Sauvegarder le rapport de classification du modèle SVM
    svm_report_df = pd.DataFrame(svm_report).transpose()
    svm_report_df.to_csv('/content/results/metrics/svm_classification_report.csv')

    # Visualiser la comparaison des 2 approches
    plt.figure(figsize=(8, 6))
    plt.bar(['XLM-RoBERTa', 'TF-IDF + SVM'], [report['accuracy'], svm_accuracy])
    plt.title('Comparaison des approches')
    plt.ylabel('Exactitude')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('/content/results/figures/model_comparison.png')
    plt.close()

    logger.info("Analyses complémentaires des résultats...")

    # 1. Indice de certitude linguistique
    logger.info("Calcul de l'indice de certitude linguistique...")
    softmax = torch.nn.Softmax(dim=1)
    confidences = softmax(torch.tensor(test_results.predictions)).max(dim=1).values

    # Confiance moyenne par langue
    avg_confidence_by_lang = {}
    correct_confidence_by_lang = {}  # Confiance quand la prédiction est correcte
    incorrect_confidence_by_lang = {}  # Confiance quand la prédiction est incorrecte

    for lang in languages:
        # Indices des exemples de cette langue
        lang_indices = [i for i, l in enumerate(true_langs) if l == lang]

        if not lang_indices:  # Si aucun exemple pour cette langue
            continue

        # Confiance moyenne globale pour cette langue
        lang_confidences = [confidences[i].item() for i in lang_indices]
        avg_confidence_by_lang[lang] = sum(lang_confidences) / len(lang_confidences)

        # Séparer les prédictions correctes et incorrectes
        correct_indices = [i for i in lang_indices if pred_langs[i] == true_langs[i]]
        incorrect_indices = [i for i in lang_indices if pred_langs[i] != true_langs[i]]

        if correct_indices:
            correct_conf = [confidences[i].item() for i in correct_indices]
            correct_confidence_by_lang[lang] = sum(correct_conf) / len(correct_conf)
        else:
            correct_confidence_by_lang[lang] = 0

        if incorrect_indices:
            incorrect_conf = [confidences[i].item() for i in incorrect_indices]
            incorrect_confidence_by_lang[lang] = sum(incorrect_conf) / len(incorrect_conf)
        else:
            incorrect_confidence_by_lang[lang] = 0

    # Visualisation des résultats
    plt.figure(figsize=(15, 8))
    langs = list(avg_confidence_by_lang.keys())
    avg_conf = [avg_confidence_by_lang[l] for l in langs]

    # Trier par confiance moyenne
    sorted_indices = sorted(range(len(avg_conf)), key=lambda i: avg_conf[i], reverse=True)
    sorted_langs = [langs[i] for i in sorted_indices]
    sorted_conf = [avg_conf[i] for i in sorted_indices]

    plt.bar(sorted_langs, sorted_conf)
    plt.title('Indice de certitude linguistique par langue')
    plt.xlabel('Langue')
    plt.ylabel('Confiance moyenne')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('/content/results/figures/language_confidence.png')

    # 2. Corrélation entre confiance et précision
    logger.info("Analyse de la corrélation entre confiance et précision...")
    precision_by_lang = {}
    for lang in languages:
        lang_indices = [i for i, l in enumerate(true_langs) if l == lang]
        if not lang_indices:
            continue
        correct = sum(1 for i in lang_indices if pred_langs[i] == true_langs[i])
        precision_by_lang[lang] = correct / len(lang_indices)

    # Visualisation de la corrélation entre confiance et précision
    plt.figure(figsize=(10, 8))
    langs = list(avg_confidence_by_lang.keys())
    conf_values = [avg_confidence_by_lang[l] for l in langs]
    prec_values = [precision_by_lang[l] for l in langs]

    plt.scatter(conf_values, prec_values)
    for i, lang in enumerate(langs):
        plt.annotate(lang, (conf_values[i], prec_values[i]))

    # Calculer et afficher la corrélation
    from scipy.stats import pearsonr, spearmanr
    correlation, p_value = pearsonr(conf_values, prec_values)
    plt.title(f'Corrélation entre confiance et précision (r={correlation:.3f}, p={p_value:.3f})')
    plt.xlabel('Confiance moyenne')
    plt.ylabel('Précision')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/content/results/figures/confidence_precision_correlation.png')
    logger.info(f"Corrélation entre confiance et précision: r={correlation:.3f}, p={p_value:.3f}")

    # 3. Indice de confusion intralinguistique vs interlinguistique
    logger.info("Calcul de l'indice de confusion intralinguistique vs interlinguistique...")
    language_groups = {
        'Langues slaves orientales': ['ru', 'uk', 'be', 'rue'],
        'Langues slaves méridionales': ['bg', 'mk', 'sr'],
        'Langues turciques': ['tt', 'ba', 'kk', 'ky'],
        'Langues finno-ougriennes': ['koi', 'kv', 'udm', 'mhr', 'myv'],
        'Langues caucasiennes': ['ab', 'kbd', 'ce'],
        'Autres langues': ['bxr', 'cv', 'mn', 'os', 'sah', 'tg', 'tyv']
    }

    # Fonction pour trouver le groupe d'une langue
    def get_language_group(lang):
        for group, langs in language_groups.items():
            if lang in langs:
                return group
        return None  # Pour les langues mixtes ou autres

    # Compter les erreurs intra et inter-groupes
    intra_group_errors = 0
    inter_group_errors = 0

    for i in range(len(true_langs)):
        if true_langs[i] != pred_langs[i]:  # Si c'est une erreur
            true_group = get_language_group(true_langs[i])
            pred_group = get_language_group(pred_langs[i])

            # Ignorer les langues mixtes ou non classifiées
            if true_group is None or pred_group is None:
                continue

            if true_group == pred_group:
                intra_group_errors += 1
            else:
                inter_group_errors += 1

    total_errors = intra_group_errors + inter_group_errors
    intra_ratio = intra_group_errors / total_errors if total_errors > 0 else 0

    logger.info(f"Erreurs intra-groupe: {intra_group_errors}")
    logger.info(f"Erreurs inter-groupe: {inter_group_errors}")
    logger.info(f"Ratio intra/total: {intra_ratio:.3f}")

    # Visualisation
    plt.figure(figsize=(8, 6))
    plt.bar(['Intra-groupe', 'Inter-groupe'], [intra_group_errors, inter_group_errors])
    plt.title('Distribution des erreurs par type de confusion')
    plt.ylabel('Nombre d\'erreurs')
    plt.tight_layout()
    plt.savefig('/content/results/figures/error_distribution.png')

    # 4. Robustesse à la longueur du texte
    logger.info("Analyse de la robustesse à la longueur du texte...")
    if 'token_count' in test_df.columns:
        # Définir des plages de longueur
        length_ranges = [(0, 50), (51, 100), (101, 200), (201, 500), (501, float('inf'))]
        range_labels = ['0-50', '51-100', '101-200', '201-500', '500+']

        # Calculer la précision pour chaque plage
        precision_by_length = []
        count_by_length = []

        for min_len, max_len in length_ranges:
            # Indices des textes dans cette plage de longueur
            indices = [i for i, count in enumerate(test_df['token_count'])
                      if min_len <= count <= max_len]

            count_by_length.append(len(indices))

            if not indices:  # Si aucun exemple dans cette plage
                precision_by_length.append(0)
                continue

            correct = sum(1 for i in indices if pred_langs[i] == true_langs[i])
            precision = correct / len(indices)
            precision_by_length.append(precision)

        # Visualiser les résultats
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range_labels, precision_by_length)

        # Ajouter les nombres d'exemples
        for i, bar in enumerate(bars):
            plt.text(bar.get_x() + bar.get_width()/2.,
                    bar.get_height() + 0.02,
                    f'n={count_by_length[i]}',
                    ha='center')

        plt.title('Précision par longueur de texte')
        plt.xlabel('Nombre de tokens')
        plt.ylabel('Précision')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('/content/results/figures/precision_by_length.png')

    # 5. Corrélation entre fréquence d'une langue et sa performance
    logger.info("Analyse de la corrélation entre fréquence des langues et performance...")
    # Compter les occurrences de chaque langue dans l'ensemble d'entraînement
    lang_frequencies = train_df['language'].value_counts().to_dict()

    # Extraire les F1-scores du rapport de classification
    f1_scores = {lang: report[lang]['f1-score']
                for lang in report
                if lang not in ['accuracy', 'macro avg', 'weighted avg']}

    # Préparer les données pour la corrélation
    common_langs = [l for l in f1_scores if l in lang_frequencies]
    freq_values = [lang_frequencies[l] for l in common_langs]
    f1_values = [f1_scores[l] for l in common_langs]

    # Calculer la corrélation
    freq_f1_corr, freq_f1_pval = pearsonr(freq_values, f1_values)

    # Visualisation
    plt.figure(figsize=(10, 8))
    plt.scatter(freq_values, f1_values)
    for i, lang in enumerate(common_langs):
        plt.annotate(lang, (freq_values[i], f1_values[i]))

    plt.title(f'Corrélation entre fréquence et F1-score (r={freq_f1_corr:.3f}, p={freq_f1_pval:.3f})')
    plt.xlabel('Nombre d\'exemples dans l\'ensemble d\'entraînement')
    plt.ylabel('F1-score')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/content/results/figures/frequency_f1_correlation.png')
    logger.info(f"Corrélation entre fréquence et F1-score: r={freq_f1_corr:.3f}, p={freq_f1_pval:.3f}")

    # 6. Distance linguistique entre prédiction et vérité
    logger.info("Calcul de la distance linguistique entre prédiction et vérité...")
    # Définir une matrice de distance linguistique simplifiée
    # 0 = même langue, 1 = même groupe, 2 = groupes différents
    def linguistic_distance(lang1, lang2):
        if lang1 == lang2:
            return 0

        group1 = get_language_group(lang1)
        group2 = get_language_group(lang2)

        if group1 is None or group2 is None:
            return 2  # Par défaut, distance maximale pour les langues mixtes

        return 1 if group1 == group2 else 2

    # Calculer la distance moyenne des erreurs
    total_distance = 0
    error_count = 0

    for i in range(len(true_langs)):
        if true_langs[i] != pred_langs[i]:  # Si c'est une erreur
            distance = linguistic_distance(true_langs[i], pred_langs[i])
            total_distance += distance
            error_count += 1

    avg_distance = total_distance / error_count if error_count > 0 else 0
    logger.info(f"Distance linguistique moyenne des erreurs: {avg_distance:.3f}")

    # 7. Analyse détaillée par paires de langues
    logger.info("Analyse détaillée des confusions par paires de langues...")

    # Calculer une matrice de confusion normalisée
    from sklearn.metrics import confusion_matrix
    normalized_conf_matrix = confusion_matrix(true_langs, pred_langs,
                                            labels=sorted(test_df['language'].unique()),
                                            normalize='true')  # normalize='true' divise chaque ligne par sa somme

    # Créer un DataFrame pour faciliter l'analyse
    langs = sorted(test_df['language'].unique())
    conf_df = pd.DataFrame(normalized_conf_matrix, index=langs, columns=langs)

    # Identifier les paires de langues avec les plus fortes confusions
    confusion_pairs = []
    for true_lang in langs:
        for pred_lang in langs:
            if true_lang != pred_lang and conf_df.loc[true_lang, pred_lang] > 0:
                confusion_pairs.append((true_lang, pred_lang, conf_df.loc[true_lang, pred_lang]))

    # Trier par niveau de confusion (décroissant)
    confusion_pairs.sort(key=lambda x: x[2], reverse=True)

    # Afficher et enregistrer les 10 paires de langues les plus confondues
    top_n = min(10, len(confusion_pairs))
    logger.info(f"Top {top_n} paires de langues les plus confondues:")
    confusion_data = []

    for true_lang, pred_lang, conf_rate in confusion_pairs[:top_n]:
        true_group = get_language_group(true_lang) or "Mixte/Autre"
        pred_group = get_language_group(pred_lang) or "Mixte/Autre"
        same_group = true_group == pred_group

        logger.info(f"  {true_lang} → {pred_lang}: {conf_rate:.3f} (Même groupe: {same_group})")

        confusion_data.append({
            'Langue réelle': true_lang,
            'Langue prédite': pred_lang,
            'Taux de confusion': conf_rate,
            'Groupe réel': true_group,
            'Groupe prédit': pred_group,
            'Même groupe': same_group
        })

    # Créer un DataFrame et le sauvegarder
    confusion_df = pd.DataFrame(confusion_data)
    confusion_df.to_csv('/content/results/metrics/top_confusion_pairs.csv', index=False)

    # Visualisation des top paires confondues
    plt.figure(figsize=(12, 8))
    bar_colors = ['orange' if row['Même groupe'] else 'blue' for _, row in confusion_df.iterrows()]
    plt.bar(range(len(confusion_df)), confusion_df['Taux de confusion'], color=bar_colors)
    plt.xticks(range(len(confusion_df)),
              [f"{row['Langue réelle']} → {row['Langue prédite']}" for _, row in confusion_df.iterrows()],
              rotation=45, ha='right')
    plt.title(f'Top {top_n} paires de langues les plus confondues')
    plt.ylabel('Taux de confusion')
    plt.legend(['Même groupe linguistique', 'Groupes différents'],
              handles=[plt.Rectangle((0,0),1,1, color=c) for c in ['orange', 'blue']])
    plt.tight_layout()
    plt.savefig('/content/results/figures/top_confusion_pairs.png')
    plt.close()

    # Analyse de la confusion par groupe linguistique
    logger.info("Analyse de la confusion par groupe linguistique...")
    group_confusion = {}

    # Initialiser la matrice de confusion entre groupes
    group_names = sorted(set(group for group in language_groups.keys()))
    for g1 in group_names:
        group_confusion[g1] = {g2: 0 for g2 in group_names}

    # Remplir la matrice de confusion entre groupes
    for i in range(len(true_langs)):
        true_group = get_language_group(true_langs[i])
        pred_group = get_language_group(pred_langs[i])

        # Ignorer les langues mixtes ou non classifiées
        if true_group is None or pred_group is None:
            continue

        # Incrémenter le compteur de confusion
        group_confusion[true_group][pred_group] += 1

    # Convertir en DataFrame pour visualisation
    group_conf_df = pd.DataFrame(group_confusion).fillna(0)

    # Normaliser par ligne
    for group in group_conf_df.index:
        row_sum = group_conf_df.loc[group].sum()
        if row_sum > 0:
            group_conf_df.loc[group] = group_conf_df.loc[group] / row_sum

    # Visualiser la matrice de confusion entre groupes
    plt.figure(figsize=(10, 8))
    sns.heatmap(group_conf_df, annot=True, fmt='.2f', cmap='Blues')
    plt.title('Matrice de confusion entre groupes linguistiques')
    plt.xlabel('Groupe prédit')
    plt.ylabel('Groupe réel')
    plt.tight_layout()
    plt.savefig('/content/results/figures/group_confusion_matrix.png')
    plt.close()

    # Calculer la précision par groupe
    precision_by_group = {}
    for group in group_names:
        true_indices = [i for i, lang in enumerate(true_langs)
                        if get_language_group(lang) == group]

        if not true_indices:
            continue

        correct = sum(1 for i in true_indices
                    if get_language_group(pred_langs[i]) == group)
        precision_by_group[group] = correct / len(true_indices)

    # Visualiser la précision par groupe
    plt.figure(figsize=(12, 6))
    groups = list(precision_by_group.keys())
    precisions = [precision_by_group[g] for g in groups]

    # Trier par précision
    sorted_indices = sorted(range(len(precisions)), key=lambda i: precisions[i], reverse=True)
    sorted_groups = [groups[i] for i in sorted_indices]
    sorted_precisions = [precisions[i] for i in sorted_indices]

    plt.bar(sorted_groups, sorted_precisions)
    plt.title('Précision de classification par groupe linguistique')
    plt.xlabel('Groupe linguistique')
    plt.ylabel('Précision')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/content/results/figures/group_precision.png')
    plt.close()

    # Afficher les résultats
    logger.info(f"Exactitude globale : {report['accuracy']:.4f}")
    logger.info("Top 5 langues les mieux classées:")
    for lang, values in sorted([(l, v['f1-score']) for l, v in report.items() if l not in ['accuracy', 'macro avg', 'weighted avg']], key=lambda x: x[1], reverse=True)[:5]:
        logger.info(f"  {lang}: {values:.4f}")

    # Compresser les résultats pour faciliter le téléchargement
    logger.info("Compression des résultats...")
    !zip -r /content/results.zip /content/results
    logger.info("Résultats compressés dans /content/results.zip")
    logger.info("N'oubliez pas de télécharger ce fichier avant de fermer la session!")

    # Générer un lien de téléchargement
    from google.colab import files
    files.download('/content/results.zip')

except Exception as e:
    logger.error(f"Une erreur s'est produite: {str(e)}", exc_info=True)
    raise