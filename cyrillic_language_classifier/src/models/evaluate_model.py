from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import json
from transformers import Trainer, AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset


def prepare_test_dataset(test_df, tokenizer, lang_to_id, max_length=128):
    """
    Prépare le dataset de test pour l'évaluation.
    
    Args:
        test_df: DataFrame contenant les données de test
        tokenizer: Tokenizer à utiliser
        lang_to_id: Dictionnaire de mapping langue -> id
        max_length: Longueur maximale des séquences
        
    Returns:
        Dataset prêt pour l'évaluation
    """
    texts = test_df['text'].tolist()
    labels = [lang_to_id[lang] for lang in test_df['language']]
    
    encodings = tokenizer(
        texts, 
        truncation=True, 
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    dataset = Dataset.from_dict({
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': labels
    })
    
    return dataset


def evaluate_detailed(model_path, test_df_path, output_dir='results'):
    """
    Évalue en détail le modèle entraîné sur l'ensemble de test.
    
    Args:
        model_path: Chemin vers le modèle entraîné
        test_df_path: Chemin vers le fichier CSV de test
        output_dir: Répertoire de sortie pour les résultats
    """
    # Créer les répertoires de sortie
    os.makedirs(f'{output_dir}/metrics', exist_ok=True)
    os.makedirs(f'{output_dir}/figures', exist_ok=True)
    
    # Charger les mappings de langues
    with open(f'{output_dir}/metrics/lang_mappings.json', 'r') as f:
        mappings = json.load(f)
        lang_to_id = mappings['lang_to_id']
        id_to_lang = {int(k): v for k, v in mappings['id_to_lang'].items()}  # Convertir les clés string en int
    
    # Charger le modèle et le tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Charger les données de test
    test_df = pd.read_csv(test_df_path)
    
    # Préparer le dataset de test
    test_dataset = prepare_test_dataset(test_df, tokenizer, lang_to_id)
    
    # Initialiser le Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer
    )
    
    # Obtenir les prédictions
    outputs = trainer.predict(test_dataset)
    predictions = np.argmax(outputs.predictions, axis=-1)
    
    # Convertir les prédictions en noms de langues
    pred_langs = [id_to_lang[pred] for pred in predictions]
    true_langs = test_df['language'].tolist()
    
    # Rapport de classification
    report = classification_report(true_langs, pred_langs, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    # Sauvegarder le rapport
    report_df.to_csv(f'{output_dir}/metrics/classification_report.csv')
    
    # Matrice de confusion
    plt.figure(figsize=(15, 12))
    conf_matrix = confusion_matrix(true_langs, pred_langs, labels=sorted(test_df['language'].unique()))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
               xticklabels=sorted(test_df['language'].unique()),
               yticklabels=sorted(test_df['language'].unique()))
    plt.title('Matrice de confusion par langue')
    plt.xlabel('Langue prédite')
    plt.ylabel('Langue réelle')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/figures/confusion_matrix.png')
    
    # Performance par type d'augmentation (si applicable)
    if 'source' in test_df.columns:
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
        plt.savefig(f'{output_dir}/figures/augmentation_performance.png')
        
    # Performance par groupe de langues
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
    for group, langs in language_groups.items():
        # Indices des langues appartenant à ce groupe
        group_indices = test_df[test_df['language'].isin(langs)].index
        
        if len(group_indices) > 0:
            group_true = [true_langs[i] for i in group_indices]
            group_pred = [pred_langs[i] for i in group_indices]
            group_acc = sum(t == p for t, p in zip(group_true, group_pred)) / len(group_true)
            group_performances[group] = group_acc
    
    # Visualiser les performances par groupe
    plt.figure(figsize=(12, 6))
    plt.bar(group_performances.keys(), group_performances.values())
    plt.title('Précision par groupe linguistique')
    plt.ylabel('Précision')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/figures/language_group_performance.png')
    
    # Performance sur les langues mixtes
    if any('_mix' in lang for lang in test_df['language'].unique()):
        # Séparer les langues normales et les langues mixtes
        mixed_indices = test_df[test_df['language'].str.contains('_mix')].index
        normal_indices = test_df[~test_df['language'].str.contains('_mix')].index
        
        if len(mixed_indices) > 0 and len(normal_indices) > 0:
            # Calculer les performances
            mixed_true = [true_langs[i] for i in mixed_indices]
            mixed_pred = [pred_langs[i] for i in mixed_indices]
            mixed_acc = sum(t == p for t, p in zip(mixed_true, mixed_pred)) / len(mixed_true)
            
            normal_true = [true_langs[i] for i in normal_indices]
            normal_pred = [pred_langs[i] for i in normal_indices]
            normal_acc = sum(t == p for t, p in zip(normal_true, normal_pred)) / len(normal_true)
            
            # Visualiser la comparaison
            plt.figure(figsize=(8, 6))
            plt.bar(['Langues normales', 'Langues mixtes'], [normal_acc, mixed_acc])
            plt.title('Précision: Langues normales vs. Langues mixtes')
            plt.ylabel('Précision')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/figures/mixed_language_performance.png')
    
    # Générer un rapport textuel résumé
    with open(f'{output_dir}/metrics/evaluation_summary.txt', 'w') as f:
        f.write('# Résumé de l\'évaluation du modèle de classification de langues cyrilliques\n\n')
        
        # Précision globale
        overall_acc = report['accuracy']
        f.write(f'Précision globale: {overall_acc:.4f}\n\n')
        
        # Top 5 meilleures langues
        f.write('## Top 5 des langues les mieux classées\n')
        top_langs = [(lang, report[lang]['f1-score']) 
                    for lang in test_df['language'].unique() 
                    if lang in report and lang not in ['accuracy', 'macro avg', 'weighted avg']]
        top_langs.sort(key=lambda x: x[1], reverse=True)
        
        for lang, score in top_langs[:5]:
            f.write(f'- {lang}: F1-score = {score:.4f}\n')
        
        # 5 langues les moins bien classées
        f.write('\n## 5 langues les moins bien classées\n')
        for lang, score in top_langs[-5:]:
            f.write(f'- {lang}: F1-score = {score:.4f}\n')
        
        # Performances par groupe
        f.write('\n## Performance par groupe linguistique\n')
        for group, score in group_performances.items():
            f.write(f'- {group}: {score:.4f}\n')
        
        # Performance par méthode d'augmentation
        if 'source' in test_df.columns:
            f.write('\n## Performance par méthode d\'augmentation\n')
            for source, score in aug_performance.items():
                f.write(f'- {source}: {score:.4f}\n')
    
    return report


# Si le script est exécuté directement
if __name__ == "__main__":
    # Définir les chemins
    model_path = "results/models/xlm-roberta-cyrillic"
    test_df_path = "data/final/test/test_corpus.csv"
    
    # Évaluer le modèle
    evaluate_detailed(model_path, test_df_path)