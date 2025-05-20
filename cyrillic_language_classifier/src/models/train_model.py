import os
import json
import torch
import evaluate
import pandas as pd
import numpy as np
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset


# Désactiver la parallélisation des tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configurer la journalisation
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Créer les répertoires nécessaires
os.makedirs('results/models/language-detection', exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('results/metrics', exist_ok=True)

logger.info("Début du script d'entraînement avec modèle pré-entraîné pour la détection de langues")

try:
    # Charger les données
    logger.info("Chargement des données d'entraînement...")
    train_df = pd.read_csv('data/final/train/train_corpus.csv')
    logger.info(f"Données d'entraînement chargées : {len(train_df)} exemples")
    
    logger.info("Chargement des données de validation...")
    val_df = pd.read_csv('data/final/validation/validation_corpus.csv')
    logger.info(f"Données de validation chargées : {len(val_df)} exemples")
    
    # Préparer les mappages
    logger.info("Préparation des mappages langue -> ID...")
    languages = sorted(train_df['language'].unique())
    lang_to_id = {lang: idx for idx, lang in enumerate(languages)}
    id_to_lang = {idx: lang for lang, idx in lang_to_id.items()}
    logger.info(f"Nombre de langues identifiées : {len(languages)}")
    
    # Sauvegarder les mappages
    with open('results/metrics/lang_mappings.json', 'w') as f:
        json.dump({
            'lang_to_id': lang_to_id,
            'id_to_lang': {str(k): v for k, v in id_to_lang.items()}
        }, f)
    logger.info("Mappages sauvegardés dans results/metrics/lang_mappings.json")
    
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
    
    # Fonction de calcul des métriques
    def compute_metrics(pred):
        logger.info("Calcul des métriques d'évaluation...")
        metric = evaluate.load("accuracy")
        logits, labels = pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    
    # Définir les arguments d'entraînement avec des paramètres plus légers
    logger.info("Configuration des arguments d'entraînement...")
    training_args = TrainingArguments(
        output_dir='results/models/language-detection',
        num_train_epochs=3,  # Réduit de 5 à 3
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        warmup_steps=100,  # Réduit de 500 à 100
        weight_decay=0.01,
        logging_dir='logs',
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        gradient_accumulation_steps=4,
        fp16=False,  # Désactiver la précision mixte si problématique
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
    logger.info("Début de l'entraînement (peut prendre plusieurs heures)...")
    trainer.train()
    logger.info("Entraînement terminé")
    
    # Sauvegarder le modèle final
    logger.info("Sauvegarde du modèle final...")
    trainer.save_model('results/models/language-detection-final')
    logger.info("Modèle sauvegardé dans results/models/language-detection-final")
    
    logger.info("Script d'entraînement terminé avec succès")

except Exception as e:
    logger.error(f"Une erreur s'est produite: {str(e)}", exc_info=True)
    raise