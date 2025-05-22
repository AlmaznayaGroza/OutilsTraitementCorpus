import os
import logging
import pandas as pd
import shutil

def save_articles_to_csv(language_code, articles, output_folder):
    """
    Sauvegarde les articles dans un fichier CSV.
    
    Args:
        language_code: code de la langue
        articles: liste des articles à sauvegarder
        output_folder: dossier de destination
    """
    if not articles:
        return
        
    # Construire le chemin du fichier CSV
    csv_path = f"{output_folder}/{language_code}_articles.csv"
    
    # Créer un DataFrame à partir des articles
    df = pd.DataFrame(articles)
    
    # Sauvegarder dans un fichier CSV
    df.to_csv(csv_path, index=False)
    
    logging.info(f"{len(articles)} articles sauvegardés dans {csv_path}")
    print(f"✓ {len(articles)} articles sauvegardés dans {csv_path}")

def merge_with_existing_data(missing_languages, temp_dir, target_dir):
    """ 
    Fusionne les données collectées avec les données existantes.
    
    Args:
        missing_languages: liste des langues à fusionner
        temp_dir: dossier contenant les nouveaux fichiers
        target_dir: dossier de destination
    """
    logging.info("Fusion des nouvelles données avec les données existantes...")
    print("Fusion des données en cours...")
    
    # S'assurer que le répertoire cible existe
    os.makedirs(target_dir, exist_ok=True)
    
    # Compteurs pour le rapport
    merged_count = 0
    created_count = 0
    error_count = 0
    
    for language in missing_languages:
        temp_file = os.path.join(temp_dir, f"{language}_articles.csv")
        target_file = os.path.join(target_dir, f"{language}_articles.csv")
        
        if not os.path.exists(temp_file):
            logging.warning(f"Pas de nouveau fichier pour {language}")
            continue
            
        # Charger les nouvelles données
        try:
            new_df = pd.read_csv(temp_file)
            logging.info(f"Nouvelles données pour {language}: {len(new_df)} articles")
            
            # Si des données existantes existent déjà, fusionner
            if os.path.exists(target_file):
                try:
                    existing_df = pd.read_csv(target_file)
                    logging.info(f"Données existantes pour {language}: {len(existing_df)} articles")
                    
                    # Fusionner en évitant les doublons (si possible sur la base du texte)
                    if "text" in new_df.columns and "text" in existing_df.columns:
                        combined_df = pd.concat([existing_df, new_df]).drop_duplicates(subset=["text"])
                    else:
                        # Sinon, fusionner simplement et espérer qu'il n'y a pas de doublons
                        combined_df = pd.concat([existing_df, new_df])
                    
                    combined_df.to_csv(target_file, index=False)
                    
                    logging.info(f"Fusion réussie pour {language}: {len(combined_df)} articles au total")
                    print(f"✓ Fusion réussie pour {language}: {len(combined_df)} articles au total")
                    merged_count += 1
                    
                except Exception as e:
                    logging.error(f"Erreur lors de la fusion pour {language}: {e}")
                    logging.info(f"Remplacement du fichier existant par le nouveau")
                    shutil.copy2(temp_file, target_file)
                    print(f"⚠️ Erreur lors de la fusion pour {language} - fichier remplacé")
                    error_count += 1
            else:
                # Si pas de données existantes, copier directement
                shutil.copy2(temp_file, target_file)
                logging.info(f"Nouveau fichier créé pour {language}")
                print(f"✓ Nouveau fichier créé pour {language}: {len(new_df)} articles")
                created_count += 1
                
        except Exception as e:
            logging.error(f"Erreur lors du traitement pour {language}: {e}")
            print(f"❌ Erreur lors du traitement pour {language}")
            error_count += 1
    
    logging.info("Fusion des données terminée")
    print(f"Fusion terminée: {merged_count} fichiers fusionnés, {created_count} fichiers créés, {error_count} erreurs")