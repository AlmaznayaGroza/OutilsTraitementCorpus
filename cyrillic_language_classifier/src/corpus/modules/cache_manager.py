# src/corpus/cache_manager.py

# Création des ensembles globaux qui seront partagés entre les modules
too_short_article_ids = set()
collected_article_ids = set()

def mark_as_too_short(article_id):
    """Marque un article comme étant trop court."""
    too_short_article_ids.add(article_id)
    
def is_too_short(article_id):
    """Vérifie si un article est déjà marqué comme trop court."""
    return article_id in too_short_article_ids