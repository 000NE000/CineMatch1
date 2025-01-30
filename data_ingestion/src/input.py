import json
import logging
import os

def get_url_of_movies(json_file='movie_brief_info.json'):
    """
    Loads movie data from a JSON file.
    Each movie should have 'title' and 'url'.
    """
    if not os.path.exists(json_file):
        logging.error(f"Input file {json_file} does not exist.")
        return []
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            movies_dict = json.load(f)
            movies_title = [item["title"] for item in movies_dict]
            movies_url = [item["url"] for item in movies_dict]
            
            
        logging.info(f"Loaded {len(movies)} movies from {json_file}.")
        return movies
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {json_file}: {e}")
        return []
    except Exception as e:
        logging.error(f"Unexpected error loading {json_file}: {e}")
        return []
