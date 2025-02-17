import ast

import pandas as pd

# Convert genres column from string dictionary to list of values
def parse_genres(genre_str):
    try:
        genre_dict = ast.literal_eval(genre_str)  # Convert string to dictionary
        if isinstance(genre_dict, dict):
            return list(genre_dict.values())  # Extract genre names as a list
        return []
    except (ValueError, SyntaxError):
        return []  # Return empty list if parsing fails


def get_plot_of_movies():
    #clean unneeded metadata
    column_names = ['Wikipedia_movie_ID', 'Freebase_movie_ID', 'movie_title', 'release_date', 'revenue', 'runtime', 'language', 'countries', 'genres']
    column_to_remove = ['Freebase_movie_ID', 'release_date', 'revenue', 'runtime', 'language', 'countries']


    with open('../../input/movie.metadata.tsv') as f:
        df = pd.read_csv(f, sep='\t', header=None, names=column_names)
        df_meta_clean = df.drop(column_to_remove, axis=1)
        df_meta_clean["genres"] = df_meta_clean["genres"].apply(parse_genres)
        plot_dict = {}
        plot_summaries_path = "../../input/plot_summaries.txt"
        with open(plot_summaries_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split("\t", 1)
                if (len(parts) == 2):
                    movie_id, plot = parts
                    plot_dict[movie_id] = plot
            df_plot = pd.DataFrame(list(plot_dict.items()), columns=['Wikipedia_movie_ID', 'plot'])

        df_meta_clean["Wikipedia_movie_ID"] = df_meta_clean["Wikipedia_movie_ID"].astype(str)
        df_plot["Wikipedia_movie_ID"] = df_plot["Wikipedia_movie_ID"].astype(str)
        # merge two temp_dataset
        df_combined = pd.merge(df_meta_clean, df_plot, on='Wikipedia_movie_ID', how="inner")
        return df_combined



