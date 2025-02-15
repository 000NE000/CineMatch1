import ast

import pandas as pd

def get_plot_of_movies():
    #clean unneeded metadata
    column_names = ['Wikipedia_movie_ID', 'Freebase_movie_ID', 'movie_title', 'release_date', 'revenue', 'runtime', 'language', 'countries', 'genres']
    column_to_remove = ['Freebase_movie_ID', 'release_date', 'revenue', 'runtime', 'language', 'countries']


    with open('../../input/movie.metadata.tsv') as f:
        df = pd.read_csv(f, sep='\t', header=None, names=column_names)
        df_meta_clean = df.drop(column_to_remove, axis=1)
        dict_df = df_meta_clean.to_dict('records')

        dict_df['genres'] = dict_df['genres'].apply(extract_genre_values)

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

def extract_genre_values(genre_data):
    if isinstance(genre_data, list):  # Check if it's a list
        return [val for d in genre_data if isinstance(d, dict) for val in d.values()]
    elif isinstance(genre_data, dict):  # Handle single dictionary case
        return list(genre_data.values())
    return []