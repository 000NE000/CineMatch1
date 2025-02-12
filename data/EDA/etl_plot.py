import ast
from config.config import MONGODB_CONFIG
import pandas as pd

#clean unneeded metadata
column_names = ['Wikipedia_movie_ID', 'Freebase_movie_ID', 'movie_title', 'release_date', 'revenue', 'runtime', 'language', 'countries', 'genres']
column_to_remove = ['Freebase_movie_ID', 'release_date', 'revenue', 'runtime', 'language', 'countries']

movie_narrative = []

with open('../input/movie.metadata.tsv') as f:
    df = pd.read_csv(f, sep='\t', header=None, names=column_names)
    df_meta_clean = df.drop(column_to_remove, axis=1)
    dict_df = df_meta_clean.to_dict('records')

    for entry in dict_df:
        if isinstance(entry['genres'], str):
            try:
                genres_dict = ast.literal_eval(entry['genres']) #실제 python object로 변환
                entry['genres'] = list(genres_dict.values())
            except:
                entry['genres'] = []

    plot_dict = {}
    plot_summaries_path = "../input/plot_summaries.txt"
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


from pymongo import MongoClient
#connect to MongoDB
client = MongoClient(MONGODB_CONFIG["uri"])
db = client["movies"]
collection = db["movies"]
# Convert dataframe to dictionary format for MongoDB
movie_records = df_combined.to_dict(orient="records")
# Insert data into MongoDB
collection.insert_many(movie_records)
print(collection.count_documents({}))

