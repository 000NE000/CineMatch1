from data.collection.raw_plot_collection import get_plot_of_movies
from data.processing.preprocessing.clean import clean_text
from data.processing.preprocessing.sentence_segmentation import segment_text
import pandas as pd
import nltk

from db.mongodb import store_mongoDB

nltk.download('punkt')
nltk.download('punkt_tab')

#1st. clean
df_plot = get_plot_of_movies()
df_plot["plot"] = df_plot["plot"].apply(clean_text)

#2nd.sentence segmentation
df_plot["segmented_sentences"] = df_plot["plot"].apply(segment_text)

#3th. Named Entity Recognition (NER)


#final. store to MongoDB
#store_mongoDB(df_plot, "movies")

