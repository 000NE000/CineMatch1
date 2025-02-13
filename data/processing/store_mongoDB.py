from pymongo import MongoClient
from config.config import MONGODB_CONFIG

def store_mongoDB(df_combined):
    # connect to MongoDB
    client = MongoClient(MONGODB_CONFIG["uri"])
    db = client["movies"]
    collection = db["movies"]
    # Convert dataframe to dictionary format for MongoDB
    movie_records = df_combined.to_dict(orient="records")
    # Insert data into MongoDB
    collection.insert_many(movie_records)
    print(collection.count_documents({}))

