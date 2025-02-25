import pandas as pd
from pymongo import MongoClient
from config.config import MONGODB_CONFIG

def get_mongo_collection(collection_name):
    client = MongoClient(MONGODB_CONFIG["uri"])
    db = client[MONGODB_CONFIG["db"]]
    return db[collection_name]

def store_mongoDB(df_combined, collection_name):
    # connect to MongoDB
    client = MongoClient(MONGODB_CONFIG["uri"])
    db = client[collection_name]
    collection = db[collection_name]
    # Convert dataframe to dictionary format for MongoDB
    movie_records = df_combined.to_dict(orient="records")
    # Insert data into MongoDB
    collection.insert_many(movie_records)
    print(collection.count_documents({}))

# Example usage
if __name__ == "__main__":
    collection = get_mongo_collection("trigger_taxonomy")
    df = pd.read_json('../data/input/plot_data/trigger_taxonomy.json')
    store_mongoDB(df, "trigger_taxonomy")