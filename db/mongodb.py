from pymongo import MongoClient
from config.config import MONGODB_CONFIG

def get_mongo_collection(collection_name):
    client = MongoClient(MONGODB_CONFIG["uri"])
    db = client[MONGODB_CONFIG["db"]]
    return db[collection_name]

# Example usage
if __name__ == "__main__":
    collection = get_mongo_collection("movies")
    sample_movie = collection.find_one({"MovieID": "1234"})
    print(sample_movie)