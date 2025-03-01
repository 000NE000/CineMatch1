import os

POSTGRES_CONFIG = {
    "dbname": "movieanalyticsdb",
    "user": "cinematch1_admin",
    "password": os.getenv("POSTGRES_PASSWORD", "5891"),
    "host": "localhost",
    "port": 5432
}

MONGODB_CONFIG = {
    "uri": "mongodb://cinematch1_admin:5891@localhost:27017/",
    "db": "MovieNarrativeDB"
}

GPT_CONFIG = {
    "api_key": os.getenv('OPENAI_API_KEY')
}