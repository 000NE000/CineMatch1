import os
from dotenv import load_dotenv

load_dotenv()

def get_api_key(key_name):
    """Retrieves API key from environment variables."""
    return os.getenv(key_name)

if __name__ == "__main__":
    print(get_api_key("OMDB_API_KEY"))