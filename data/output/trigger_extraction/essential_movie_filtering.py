from data.collection.raw_plot_collection import get_plot_of_movies
from pymongo import MongoClient
import wikipediaapi
import json

client = MongoClient('mongodb://localhost:27017/')
db = client['movies']  # 사용할 데이터베이스 이름
collection = db['movies']  # 사용할 컬렉션 이름
essential_movie_list = movie_titles = [
    "Casablanca (1942)",
    "Gone with the Wind (1939)",
    "The Godfather (1972)",
    "Star Wars: Episode IV – A New Hope (1977)",
    "The Wizard of Oz (1939)",
    "Schindler's List (1993)",
    "Forrest Gump (1994)",
    "Titanic (1997)",
    "The Shawshank Redemption (1994)",
    "The Lion King (1994)",
    "Fight Club (1999)",
    "The Matrix (1999)",
    "Braveheart (1995)",
    "Jurassic Park (1993)",
    "Indiana Jones and the Raiders of the Lost Ark (1981)",
    "Back to the Future (1985)",
    "E.T. the Extra-Terrestrial (1982)",
    "The Terminator (1984)",
    "Aliens (1986)",
    "Die Hard (1988)",
    "Saving Private Ryan (1998)",
    "The Green Mile (1999)",
    "Good Will Hunting (1997)",
    "A Beautiful Mind (2001)",
    "Memento (2000)",
    "Requiem for a Dream (2000)",
    "American Beauty (1999)",
    "The Sixth Sense (1999)",
    "Pulp Fiction (1994)",
    "Reservoir Dogs (1992)",
    "The Usual Suspects (1995)",
    "Se7en (1995)",
    "L.A. Confidential (1997)",
    "Heat (1995)",
    "The Departed (2006)",
    "No Country for Old Men (2007)",
    "The Prestige (2006)",
    "Mulholland Drive (2001)",
    "City of God (2002)",
    "Amélie (2001)",
    "Spirited Away (2001)",
    "Crouching Tiger, Hidden Dragon (2000)",
    "Gladiator (2000)",
    "The Pianist (2002)",
    "Life is Beautiful (1997)",
    "The Iron Giant (1999)",
    "The Lord of the Rings: The Fellowship of the Ring (2001)",
    "The Lord of the Rings: The Two Towers (2002)",
    "The Lord of the Rings: The Return of the King (2003)",
    "Pirates of the Caribbean: The Curse of the Black Pearl (2003)",
    "The Incredibles (2004)",
    "Finding Nemo (2003)",
    "Shrek (2001)",
    "Lord of the Flies (1963)",
    "One Flew Over the Cuckoo's Nest (1975)",
    "Rocky (1976)",
    "Raging Bull (1980)",
    "Taxi Driver (1976)",
    "Blade Runner (1982)",
    "The Shining (1980)",
    "A Clockwork Orange (1971)",
    "Dr. Strangelove (1964)",
    "2001: A Space Odyssey (1968)",
    "The Good, the Bad and the Ugly (1966)",
    "Once Upon a Time in the West (1968)",
    "The Magnificent Seven (1960)",
    "Ben-Hur (1959)",
    "Lawrence of Arabia (1962)",
    "Million Dollar Baby (2004)",
    "Rocky Balboa (2006)",
    "The Breakfast Club (1985)",
    "Dead Poets Society (1989)",
    "Goodfellas (1990)",
    "Casino (1995)",
    "The Truman Show (1998)",
    "Fargo (1996)",
    "Boogie Nights (1997)",
    "Trainspotting (1996)",
    "The Big Lebowski (1998)",
    "Snatch (2000)",
    "Lock, Stock and Two Smoking Barrels (1998)",
    "Scream (1996)",
    "The Blair Witch Project (1999)",
    "American History X (1998)",
    "Hotel Rwanda (2004)",
    "Erin Brockovich (2000)",
    "The Pursuit of Happyness (2006)",
    "Slumdog Millionaire (2008)",
    "Brokeback Mountain (2005)",
    "Milk (2008)",
    "V for Vendetta (2005)",
    "The Bourne Identity (2002)",
    "The Bourne Supremacy (2004)",
    "The Bourne Ultimatum (2007)",
    "The King's Speech (2010)",
    "Inglourious Basterds (2009)",
    "Avatar (2009)",
    "District 9 (2009)",
    "Pan's Labyrinth (2006)",
    "The Curious Case of Benjamin Button (2008)"
]
#wikipedia movie id
wiki_wiki = wikipediaapi.Wikipedia(
    language='en',  # Specify language ('en' for English)
    user_agent='cinematch1 (tkdwnd@gmail.com)'  # Optional user agent
)

def get_wikipedia_page_id(title):
    page = wiki_wiki.page(title)
    if page.exists():
        return page.pageid  # Return Wikipedia page ID
    else:
        return None  # If the page doesn't exist

# Fetch and store Wikipedia IDs
movie_ids = {}
for title in essential_movie_list:
    page_id = get_wikipedia_page_id(title)
    if page_id:
        movie_ids[title] = page_id

query = {"Wikipedia_movie_ID": {"$in": list(movie_ids.values())}}
data = list(collection.find(query))
data = list(collection.find())

def convert_to_json(data):
    for item in data:
        item['_id'] = str(item['_id'])
    return data

with open('../../data/output/trigger_extraction/essential_plot_data.json', 'w', encoding='utf-8') as f:
    json.dump(convert_to_json(data), f, ensure_ascii=False, indent=4)



