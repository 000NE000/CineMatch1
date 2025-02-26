import pandas as pd
import re
import unicodedata
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download("punkt")
nltk.download('stopwords')

def clean_text(text):
    text = re.sub(r"\s+", " ", text)  # Normalize whitespace
    text = re.sub(r"[^a-zA-Z0-9.,'?! ]", "", text)  # Remove special characters
    text = text.strip()
    return text



# Contractions dictionary (customizable)
CONTRACTIONS = {
    "isn't": "is not", "can't": "cannot", "won't": "will not",
    "I'm": "I am", "it's": "it is", "he's": "he is", "she's": "she is",
    "they're": "they are", "we're": "we are", "you're": "you are"
}


def clean_text_nltk(text):
    """Performs text cleaning using NLTK for fine-tuning T5/BART"""

    text = unicodedata.normalize("NFKC", text) # Normalize unicode characters

    for contraction, full_form in CONTRACTIONS.items():  # Expand contractions
        text = text.replace(contraction, full_form)

    # Remove multiple spaces and strip text
    text = re.sub(r'\s+', ' ', text).strip()

    # Normalize punctuation (removing excessive special characters, but keeping .,!?-)
    text = re.sub(r'[^a-zA-Z0-9.,!?\'"\s-]', '', text)

    # Tokenization
    tokens = word_tokenize(text)

    # Reconstruct cleaned text
    cleaned_text = " ".join(tokens)

    return cleaned_text


# # Example usage
# plot = "A woman gives up her dreams to take care of her ailing husband.   "
# cleaned_plot = clean_text_nltk(plot)
# print(cleaned_plot)






