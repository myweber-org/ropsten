
import re
import string

def clean_text(text):
    """
    Clean and normalize a given text string.
    Removes extra whitespace, converts to lowercase, and removes punctuation.
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def tokenize_text(text):
    """
    Tokenize a cleaned text string into a list of words.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []
    return cleaned.split()

def remove_stopwords(word_list, stopwords=None):
    """
    Remove stopwords from a list of tokens.
    Uses a default list if none is provided.
    """
    if stopwords is None:
        stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}

    return [word for word in word_list if word not in stopwords]