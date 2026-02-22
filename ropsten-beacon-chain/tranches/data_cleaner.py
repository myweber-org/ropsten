
import re

def clean_string(text):
    """
    Cleans a string by:
    - Stripping leading/trailing whitespace
    - Replacing multiple spaces/newlines/tabs with a single space
    - Converting to lowercase
    """
    if not isinstance(text, str):
        return text
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text.lower()