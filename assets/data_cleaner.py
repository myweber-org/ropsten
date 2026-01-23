import re
import unicodedata

def clean_text(text, remove_digits=False, keep_punctuation=False):
    """
    Clean and normalize text by:
    1. Converting to lowercase
    2. Removing extra whitespace
    3. Normalizing unicode characters
    4. Optionally removing digits
    5. Optionally removing punctuation
    """
    if not isinstance(text, str):
        return ""

    # Normalize unicode (e.g., convert é to e)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

    # Convert to lowercase
    text = text.lower()

    # Remove digits if specified
    if remove_digits:
        text = re.sub(r'\d+', '', text)

    # Handle punctuation
    if not keep_punctuation:
        # Remove punctuation except for basic sentence boundaries (., !, ?)
        # This pattern keeps periods, exclamation marks, and question marks
        text = re.sub(r'[^\w\s.!?]', '', text)
        # Optionally, collapse multiple punctuation marks
        text = re.sub(r'[.!?]+', '.', text)

    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def tokenize_text(text, tokenizer=None):
    """
    Tokenize text using a simple split or custom tokenizer.
    """
    if tokenizer is None:
        # Simple word tokenizer (splits on whitespace)
        tokens = text.split()
    else:
        tokens = tokenizer(text)
    return tokens

if __name__ == "__main__":
    # Example usage
    sample_text = "Hello, World! This is a TEST. 123 Main St. Let's go!!!"
    cleaned = clean_text(sample_text, remove_digits=True, keep_punctuation=False)
    print(f"Original: {sample_text}")
    print(f"Cleaned: {cleaned}")
    print(f"Tokens: {tokenize_text(cleaned)}")