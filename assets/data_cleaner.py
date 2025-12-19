
import re

def clean_string(text):
    if not isinstance(text, str):
        return text
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Remove non-printable characters except newline and tab
    text = ''.join(char for char in text if char.isprintable() or char in '\n\t')
    
    return text

def normalize_whitespace(text):
    if not isinstance(text, str):
        return text
    
    # Replace multiple spaces with single space
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Normalize newlines to Unix style
    text = re.sub(r'\r\n|\r', '\n', text)
    
    # Remove trailing whitespace from each line
    lines = [line.rstrip() for line in text.split('\n')]
    
    return '\n'.join(lines)

def remove_special_characters(text, keep_chars=''):
    if not isinstance(text, str):
        return text
    
    # Keep alphanumeric, whitespace, and specified characters
    pattern = f'[^a-zA-Z0-9\\s{re.escape(keep_chars)}]'
    return re.sub(pattern, '', text)