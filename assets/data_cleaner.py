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
    print(f"Tokens: {tokenize_text(cleaned)}")import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(file_path):
    df = pd.read_csv(file_path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        df = remove_outliers_iqr(df, col)
        df = normalize_minmax(df, col)
    
    return df

if __name__ == "__main__":
    cleaned_data = clean_dataset('sample_data.csv')
    cleaned_data.to_csv('cleaned_data.csv', index=False)
    print("Data cleaning completed. Saved to cleaned_data.csv")