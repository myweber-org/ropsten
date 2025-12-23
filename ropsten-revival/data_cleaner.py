
import pandas as pd
import re

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df: pandas DataFrame to clean
        column_mapping: Optional dictionary to rename columns
        drop_duplicates: Whether to remove duplicate rows
        normalize_text: Whether to normalize text columns (strip, lower case)
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    
    if normalize_text:
        for column in cleaned_df.select_dtypes(include=['object']).columns:
            cleaned_df[column] = cleaned_df[column].astype(str).str.strip().str.lower()
    
    return cleaned_df

def validate_email(email):
    """
    Validate email format using regex.
    
    Args:
        email: Email string to validate
    
    Returns:
        Boolean indicating if email is valid
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, str(email)))

def remove_special_characters(text, keep_chars=''):
    """
    Remove special characters from text, keeping only alphanumeric and specified characters.
    
    Args:
        text: Input text
        keep_chars: Additional characters to keep (e.g., spaces, punctuation)
    
    Returns:
        Cleaned text
    """
    pattern = f'[^a-zA-Z0-9{re.escape(keep_chars)}]'
    return re.sub(pattern, '', str(text))