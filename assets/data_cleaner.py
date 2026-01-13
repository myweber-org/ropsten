
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd
import re

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    column_mapping (dict): Optional dictionary to rename columns.
    drop_duplicates (bool): Whether to remove duplicate rows.
    normalize_text (bool): Whether to normalize text columns (strip, lower case).
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    if column_mapping:
        df_clean = df_clean.rename(columns=column_mapping)
    
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    
    if normalize_text:
        for col in df_clean.select_dtypes(include=['object']).columns:
            df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()
    
    return df_clean

def remove_special_characters(text, keep_pattern=r'[^a-zA-Z0-9\s]'):
    """
    Remove special characters from a string.
    
    Parameters:
    text (str): Input text.
    keep_pattern (str): Regex pattern of characters to remove.
    
    Returns:
    str: Cleaned text.
    """
    if pd.isna(text):
        return text
    return re.sub(keep_pattern, '', str(text))

def validate_email(email):
    """
    Validate email format using regex.
    
    Parameters:
    email (str): Email address to validate.
    
    Returns:
    bool: True if email format is valid.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, str(email)))