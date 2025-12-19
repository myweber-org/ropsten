
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
    return re.sub(pattern, '', text)import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing=False, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows.
        fill_missing (bool): Whether to fill missing values.
        fill_value: Value to use for filling missing data.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    return True, "DataFrame is valid"

def calculate_statistics(df):
    """
    Calculate basic statistics for numeric columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
    
    Returns:
        dict: Dictionary containing statistics for each numeric column.
    """
    numeric_cols = df.select_dtypes(include=['number']).columns
    stats = {}
    
    for col in numeric_cols:
        stats[col] = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max()
        }
    
    return stats