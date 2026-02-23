import pandas as pd
import re

def clean_dataframe(df, text_columns=None):
    """
    Clean a DataFrame by removing duplicates and standardizing text columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        text_columns (list, optional): List of column names to standardize text.
            If None, all object dtype columns are processed.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Remove duplicate rows
    df_clean = df.drop_duplicates().reset_index(drop=True)
    
    # Identify text columns if not specified
    if text_columns is None:
        text_columns = df_clean.select_dtypes(include=['object']).columns.tolist()
    
    # Standardize text in specified columns
    for col in text_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(_standardize_text)
    
    return df_clean

def _standardize_text(text):
    """
    Standardize text by converting to lowercase, removing extra whitespace,
    and stripping special characters.
    
    Args:
        text: Input text (can be any type).
    
    Returns:
        str: Standardized text or original value if not a string.
    """
    if not isinstance(text, str):
        return text
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove special characters (keep alphanumeric and basic punctuation)
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    
    return text

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        email_column (str): Name of the column containing email addresses.
    
    Returns:
        pd.DataFrame: DataFrame with additional 'email_valid' boolean column.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    df['email_valid'] = df[email_column].apply(
        lambda x: bool(re.match(email_pattern, str(x))) if pd.notnull(x) else False
    )
    
    return df