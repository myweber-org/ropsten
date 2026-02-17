
import pandas as pd
import re

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    column_mapping (dict): Optional dictionary to rename columns
    drop_duplicates (bool): Whether to remove duplicate rows
    normalize_text (bool): Whether to normalize text columns
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Rename columns if mapping provided
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    # Remove duplicate rows
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    
    # Normalize text columns
    if normalize_text:
        for col in cleaned_df.select_dtypes(include=['object']).columns:
            cleaned_df[col] = cleaned_df[col].apply(
                lambda x: re.sub(r'\s+', ' ', str(x).strip().lower()) if pd.notna(x) else x
            )
    
    return cleaned_df

def validate_email(email):
    """
    Validate email format using regex.
    
    Parameters:
    email (str): Email address to validate
    
    Returns:
    bool: True if email is valid, False otherwise
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, str(email))) if pd.notna(email) else False

def filter_valid_emails(df, email_column):
    """
    Filter DataFrame to only include rows with valid email addresses.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    email_column (str): Name of the column containing email addresses
    
    Returns:
    pd.DataFrame: DataFrame with only valid emails
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    mask = df[email_column].apply(validate_email)
    return df[mask].reset_index(drop=True)