
import pandas as pd
import re

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        column_mapping (dict, optional): Dictionary mapping old column names to new ones
        drop_duplicates (bool): Whether to remove duplicate rows
        normalize_text (bool): Whether to normalize text columns (strip, lower case)
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    
    if normalize_text:
        for col in cleaned_df.select_dtypes(include=['object']).columns:
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip().str.lower()
    
    return cleaned_df

def remove_special_characters(text, keep_pattern=r'[a-zA-Z0-9\s]'):
    """
    Remove special characters from text, keeping only alphanumeric and spaces by default.
    
    Args:
        text (str): Input text
        keep_pattern (str): Regex pattern of characters to keep
    
    Returns:
        str: Cleaned text
    """
    if pd.isna(text):
        return text
    return re.sub(f'[^{keep_pattern}]', '', str(text))

def validate_email(email):
    """
    Validate email format using regex pattern.
    
    Args:
        email (str): Email address to validate
    
    Returns:
        bool: True if email format is valid
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, str(email))) if pd.notna(email) else False

def standardize_dates(df, date_columns, target_format='%Y-%m-%d'):
    """
    Standardize date columns to a consistent format.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        date_columns (list): List of column names containing dates
        target_format (str): Target date format string
    
    Returns:
        pd.DataFrame: DataFrame with standardized dates
    """
    df_copy = df.copy()
    for col in date_columns:
        if col in df_copy.columns:
            df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce').dt.strftime(target_format)
    return df_copy

if __name__ == "__main__":
    sample_data = {
        'Name': [' John Doe ', 'Jane Smith', 'john doe', 'Bob Johnson  '],
        'Email': ['john@example.com', 'invalid-email', 'JANE@EXAMPLE.COM', 'bob@test.org'],
        'Date': ['2023-01-15', '01/20/2023', '2023.02.28', 'March 15, 2023'],
        'Value': [100, 200, 100, 300]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataframe(df, normalize_text=True, drop_duplicates=True)
    cleaned['Email'] = cleaned['Email'].apply(remove_special_characters)
    cleaned = standardize_dates(cleaned, ['Date'])
    
    print("Cleaned DataFrame:")
    print(cleaned)
    print("\nEmail validation:")
    print(cleaned['Email'].apply(validate_email))