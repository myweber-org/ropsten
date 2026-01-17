
import pandas as pd
import re

def clean_dataframe(df, columns_to_normalize=None, remove_duplicates=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing string columns.
    
    Args:
        df: pandas DataFrame to clean
        columns_to_normalize: list of column names to normalize (default: all object columns)
        remove_duplicates: whether to remove duplicate rows (default: True)
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    if remove_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if columns_to_normalize is None:
        columns_to_normalize = cleaned_df.select_dtypes(include=['object']).columns.tolist()
    
    for col in columns_to_normalize:
        if col in cleaned_df.columns and cleaned_df[col].dtype == 'object':
            cleaned_df[col] = cleaned_df[col].apply(_normalize_string)
            print(f"Normalized column: {col}")
    
    return cleaned_df

def _normalize_string(text):
    """
    Normalize a string by converting to lowercase, removing extra whitespace,
    and stripping special characters.
    
    Args:
        text: string to normalize
    
    Returns:
        Normalized string
    """
    if pd.isna(text):
        return text
    
    text = str(text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = re.sub(r'[^\w\s-]', '', text)
    
    return text

def validate_email_column(df, email_column):
    """
    Validate email addresses in a DataFrame column.
    
    Args:
        df: pandas DataFrame
        email_column: name of the column containing email addresses
    
    Returns:
        DataFrame with valid emails and validation status
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    df_validated = df.copy()
    df_validated['email_valid'] = df_validated[email_column].apply(
        lambda x: bool(re.match(email_pattern, str(x))) if pd.notna(x) else False
    )
    
    valid_count = df_validated['email_valid'].sum()
    total_count = len(df_validated)
    
    print(f"Valid emails: {valid_count}/{total_count} ({valid_count/total_count*100:.1f}%)")
    
    return df_validated