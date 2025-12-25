import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        subset (list, optional): Column labels to consider for duplicates.
        keep (str, optional): Which duplicates to keep.
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    return cleaned_df

def clean_numeric_column(df, column):
    """
    Clean a numeric column by removing non-numeric characters.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to clean.
    
    Returns:
        pd.DataFrame: DataFrame with cleaned column.
    """
    if column not in df.columns:
        return df
    
    df[column] = pd.to_numeric(df[column].astype(str).str.replace(r'[^\d.-]', '', regex=True), errors='coerce')
    return df

def validate_email_column(df, column):
    """
    Validate email addresses in a column.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name containing email addresses.
    
    Returns:
        pd.DataFrame: DataFrame with validation results.
    """
    import re
    
    if column not in df.columns:
        return df
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    df['email_valid'] = df[column].apply(lambda x: bool(re.match(pattern, str(x))) if pd.notnull(x) else False)
    return df