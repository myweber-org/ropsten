
import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    subset (list, optional): Column labels to consider for duplicates
    keep (str): Which duplicates to keep - 'first', 'last', or False
    
    Returns:
    pd.DataFrame: DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(df) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df

def clean_missing_values(df, strategy='drop', fill_value=None):
    """
    Handle missing values in DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): 'drop' to remove rows, 'fill' to fill values
    fill_value: Value to fill missing values with
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if strategy == 'drop':
        cleaned_df = df.dropna()
        print(f"Removed rows with missing values")
    elif strategy == 'fill':
        cleaned_df = df.fillna(fill_value)
        print(f"Filled missing values with {fill_value}")
    else:
        raise ValueError("Strategy must be 'drop' or 'fill'")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    bool: True if validation passes
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True

def process_dataframe(df, remove_dups=True, handle_missing=True):
    """
    Complete data cleaning pipeline.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    remove_dups (bool): Whether to remove duplicates
    handle_missing (bool): Whether to handle missing values
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    validate_dataframe(df)
    
    if remove_dups:
        df = remove_duplicates(df)
    
    if handle_missing:
        df = clean_missing_values(df, strategy='fill', fill_value=0)
    
    print(f"Data cleaning complete. Final shape: {df.shape}")
    return df