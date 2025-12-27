import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        subset (list, optional): Column names to consider for duplicates
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def fill_missing_values(df, strategy='mean', columns=None):
    """
    Fill missing values in DataFrame columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        strategy (str): 'mean', 'median', 'mode', or 'constant'
        columns (list): Specific columns to fill, None for all numeric columns
    
    Returns:
        pd.DataFrame: DataFrame with filled values
    """
    df_filled = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df.columns:
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mode':
                fill_value = df[col].mode()[0] if not df[col].mode().empty else 0
            elif strategy == 'constant':
                fill_value = 0
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            df_filled[col] = df[col].fillna(fill_value)
    
    return df_filled

def normalize_columns(df, columns=None, method='minmax'):
    """
    Normalize numeric columns to specified range.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): Columns to normalize, None for all numeric columns
        method (str): 'minmax' or 'zscore'
    
    Returns:
        pd.DataFrame: DataFrame with normalized columns
    """
    df_normalized = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df.columns and df[col].dtype in [np.float64, np.int64]:
            if method == 'minmax':
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
            elif method == 'zscore':
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val > 0:
                    df_normalized[col] = (df[col] - mean_val) / std_val
    
    return df_normalized

def clean_dataframe(df, remove_dups=True, fill_na=True, normalize=True):
    """
    Apply multiple cleaning operations to DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        remove_dups (bool): Whether to remove duplicates
        fill_na (bool): Whether to fill missing values
        normalize (bool): Whether to normalize numeric columns
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if remove_dups:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if fill_na:
        cleaned_df = fill_missing_values(cleaned_df, strategy='mean')
    
    if normalize:
        cleaned_df = normalize_columns(cleaned_df, method='minmax')
    
    return cleaned_df