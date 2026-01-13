import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        subset (list, optional): Columns to consider for duplicates
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        strategy (str): 'mean', 'median', 'mode', or 'drop'
        columns (list): Specific columns to process
    
    Returns:
        pd.DataFrame: DataFrame with handled missing values
    """
    df_copy = df.copy()
    
    if columns is None:
        columns = df_copy.columns
    
    for col in columns:
        if df_copy[col].isnull().any():
            if strategy == 'mean':
                df_copy[col].fillna(df_copy[col].mean(), inplace=True)
            elif strategy == 'median':
                df_copy[col].fillna(df_copy[col].median(), inplace=True)
            elif strategy == 'mode':
                df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
            elif strategy == 'drop':
                df_copy = df_copy.dropna(subset=[col])
    
    return df_copy

def normalize_columns(df, columns=None):
    """
    Normalize specified columns using min-max scaling.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): Columns to normalize
    
    Returns:
        pd.DataFrame: DataFrame with normalized columns
    """
    df_copy = df.copy()
    
    if columns is None:
        columns = df_copy.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df_copy.columns and df_copy[col].dtype in [np.float64, np.int64]:
            min_val = df_copy[col].min()
            max_val = df_copy[col].max()
            
            if max_val > min_val:
                df_copy[col] = (df_copy[col] - min_val) / (max_val - min_val)
    
    return df_copy

def clean_dataframe(df, operations=None):
    """
    Apply multiple cleaning operations to DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        operations (dict): Dictionary of operations to apply
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if operations is None:
        operations = {
            'remove_duplicates': True,
            'handle_missing': {'strategy': 'mean'},
            'normalize': True
        }
    
    df_clean = df.copy()
    
    if operations.get('remove_duplicates'):
        df_clean = remove_duplicates(df_clean)
    
    if 'handle_missing' in operations:
        config = operations['handle_missing']
        strategy = config.get('strategy', 'mean')
        columns = config.get('columns')
        df_clean = handle_missing_values(df_clean, strategy, columns)
    
    if operations.get('normalize'):
        columns = operations.get('normalize_columns')
        df_clean = normalize_columns(df_clean, columns)
    
    return df_clean

def validate_dataframe(df, checks=None):
    """
    Validate DataFrame for common data quality issues.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        checks (list): List of checks to perform
    
    Returns:
        dict: Dictionary of validation results
    """
    if checks is None:
        checks = ['missing', 'duplicates', 'data_types']
    
    results = {}
    
    if 'missing' in checks:
        missing_counts = df.isnull().sum()
        results['missing_values'] = missing_counts[missing_counts > 0].to_dict()
    
    if 'duplicates' in checks:
        duplicate_count = df.duplicated().sum()
        results['duplicate_rows'] = duplicate_count
    
    if 'data_types' in checks:
        results['data_types'] = df.dtypes.astype(str).to_dict()
    
    return results