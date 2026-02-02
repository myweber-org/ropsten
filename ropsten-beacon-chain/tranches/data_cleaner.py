import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        subset (list, optional): Columns to consider for duplicates.
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    return df.drop_duplicates(subset=subset, keep='first')

def fill_missing_values(df, strategy='mean', columns=None):
    """
    Fill missing values in specified columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        strategy (str): 'mean', 'median', 'mode', or 'constant'.
        columns (list): Columns to fill. If None, all columns are considered.
    
    Returns:
        pd.DataFrame: DataFrame with missing values filled.
    """
    df_filled = df.copy()
    
    if columns is None:
        columns = df.columns
    
    for col in columns:
        if df[col].dtype in ['int64', 'float64']:
            if strategy == 'mean':
                df_filled[col] = df[col].fillna(df[col].mean())
            elif strategy == 'median':
                df_filled[col] = df[col].fillna(df[col].median())
            elif strategy == 'constant':
                df_filled[col] = df[col].fillna(0)
        elif strategy == 'mode':
            df_filled[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else None)
    
    return df_filled

def normalize_column(df, column, method='minmax'):
    """
    Normalize a column using specified method.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to normalize.
        method (str): 'minmax' or 'zscore'.
    
    Returns:
        pd.DataFrame: DataFrame with normalized column.
    """
    df_normalized = df.copy()
    
    if method == 'minmax':
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val != min_val:
            df_normalized[column] = (df[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df[column].mean()
        std_val = df[column].std()
        if std_val != 0:
            df_normalized[column] = (df[column] - mean_val) / std_val
    
    return df_normalized

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers using IQR method.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to check for outliers.
        multiplier (float): IQR multiplier.
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    if df[column].dtype not in ['int64', 'float64']:
        return df
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def clean_dataset(df, config):
    """
    Apply multiple cleaning operations based on configuration.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        config (dict): Configuration dictionary with cleaning steps.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if config.get('remove_duplicates'):
        cleaned_df = remove_duplicates(cleaned_df, config.get('duplicate_subset'))
    
    if config.get('fill_missing'):
        cleaned_df = fill_missing_values(
            cleaned_df,
            strategy=config.get('fill_strategy', 'mean'),
            columns=config.get('fill_columns')
        )
    
    if config.get('normalize'):
        for col, method in config.get('normalize_columns', {}).items():
            if col in cleaned_df.columns:
                cleaned_df = normalize_column(cleaned_df, col, method)
    
    if config.get('remove_outliers'):
        for col in config.get('outlier_columns', []):
            if col in cleaned_df.columns:
                cleaned_df = remove_outliers_iqr(
                    cleaned_df,
                    col,
                    multiplier=config.get('outlier_multiplier', 1.5)
                )
    
    return cleaned_df