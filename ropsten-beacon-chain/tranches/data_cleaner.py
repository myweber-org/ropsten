
import pandas as pd
import numpy as np
from typing import List, Union

def remove_duplicates(df: pd.DataFrame, subset: Union[List[str], str] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df: Input DataFrame
        subset: Column(s) to consider for identifying duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def normalize_column(df: pd.DataFrame, column: str, method: str = 'minmax') -> pd.DataFrame:
    """
    Normalize a column using specified method.
    
    Args:
        df: Input DataFrame
        column: Column name to normalize
        method: Normalization method ('minmax' or 'zscore')
    
    Returns:
        DataFrame with normalized column
    """
    df = df.copy()
    
    if method == 'minmax':
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val != min_val:
            df[column] = (df[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df[column].mean()
        std_val = df[column].std()
        if std_val > 0:
            df[column] = (df[column] - mean_val) / std_val
    
    return df

def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
    """
    Handle missing values in numeric columns.
    
    Args:
        df: Input DataFrame
        strategy: Imputation strategy ('mean', 'median', or 'drop')
    
    Returns:
        DataFrame with handled missing values
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        df = df.dropna(subset=numeric_cols)
    else:
        for col in numeric_cols:
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            else:
                fill_value = 0
            
            df[col] = df[col].fillna(fill_value)
    
    return df

def clean_dataframe(df: pd.DataFrame, 
                   deduplicate: bool = True,
                   normalize_cols: List[str] = None,
                   missing_strategy: str = 'mean') -> pd.DataFrame:
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: Input DataFrame
        deduplicate: Whether to remove duplicates
        normalize_cols: Columns to normalize
        missing_strategy: Strategy for handling missing values
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if deduplicate:
        cleaned_df = remove_duplicates(cleaned_df)
    
    cleaned_df = handle_missing_values(cleaned_df, strategy=missing_strategy)
    
    if normalize_cols:
        for col in normalize_cols:
            if col in cleaned_df.columns:
                cleaned_df = normalize_column(cleaned_df, col)
    
    return cleaned_df.reset_index(drop=True)import pandas as pd

def clean_dataset(df, columns_to_check=None, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        columns_to_check (list, optional): Specific columns to check for duplicates.
            If None, checks all columns. Defaults to None.
        fill_missing (str or value): Method to fill missing values.
            Can be 'mean', 'median', 'mode', or a specific value. Defaults to 'mean'.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove duplicates
    if columns_to_check:
        cleaned_df = cleaned_df.drop_duplicates(subset=columns_to_check)
    else:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Handle missing values
    for column in cleaned_df.columns:
        if cleaned_df[column].isnull().any():
            if fill_missing == 'mean':
                if pd.api.types.is_numeric_dtype(cleaned_df[column]):
                    cleaned_df[column].fillna(cleaned_df[column].mean(), inplace=True)
            elif fill_missing == 'median':
                if pd.api.types.is_numeric_dtype(cleaned_df[column]):
                    cleaned_df[column].fillna(cleaned_df[column].median(), inplace=True)
            elif fill_missing == 'mode':
                cleaned_df[column].fillna(cleaned_df[column].mode()[0], inplace=True)
            else:
                cleaned_df[column].fillna(fill_missing, inplace=True)
    
    return cleaned_df

def validate_dataset(df, required_columns=None, min_rows=1):
    """
    Validate the structure and content of a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list, optional): List of columns that must be present.
        min_rows (int): Minimum number of rows required.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "Dataset is valid"