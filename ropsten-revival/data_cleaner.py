
import pandas as pd
import numpy as np
from typing import List, Optional

def remove_duplicate_rows(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df: Input DataFrame
        subset: Columns to consider for identifying duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def normalize_string_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Normalize string columns by stripping whitespace and converting to lowercase.
    
    Args:
        df: Input DataFrame
        columns: List of column names to normalize
    
    Returns:
        DataFrame with normalized string columns
    """
    df_copy = df.copy()
    for col in columns:
        if col in df_copy.columns and df_copy[col].dtype == 'object':
            df_copy[col] = df_copy[col].astype(str).str.strip().str.lower()
    return df_copy

def clean_numeric_outliers(df: pd.DataFrame, column: str, 
                          lower_percentile: float = 0.01, 
                          upper_percentile: float = 0.99) -> pd.DataFrame:
    """
    Remove outliers from a numeric column using percentile-based filtering.
    
    Args:
        df: Input DataFrame
        column: Numeric column to clean
        lower_percentile: Lower percentile threshold
        upper_percentile: Upper percentile threshold
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in df.columns:
        return df
    
    lower_bound = df[column].quantile(lower_percentile)
    upper_bound = df[column].quantile(upper_percentile)
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def fill_missing_values(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
    """
    Fill missing values in numeric columns.
    
    Args:
        df: Input DataFrame
        strategy: Method to fill missing values ('mean', 'median', 'mode', 'zero')
    
    Returns:
        DataFrame with filled missing values
    """
    df_copy = df.copy()
    
    for col in df_copy.select_dtypes(include=[np.number]).columns:
        if df_copy[col].isnull().any():
            if strategy == 'mean':
                df_copy[col].fillna(df_copy[col].mean(), inplace=True)
            elif strategy == 'median':
                df_copy[col].fillna(df_copy[col].median(), inplace=True)
            elif strategy == 'mode':
                df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
            elif strategy == 'zero':
                df_copy[col].fillna(0, inplace=True)
    
    return df_copy

def clean_dataframe(df: pd.DataFrame, 
                   duplicate_subset: Optional[List[str]] = None,
                   normalize_cols: Optional[List[str]] = None,
                   outlier_cols: Optional[List[str]] = None,
                   fill_missing_strategy: str = 'mean') -> pd.DataFrame:
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: Input DataFrame
        duplicate_subset: Columns for duplicate removal
        normalize_cols: Columns to normalize
        outlier_cols: Columns for outlier removal
        fill_missing_strategy: Strategy for filling missing values
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    # Remove duplicates
    cleaned_df = remove_duplicate_rows(cleaned_df, duplicate_subset)
    
    # Normalize string columns
    if normalize_cols:
        cleaned_df = normalize_string_columns(cleaned_df, normalize_cols)
    
    # Remove outliers
    if outlier_cols:
        for col in outlier_cols:
            cleaned_df = clean_numeric_outliers(cleaned_df, col)
    
    # Fill missing values
    cleaned_df = fill_missing_values(cleaned_df, fill_missing_strategy)
    
    return cleaned_df