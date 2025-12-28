import pandas as pd
import numpy as np
from typing import List, Optional

def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.
    """
    return df.drop_duplicates(subset=subset, keep='first')

def normalize_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Normalize specified column to range [0, 1].
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    col_min = df[column_name].min()
    col_max = df[column_name].max()
    
    if col_max == col_min:
        df[column_name] = 0.5
    else:
        df[column_name] = (df[column_name] - col_min) / (col_max - col_min)
    
    return df

def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
    """
    Handle missing values using specified strategy.
    """
    if strategy == 'mean':
        return df.fillna(df.mean(numeric_only=True))
    elif strategy == 'median':
        return df.fillna(df.median(numeric_only=True))
    elif strategy == 'mode':
        return df.fillna(df.mode().iloc[0])
    elif strategy == 'drop':
        return df.dropna()
    else:
        raise ValueError("Strategy must be 'mean', 'median', 'mode', or 'drop'")

def clean_dataframe(df: pd.DataFrame, 
                   deduplicate: bool = True,
                   normalize_cols: Optional[List[str]] = None,
                   missing_strategy: str = 'mean') -> pd.DataFrame:
    """
    Comprehensive data cleaning pipeline.
    """
    cleaned_df = df.copy()
    
    if deduplicate:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if normalize_cols:
        for col in normalize_cols:
            if col in cleaned_df.columns:
                cleaned_df = normalize_column(cleaned_df, col)
    
    cleaned_df = handle_missing_values(cleaned_df, strategy=missing_strategy)
    
    return cleaned_dfimport numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    if max_val == min_val:
        return df[column].apply(lambda x: 0.0)
    return df[column].apply(lambda x: (x - min_val) / (max_val - min_val))

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    return cleaned_df.reset_index(drop=True)

def validate_dataframe(df):
    required_checks = [
        (lambda d: isinstance(d, pd.DataFrame), "Input must be a pandas DataFrame"),
        (lambda d: not d.empty, "DataFrame cannot be empty"),
        (lambda d: d.isnull().sum().sum() == 0, "DataFrame contains null values")
    ]
    for check, message in required_checks:
        if not check(df):
            raise ValueError(message)
    return True