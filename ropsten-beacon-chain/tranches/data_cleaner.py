
import pandas as pd
import numpy as np
from typing import Union, List, Optional

def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df: Input DataFrame
        subset: List of column names to consider for identifying duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def handle_missing_values(df: pd.DataFrame, strategy: str = 'drop', fill_value: Union[int, float, str] = None) -> pd.DataFrame:
    """
    Handle missing values in a DataFrame.
    
    Args:
        df: Input DataFrame
        strategy: One of 'drop', 'fill', or 'interpolate'
        fill_value: Value to use when strategy is 'fill'
    
    Returns:
        DataFrame with missing values handled
    """
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'fill':
        if fill_value is None:
            raise ValueError("fill_value must be provided when strategy is 'fill'")
        return df.fillna(fill_value)
    elif strategy == 'interpolate':
        return df.interpolate()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def normalize_column(df: pd.DataFrame, column: str, method: str = 'minmax') -> pd.DataFrame:
    """
    Normalize a column in the DataFrame.
    
    Args:
        df: Input DataFrame
        column: Name of column to normalize
        method: Normalization method ('minmax' or 'zscore')
    
    Returns:
        DataFrame with normalized column
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_copy = df.copy()
    
    if method == 'minmax':
        min_val = df_copy[column].min()
        max_val = df_copy[column].max()
        if max_val != min_val:
            df_copy[column] = (df_copy[column] - min_val) / (max_val - min_val)
    elif method == 'zscore':
        mean_val = df_copy[column].mean()
        std_val = df_copy[column].std()
        if std_val != 0:
            df_copy[column] = (df_copy[column] - mean_val) / std_val
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return df_copy

def filter_outliers(df: pd.DataFrame, column: str, method: str = 'iqr', multiplier: float = 1.5) -> pd.DataFrame:
    """
    Filter outliers from a DataFrame based on a specific column.
    
    Args:
        df: Input DataFrame
        column: Column to check for outliers
        method: Method to detect outliers ('iqr' or 'zscore')
        multiplier: Multiplier for IQR method or threshold for zscore method
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_copy = df.copy()
    
    if method == 'iqr':
        Q1 = df_copy[column].quantile(0.25)
        Q3 = df_copy[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        mask = (df_copy[column] >= lower_bound) & (df_copy[column] <= upper_bound)
    elif method == 'zscore':
        mean_val = df_copy[column].mean()
        std_val = df_copy[column].std()
        if std_val != 0:
            z_scores = np.abs((df_copy[column] - mean_val) / std_val)
            mask = z_scores <= multiplier
        else:
            mask = pd.Series([True] * len(df_copy))
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")
    
    return df_copy[mask]

def convert_column_types(df: pd.DataFrame, column_types: dict) -> pd.DataFrame:
    """
    Convert columns to specified data types.
    
    Args:
        df: Input DataFrame
        column_types: Dictionary mapping column names to target data types
    
    Returns:
        DataFrame with converted column types
    """
    df_copy = df.copy()
    
    for column, dtype in column_types.items():
        if column in df_copy.columns:
            try:
                df_copy[column] = df_copy[column].astype(dtype)
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not convert column '{column}' to {dtype}: {e}")
    
    return df_copy

def validate_dataframe(df: pd.DataFrame, required_columns: List[str] = None) -> bool:
    """
    Validate a DataFrame for basic integrity.
    
    Args:
        df: DataFrame to validate
        required_columns: List of columns that must be present
    
    Returns:
        True if DataFrame passes validation
    """
    if not isinstance(df, pd.DataFrame):
        return False
    
    if df.empty:
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    return True