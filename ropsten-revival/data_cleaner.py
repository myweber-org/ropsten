
import pandas as pd
import numpy as np
from typing import List, Optional

def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: Input DataFrame
        subset: Columns to consider for identifying duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def normalize_column(df: pd.DataFrame, column: str, method: str = 'minmax') -> pd.DataFrame:
    """
    Normalize specified column using different methods.
    
    Args:
        df: Input DataFrame
        column: Column name to normalize
        method: Normalization method ('minmax', 'zscore', 'log')
    
    Returns:
        DataFrame with normalized column
    """
    df = df.copy()
    
    if method == 'minmax':
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val > min_val:
            df[column] = (df[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df[column].mean()
        std_val = df[column].std()
        if std_val > 0:
            df[column] = (df[column] - mean_val) / std_val
    
    elif method == 'log':
        if df[column].min() > 0:
            df[column] = np.log(df[column])
    
    return df

def clean_missing_values(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
    """
    Handle missing values in DataFrame.
    
    Args:
        df: Input DataFrame
        strategy: Imputation strategy ('mean', 'median', 'mode', 'drop')
    
    Returns:
        DataFrame with handled missing values
    """
    df = df.copy()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if strategy == 'mean':
        for col in numeric_cols:
            df[col].fillna(df[col].mean(), inplace=True)
    
    elif strategy == 'median':
        for col in numeric_cols:
            df[col].fillna(df[col].median(), inplace=True)
    
    elif strategy == 'mode':
        for col in df.columns:
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else None, inplace=True)
    
    elif strategy == 'drop':
        df.dropna(inplace=True)
    
    return df

def validate_dataframe(df: pd.DataFrame) -> dict:
    """
    Validate DataFrame and return statistics.
    
    Args:
        df: Input DataFrame
    
    Returns:
        Dictionary with validation statistics
    """
    stats = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': len(df.select_dtypes(include=['object']).columns)
    }
    
    return stats

def example_usage():
    """
    Example usage of data cleaning functions.
    """
    data = {
        'id': [1, 2, 3, 4, 5, 5],
        'value': [10.5, 20.3, np.nan, 15.7, 30.1, 30.1],
        'category': ['A', 'B', 'A', 'C', 'B', 'B']
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\nValidation Stats:", validate_dataframe(df))
    
    df_clean = remove_duplicates(df, subset=['id'])
    df_clean = clean_missing_values(df_clean, strategy='mean')
    df_clean = normalize_column(df_clean, 'value', method='minmax')
    
    print("\nCleaned DataFrame:")
    print(df_clean)
    print("\nFinal Validation Stats:", validate_dataframe(df_clean))

if __name__ == "__main__":
    example_usage()