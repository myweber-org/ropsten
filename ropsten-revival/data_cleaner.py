
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        multiplier: IQR multiplier for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(data), index=data.index)
    
    return (data[column] - min_val) / (max_val - min_val)

def standardize_zscore(data, column):
    """
    Standardize data using Z-score normalization.
    
    Args:
        data: pandas DataFrame
        column: column name to standardize
    
    Returns:
        Series with standardized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    return (data[column] - mean_val) / std_val

def handle_missing_values(data, strategy='mean'):
    """
    Handle missing values in numeric columns.
    
    Args:
        data: pandas DataFrame
        strategy: imputation strategy ('mean', 'median', 'mode', 'zero')
    
    Returns:
        DataFrame with missing values handled
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    if strategy == 'mean':
        for col in numeric_cols:
            data[col] = data[col].fillna(data[col].mean())
    elif strategy == 'median':
        for col in numeric_cols:
            data[col] = data[col].fillna(data[col].median())
    elif strategy == 'mode':
        for col in numeric_cols:
            data[col] = data[col].fillna(data[col].mode()[0])
    elif strategy == 'zero':
        data[numeric_cols] = data[numeric_cols].fillna(0)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return data

def clean_dataset(data, outlier_columns=None, normalize_columns=None, 
                  standardize_columns=None, missing_strategy='mean'):
    """
    Comprehensive dataset cleaning pipeline.
    
    Args:
        data: pandas DataFrame
        outlier_columns: list of columns for outlier removal
        normalize_columns: list of columns for min-max normalization
        standardize_columns: list of columns for z-score standardization
        missing_strategy: strategy for handling missing values
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_data = data.copy()
    
    # Handle missing values
    cleaned_data = handle_missing_values(cleaned_data, strategy=missing_strategy)
    
    # Remove outliers
    if outlier_columns:
        for col in outlier_columns:
            if col in cleaned_data.columns:
                cleaned_data = remove_outliers_iqr(cleaned_data, col)
    
    # Apply normalization
    if normalize_columns:
        for col in normalize_columns:
            if col in cleaned_data.columns:
                cleaned_data[f'{col}_normalized'] = normalize_minmax(cleaned_data, col)
    
    # Apply standardization
    if standardize_columns:
        for col in standardize_columns:
            if col in cleaned_data.columns:
                cleaned_data[f'{col}_standardized'] = standardize_zscore(cleaned_data, col)
    
    return cleaned_dataimport pandas as pd
import numpy as np
from typing import Union, List, Dict, Any

def remove_duplicates(df: pd.DataFrame, subset: Union[List[str], None] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.
    """
    return df.drop_duplicates(subset=subset, keep='first')

def convert_column_types(df: pd.DataFrame, 
                         type_mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Convert column types based on mapping dictionary.
    """
    df_copy = df.copy()
    for column, dtype in type_mapping.items():
        if column in df_copy.columns:
            try:
                if dtype == 'datetime':
                    df_copy[column] = pd.to_datetime(df_copy[column])
                elif dtype == 'numeric':
                    df_copy[column] = pd.to_numeric(df_copy[column], errors='coerce')
                elif dtype == 'category':
                    df_copy[column] = df_copy[column].astype('category')
                else:
                    df_copy[column] = df_copy[column].astype(dtype)
            except Exception as e:
                print(f"Error converting column {column}: {e}")
    return df_copy

def handle_missing_values(df: pd.DataFrame, 
                          strategy: str = 'drop',
                          fill_value: Any = None) -> pd.DataFrame:
    """
    Handle missing values using specified strategy.
    """
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'fill':
        if fill_value is not None:
            return df.fillna(fill_value)
        else:
            return df.fillna(df.mean(numeric_only=True))
    else:
        return df

def normalize_column(df: pd.DataFrame, 
                     column: str,
                     method: str = 'minmax') -> pd.DataFrame:
    """
    Normalize specified column using given method.
    """
    if column not in df.columns:
        return df
    
    df_copy = df.copy()
    
    if method == 'minmax':
        min_val = df_copy[column].min()
        max_val = df_copy[column].max()
        if max_val > min_val:
            df_copy[column] = (df_copy[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df_copy[column].mean()
        std_val = df_copy[column].std()
        if std_val > 0:
            df_copy[column] = (df_copy[column] - mean_val) / std_val
    
    return df_copy

def clean_dataframe(df: pd.DataFrame,
                    deduplicate: bool = True,
                    type_conversions: Union[Dict[str, str], None] = None,
                    missing_strategy: str = 'drop',
                    normalize_columns: Union[List[str], None] = None) -> pd.DataFrame:
    """
    Comprehensive data cleaning pipeline.
    """
    cleaned_df = df.copy()
    
    if deduplicate:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if type_conversions:
        cleaned_df = convert_column_types(cleaned_df, type_conversions)
    
    cleaned_df = handle_missing_values(cleaned_df, strategy=missing_strategy)
    
    if normalize_columns:
        for column in normalize_columns:
            cleaned_df = normalize_column(cleaned_df, column)
    
    return cleaned_df