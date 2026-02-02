
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
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column):
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]

def remove_outliers_zscore(dataframe, column, threshold=3):
    z_scores = np.abs(stats.zscore(dataframe[column]))
    return dataframe[z_scores < threshold]

def normalize_minmax(dataframe, column):
    min_val = dataframe[column].min()
    max_val = dataframe[column].max()
    dataframe[column + '_normalized'] = (dataframe[column] - min_val) / (max_val - min_val)
    return dataframe

def normalize_zscore(dataframe, column):
    mean_val = dataframe[column].mean()
    std_val = dataframe[column].std()
    dataframe[column + '_standardized'] = (dataframe[column] - mean_val) / std_val
    return dataframe

def handle_missing_values(dataframe, column, strategy='mean'):
    if strategy == 'mean':
        fill_value = dataframe[column].mean()
    elif strategy == 'median':
        fill_value = dataframe[column].median()
    elif strategy == 'mode':
        fill_value = dataframe[column].mode()[0]
    else:
        fill_value = 0
    
    dataframe[column] = dataframe[column].fillna(fill_value)
    return dataframe

def validate_dataframe(dataframe, required_columns):
    missing_columns = [col for col in required_columns if col not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    return True
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column in a dataset using the IQR method.
    
    Parameters:
    data (np.ndarray or list): The dataset.
    column (int): Index of the column to process.
    
    Returns:
    np.ndarray: Data with outliers removed from the specified column.
    """
    data = np.array(data)
    col_data = data[:, column]
    
    Q1 = np.percentile(col_data, 25)
    Q3 = np.percentile(col_data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    mask = (col_data >= lower_bound) & (col_data <= upper_bound)
    filtered_data = data[mask]
    
    return filtered_data