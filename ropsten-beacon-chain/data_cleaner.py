
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers from a DataFrame column using IQR method.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    threshold (float): IQR multiplier for outlier detection
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    return filtered_df.copy()

def normalize_minmax(dataframe, columns=None):
    """
    Normalize specified columns using Min-Max scaling.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    columns (list): List of column names to normalize. If None, normalize all numeric columns.
    
    Returns:
    pd.DataFrame: DataFrame with normalized columns
    """
    if columns is None:
        numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    normalized_df = dataframe.copy()
    
    for col in columns:
        if col not in normalized_df.columns:
            continue
            
        if normalized_df[col].dtype in [np.float64, np.int64]:
            col_min = normalized_df[col].min()
            col_max = normalized_df[col].max()
            
            if col_max != col_min:
                normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)
            else:
                normalized_df[col] = 0.0
    
    return normalized_df

def handle_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame columns.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    strategy (str): Imputation strategy ('mean', 'median', 'mode', 'drop')
    columns (list): List of column names to process. If None, process all columns.
    
    Returns:
    pd.DataFrame: DataFrame with handled missing values
    """
    if columns is None:
        columns = dataframe.columns
    
    processed_df = dataframe.copy()
    
    for col in columns:
        if col not in processed_df.columns:
            continue
            
        if processed_df[col].isnull().any():
            if strategy == 'drop':
                processed_df = processed_df.dropna(subset=[col])
            elif strategy == 'mean':
                if processed_df[col].dtype in [np.float64, np.int64]:
                    processed_df[col].fillna(processed_df[col].mean(), inplace=True)
            elif strategy == 'median':
                if processed_df[col].dtype in [np.float64, np.int64]:
                    processed_df[col].fillna(processed_df[col].median(), inplace=True)
            elif strategy == 'mode':
                processed_df[col].fillna(processed_df[col].mode()[0], inplace=True)
    
    return processed_df

def calculate_statistics(dataframe):
    """
    Calculate basic statistics for numeric columns.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    
    Returns:
    dict: Dictionary containing statistics for each numeric column
    """
    stats_dict = {}
    
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        stats_dict[col] = {
            'mean': dataframe[col].mean(),
            'median': dataframe[col].median(),
            'std': dataframe[col].std(),
            'min': dataframe[col].min(),
            'max': dataframe[col].max(),
            'count': dataframe[col].count(),
            'missing': dataframe[col].isnull().sum()
        }
    
    return stats_dict

def clean_dataset(dataframe, numeric_columns=None, outlier_threshold=1.5, 
                  normalize=True, missing_strategy='mean'):
    """
    Complete data cleaning pipeline.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of numeric column names to process
    outlier_threshold (float): IQR threshold for outlier removal
    normalize (bool): Whether to normalize numeric columns
    missing_strategy (str): Strategy for handling missing values
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = dataframe.copy()
    
    cleaned_df = handle_missing_values(cleaned_df, strategy=missing_strategy)
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col, outlier_threshold)
    
    if normalize:
        cleaned_df = normalize_minmax(cleaned_df, numeric_columns)
    
    return cleaned_df