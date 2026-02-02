
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
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def clean_dataset(df, columns=None):
    """
    Clean multiple columns in a DataFrame by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to clean. If None, clean all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    for column in columns:
        if column in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[column]):
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 3, 4, 5, 100],
        'B': [10, 20, 30, 40, 50, 200],
        'C': ['a', 'b', 'c', 'd', 'e', 'f']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, columns=['A', 'B'])
    print("\nCleaned DataFrame:")
    print(cleaned)import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df: pandas DataFrame to clean.
        drop_duplicates: Boolean indicating whether to remove duplicate rows.
        fill_missing: Strategy for filling missing values. Can be None, 'mean', 'median', or a scalar value.
    
    Returns:
        Cleaned pandas DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing is not None:
        if fill_missing == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_missing == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        else:
            cleaned_df = cleaned_df.fillna(fill_missing)
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate a pandas DataFrame for required columns and data types.
    
    Args:
        df: pandas DataFrame to validate.
        required_columns: List of column names that must be present in the DataFrame.
    
    Returns:
        Boolean indicating whether the DataFrame passes validation.
    """
    if required_columns is not None:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    if df.empty:
        print("DataFrame is empty")
        return False
    
    return True

def normalize_numeric_columns(df, columns=None):
    """
    Normalize numeric columns in a DataFrame using min-max scaling.
    
    Args:
        df: pandas DataFrame containing numeric columns to normalize.
        columns: List of column names to normalize. If None, all numeric columns are normalized.
    
    Returns:
        DataFrame with normalized numeric columns.
    """
    normalized_df = df.copy()
    
    if columns is None:
        numeric_cols = normalized_df.select_dtypes(include=['float64', 'int64']).columns
    else:
        numeric_cols = [col for col in columns if col in normalized_df.columns]
    
    for col in numeric_cols:
        col_min = normalized_df[col].min()
        col_max = normalized_df[col].max()
        
        if col_max != col_min:
            normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)
        else:
            normalized_df[col] = 0
    
    return normalized_df