import pandas as pd
import numpy as np
from typing import Optional

def remove_duplicates(df: pd.DataFrame, subset: Optional[list] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: Input DataFrame
        subset: Columns to consider for identifying duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def fill_missing_values(df: pd.DataFrame, strategy: str = 'mean', columns: Optional[list] = None) -> pd.DataFrame:
    """
    Fill missing values in DataFrame using specified strategy.
    
    Args:
        df: Input DataFrame
        strategy: 'mean', 'median', 'mode', or 'zero'
        columns: Specific columns to fill, None for all numeric columns
    
    Returns:
        DataFrame with missing values filled
    """
    df_filled = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df.columns:
            if strategy == 'mean':
                df_filled[col] = df[col].fillna(df[col].mean())
            elif strategy == 'median':
                df_filled[col] = df[col].fillna(df[col].median())
            elif strategy == 'mode':
                df_filled[col] = df[col].fillna(df[col].mode()[0])
            elif strategy == 'zero':
                df_filled[col] = df[col].fillna(0)
    
    return df_filled

def normalize_columns(df: pd.DataFrame, columns: Optional[list] = None) -> pd.DataFrame:
    """
    Normalize specified columns using min-max scaling.
    
    Args:
        df: Input DataFrame
        columns: Columns to normalize, None for all numeric columns
    
    Returns:
        DataFrame with normalized columns
    """
    df_normalized = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df.columns:
            col_min = df[col].min()
            col_max = df[col].max()
            if col_max > col_min:
                df_normalized[col] = (df[col] - col_min) / (col_max - col_min)
    
    return df_normalized

def clean_dataframe(df: pd.DataFrame, 
                   remove_dups: bool = True,
                   fill_na: bool = True,
                   normalize: bool = False) -> pd.DataFrame:
    """
    Main function to clean DataFrame with multiple operations.
    
    Args:
        df: Input DataFrame
        remove_dups: Whether to remove duplicates
        fill_na: Whether to fill missing values
        normalize: Whether to normalize numeric columns
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if remove_dups:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if fill_na:
        cleaned_df = fill_missing_values(cleaned_df, strategy='mean')
    
    if normalize:
        cleaned_df = normalize_columns(cleaned_df)
    
    return cleaned_df
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers from specified column using IQR method.
    
    Args:
        dataframe: pandas DataFrame
        column: column name to process
        threshold: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    return dataframe[(dataframe[column] >= lower_bound) & 
                     (dataframe[column] <= upper_bound)]

def normalize_minmax(dataframe, columns=None):
    """
    Apply min-max normalization to specified columns.
    
    Args:
        dataframe: pandas DataFrame
        columns: list of column names or None for all numeric columns
    
    Returns:
        Normalized DataFrame
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns
    
    result = dataframe.copy()
    for col in columns:
        if col in dataframe.columns:
            min_val = dataframe[col].min()
            max_val = dataframe[col].max()
            if max_val > min_val:
                result[col] = (dataframe[col] - min_val) / (max_val - min_val)
    
    return result

def z_score_normalize(dataframe, columns=None):
    """
    Apply z-score normalization to specified columns.
    
    Args:
        dataframe: pandas DataFrame
        columns: list of column names or None for all numeric columns
    
    Returns:
        Z-score normalized DataFrame
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns
    
    result = dataframe.copy()
    for col in columns:
        if col in dataframe.columns:
            mean_val = dataframe[col].mean()
            std_val = dataframe[col].std()
            if std_val > 0:
                result[col] = (dataframe[col] - mean_val) / std_val
    
    return result

def handle_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame.
    
    Args:
        dataframe: pandas DataFrame
        strategy: 'mean', 'median', 'mode', or 'drop'
        columns: list of column names or None for all columns
    
    Returns:
        DataFrame with handled missing values
    """
    if columns is None:
        columns = dataframe.columns
    
    result = dataframe.copy()
    
    for col in columns:
        if col in dataframe.columns and dataframe[col].isnull().any():
            if strategy == 'drop':
                result = result.dropna(subset=[col])
            elif strategy == 'mean' and pd.api.types.is_numeric_dtype(dataframe[col]):
                result[col] = result[col].fillna(dataframe[col].mean())
            elif strategy == 'median' and pd.api.types.is_numeric_dtype(dataframe[col]):
                result[col] = result[col].fillna(dataframe[col].median())
            elif strategy == 'mode':
                result[col] = result[col].fillna(dataframe[col].mode()[0])
    
    return result

def clean_dataset(dataframe, numeric_columns=None, outlier_threshold=1.5):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        dataframe: pandas DataFrame
        numeric_columns: list of numeric column names
        outlier_threshold: IQR threshold for outlier removal
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = dataframe.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col, outlier_threshold)
    
    cleaned_df = handle_missing_values(cleaned_df, strategy='median', columns=numeric_columns)
    cleaned_df = z_score_normalize(cleaned_df, columns=numeric_columns)
    
    return cleaned_df
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df.reset_index(drop=True)

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to analyze.
    
    Returns:
    dict: Dictionary containing statistical measures.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    If no columns specified, clean all numeric columns.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns (list): List of column names to clean.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{col}'")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[np.random.choice(df.index, 50), 'A'] = np.random.uniform(500, 1000, 50)
    
    print("Original DataFrame shape:", df.shape)
    print("\nOriginal statistics for column 'A':")
    print(calculate_basic_stats(df, 'A'))
    
    cleaned_df = clean_numeric_data(df, ['A'])
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("\nCleaned statistics for column 'A':")
    print(calculate_basic_stats(cleaned_df, 'A'))