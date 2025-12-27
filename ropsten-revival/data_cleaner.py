
import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    subset (list, optional): Column labels to consider for duplicates
    keep (str, optional): Which duplicates to keep ('first', 'last', False)
    
    Returns:
    pd.DataFrame: DataFrame with duplicates removed
    """
    if subset is None:
        subset = df.columns.tolist()
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(df) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df

def clean_missing_values(df, strategy='drop', fill_value=None):
    """
    Handle missing values in DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): 'drop' to remove rows, 'fill' to fill values
    fill_value: Value to use when strategy is 'fill'
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if strategy == 'drop':
        cleaned_df = df.dropna()
        removed_count = len(df) - len(cleaned_df)
        if removed_count > 0:
            print(f"Removed {removed_count} rows with missing values")
    elif strategy == 'fill':
        if fill_value is None:
            fill_value = df.mean(numeric_only=True)
        cleaned_df = df.fillna(fill_value)
        print(f"Filled missing values with {fill_value}")
    else:
        raise ValueError("Strategy must be 'drop' or 'fill'")
    
    return cleaned_df

def normalize_columns(df, columns=None):
    """
    Normalize specified columns to 0-1 range.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): Columns to normalize
    
    Returns:
    pd.DataFrame: DataFrame with normalized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    normalized_df = df.copy()
    
    for col in columns:
        if col in df.columns:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_max > col_min:
                normalized_df[col] = (df[col] - col_min) / (col_max - col_min)
                print(f"Normalized column '{col}'")
    
    return normalized_df

def clean_data_pipeline(df, remove_dups=True, handle_missing=True, normalize=True):
    """
    Complete data cleaning pipeline.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    remove_dups (bool): Whether to remove duplicates
    handle_missing (bool): Whether to handle missing values
    normalize (bool): Whether to normalize numeric columns
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if remove_dups:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if handle_missing:
        cleaned_df = clean_missing_values(cleaned_df, strategy='fill')
    
    if normalize:
        cleaned_df = normalize_columns(cleaned_df)
    
    print(f"Data cleaning complete. Original shape: {df.shape}, Cleaned shape: {cleaned_df.shape}")
    
    return cleaned_df
import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        subset (list, optional): Columns to consider for duplicates
        keep (str, optional): Which duplicates to keep ('first', 'last', False)
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(df) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df

def clean_numeric_column(df, column_name, fill_method='mean'):
    """
    Clean numeric column by handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column_name (str): Name of column to clean
        fill_method (str): Method to fill missing values ('mean', 'median', 'zero')
    
    Returns:
        pd.DataFrame: DataFrame with cleaned column
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    if not pd.api.types.is_numeric_dtype(df[column_name]):
        raise ValueError(f"Column '{column_name}' is not numeric")
    
    df_clean = df.copy()
    
    if fill_method == 'mean':
        fill_value = df_clean[column_name].mean()
    elif fill_method == 'median':
        fill_value = df_clean[column_name].median()
    elif fill_method == 'zero':
        fill_value = 0
    else:
        raise ValueError("fill_method must be 'mean', 'median', or 'zero'")
    
    df_clean[column_name] = df_clean[column_name].fillna(fill_value)
    
    missing_count = df[column_name].isna().sum()
    if missing_count > 0:
        print(f"Filled {missing_count} missing values in '{column_name}' with {fill_method}")
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list, optional): List of required column names
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return True
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
    
    return True
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    filtered_indices = np.where(z_scores < threshold)[0]
    
    return data.iloc[filtered_indices]

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(df, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Main function to clean dataset by removing outliers and normalizing numeric columns
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column not in cleaned_df.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
        elif outlier_method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, column)
        
        if normalize_method == 'minmax':
            cleaned_df[f'{column}_normalized'] = normalize_minmax(cleaned_df, column)
        elif normalize_method == 'zscore':
            cleaned_df[f'{column}_standardized'] = normalize_zscore(cleaned_df, column)
    
    return cleaned_df

def validate_data(df, required_columns=None, allow_nan_ratio=0.1):
    """
    Validate data quality before processing
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    nan_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
    if nan_ratio > allow_nan_ratio:
        print(f"Warning: High NaN ratio ({nan_ratio:.2%}) detected")
    
    return True