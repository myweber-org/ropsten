
import pandas as pd

def clean_dataset(df, remove_duplicates=True, fill_method=None):
    """
    Clean a pandas DataFrame by handling missing values and optionally removing duplicates.
    
    Args:
        df: pandas DataFrame to clean
        remove_duplicates: Boolean, if True remove duplicate rows
        fill_method: None, 'mean', 'median', or 'ffill' for handling missing values
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if fill_method is None:
        cleaned_df = cleaned_df.dropna()
    elif fill_method == 'mean':
        cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
    elif fill_method == 'median':
        cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
    elif fill_method == 'ffill':
        cleaned_df = cleaned_df.fillna(method='ffill')
    
    # Remove duplicates if requested
    if remove_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate that a DataFrame meets basic requirements.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: List of column names that must be present
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    removed_count = len(dataframe) - len(filtered_df)
    if removed_count > 0:
        print(f"Removed {removed_count} outliers from column '{column}'")
    
    return filtered_df

def normalize_minmax(dataframe, columns=None):
    """
    Normalize specified columns using min-max scaling
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns
    
    normalized_df = dataframe.copy()
    
    for col in columns:
        if col in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[col]):
            min_val = normalized_df[col].min()
            max_val = normalized_df[col].max()
            
            if max_val > min_val:
                normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
            else:
                normalized_df[col] = 0
    
    return normalized_df

def detect_skewed_columns(dataframe, skew_threshold=0.5):
    """
    Identify columns with significant skewness
    """
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
    skewed_cols = []
    
    for col in numeric_cols:
        skewness = dataframe[col].skew()
        if abs(skewness) > skew_threshold:
            skewed_cols.append((col, skewness))
    
    return skewed_cols

def log_transform(dataframe, columns):
    """
    Apply log transformation to specified columns
    """
    transformed_df = dataframe.copy()
    
    for col in columns:
        if col in transformed_df.columns:
            if transformed_df[col].min() <= 0:
                transformed_df[col] = np.log1p(transformed_df[col] - transformed_df[col].min() + 1)
            else:
                transformed_df[col] = np.log(transformed_df[col])
    
    return transformed_df

def clean_dataset(dataframe, outlier_columns=None, normalize=True, handle_skewness=True):
    """
    Comprehensive data cleaning pipeline
    """
    cleaned_df = dataframe.copy()
    
    if outlier_columns is None:
        outlier_columns = cleaned_df.select_dtypes(include=[np.number]).columns
    
    for col in outlier_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    
    if handle_skewness:
        skewed = detect_skewed_columns(cleaned_df)
        skewed_cols = [col for col, _ in skewed]
        if skewed_cols:
            cleaned_df = log_transform(cleaned_df, skewed_cols)
    
    if normalize:
        cleaned_df = normalize_minmax(cleaned_df)
    
    return cleaned_df