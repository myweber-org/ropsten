
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

def normalize_column_zscore(dataframe, column):
    """
    Normalize a column using Z-score normalization.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    column (str): Column name to normalize
    
    Returns:
    pd.DataFrame: DataFrame with normalized column
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    result_df = dataframe.copy()
    mean_val = result_df[column].mean()
    std_val = result_df[column].std()
    
    if std_val > 0:
        result_df[f'{column}_normalized'] = (result_df[column] - mean_val) / std_val
    else:
        result_df[f'{column}_normalized'] = 0
    
    return result_df

def winsorize_column(dataframe, column, limits=(0.05, 0.05)):
    """
    Apply winsorization to a column to reduce outlier impact.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    column (str): Column name to winsorize
    limits (tuple): Lower and upper percentile limits
    
    Returns:
    pd.DataFrame: DataFrame with winsorized column
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    result_df = dataframe.copy()
    winsorized_data = stats.mstats.winsorize(result_df[column], limits=limits)
    result_df[f'{column}_winsorized'] = winsorized_data
    
    return result_df

def clean_dataset(dataframe, numeric_columns, outlier_threshold=1.5, normalize=True):
    """
    Comprehensive data cleaning pipeline.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of numeric column names to process
    outlier_threshold (float): IQR threshold for outlier removal
    normalize (bool): Whether to apply Z-score normalization
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    cleaned_df = dataframe.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns:
            # Remove outliers
            cleaned_df = remove_outliers_iqr(cleaned_df, column, outlier_threshold)
            
            # Apply normalization if requested
            if normalize:
                cleaned_df = normalize_column_zscore(cleaned_df, column)
    
    return cleaned_df.reset_index(drop=True)

def validate_dataframe(dataframe, required_columns):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    dataframe (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    dict: Validation results with status and messages
    """
    validation_result = {
        'is_valid': True,
        'missing_columns': [],
        'empty_columns': [],
        'messages': []
    }
    
    # Check required columns
    for column in required_columns:
        if column not in dataframe.columns:
            validation_result['missing_columns'].append(column)
            validation_result['is_valid'] = False
    
    # Check for empty columns
    for column in dataframe.columns:
        if dataframe[column].isnull().all():
            validation_result['empty_columns'].append(column)
            validation_result['is_valid'] = False
    
    if validation_result['missing_columns']:
        validation_result['messages'].append(
            f"Missing columns: {validation_result['missing_columns']}"
        )
    
    if validation_result['empty_columns']:
        validation_result['messages'].append(
            f"Empty columns: {validation_result['empty_columns']}"
        )
    
    return validation_result