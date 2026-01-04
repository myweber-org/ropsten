
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, multiplier=1.5):
    """
    Remove outliers from a DataFrame column using IQR method.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    multiplier (float): IQR multiplier for outlier detection
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    return filtered_df.copy()

def zscore_normalize(dataframe, column):
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

def minmax_normalize(dataframe, column, feature_range=(0, 1)):
    """
    Normalize a column using Min-Max scaling.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    column (str): Column name to normalize
    feature_range (tuple): Desired range of transformed data
    
    Returns:
    pd.DataFrame: DataFrame with normalized column
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    result_df = dataframe.copy()
    min_val = result_df[column].min()
    max_val = result_df[column].max()
    
    if max_val > min_val:
        scaled = (result_df[column] - min_val) / (max_val - min_val)
        result_df[f'{column}_scaled'] = scaled * (feature_range[1] - feature_range[0]) + feature_range[0]
    else:
        result_df[f'{column}_scaled'] = feature_range[0]
    
    return result_df

def detect_skewed_columns(dataframe, threshold=0.5):
    """
    Detect columns with skewed distributions.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    threshold (float): Absolute skewness threshold
    
    Returns:
    dict: Dictionary with column names and their skewness values
    """
    skewed_cols = {}
    
    for col in dataframe.select_dtypes(include=[np.number]).columns:
        skewness = dataframe[col].skew()
        if abs(skewness) > threshold:
            skewed_cols[col] = skewness
    
    return skewed_cols

def log_transform(dataframe, column):
    """
    Apply log transformation to reduce skewness.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    column (str): Column name to transform
    
    Returns:
    pd.DataFrame: DataFrame with transformed column
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    result_df = dataframe.copy()
    
    if result_df[column].min() <= 0:
        offset = abs(result_df[column].min()) + 1
        result_df[f'{column}_log'] = np.log(result_df[column] + offset)
    else:
        result_df[f'{column}_log'] = np.log(result_df[column])
    
    return result_df