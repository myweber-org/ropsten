
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        factor: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
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
    Remove outliers using Z-score method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        threshold: Z-score threshold (default 3)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    
    if len(z_scores) != len(data):
        valid_indices = data[column].dropna().index
        mask = pd.Series(False, index=data.index)
        mask.loc[valid_indices] = z_scores < threshold
        return data[mask]
    
    return data[z_scores < threshold]

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        DataFrame with normalized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    result = data.copy()
    min_val = result[column].min()
    max_val = result[column].max()
    
    if max_val == min_val:
        result[column] = 0.5
    else:
        result[column] = (result[column] - min_val) / (max_val - min_val)
    
    return result

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        DataFrame with standardized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    result = data.copy()
    mean_val = result[column].mean()
    std_val = result[column].std()
    
    if std_val == 0:
        result[column] = 0
    else:
        result[column] = (result[column] - mean_val) / std_val
    
    return result

def clean_dataset(data, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric columns to process (default: all numeric)
        outlier_method: 'iqr', 'zscore', or None
        normalize_method: 'minmax', 'zscore', or None
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    result = data.copy()
    
    # Remove outliers
    if outlier_method == 'iqr':
        for col in numeric_columns:
            if col in result.columns:
                result = remove_outliers_iqr(result, col)
    elif outlier_method == 'zscore':
        for col in numeric_columns:
            if col in result.columns:
                result = remove_outliers_zscore(result, col)
    
    # Normalize data
    if normalize_method == 'minmax':
        for col in numeric_columns:
            if col in result.columns:
                result = normalize_minmax(result, col)
    elif normalize_method == 'zscore':
        for col in numeric_columns:
            if col in result.columns:
                result = normalize_zscore(result, col)
    
    return result

def validate_data(data, require_columns=None, allow_nan_ratio=0.3):
    """
    Validate data quality.
    
    Args:
        data: pandas DataFrame
        require_columns: list of required columns
        allow_nan_ratio: maximum allowed NaN ratio per column
    
    Returns:
        tuple: (is_valid, issues)
    """
    issues = []
    
    if require_columns:
        missing_columns = [col for col in require_columns if col not in data.columns]
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
    
    # Check for excessive NaN values
    for col in data.columns:
        nan_ratio = data[col].isna().mean()
        if nan_ratio > allow_nan_ratio:
            issues.append(f"Column '{col}' has {nan_ratio:.1%} NaN values")
    
    # Check for constant columns
    for col in data.select_dtypes(include=[np.number]).columns:
        if data[col].nunique() == 1:
            issues.append(f"Column '{col}' has constant values")
    
    return len(issues) == 0, issues
def remove_duplicates(input_list):
    """
    Remove duplicate elements from a list while preserving order.
    Returns a new list with unique elements.
    """
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def clean_data_with_threshold(data, threshold=None):
    """
    Clean data by removing duplicates and optionally filtering by threshold.
    If threshold is provided, only items meeting the threshold are kept.
    """
    unique_data = remove_duplicates(data)
    
    if threshold is not None:
        filtered_data = [item for item in unique_data if item >= threshold]
        return filtered_data
    
    return unique_data

if __name__ == "__main__":
    sample_data = [1, 3, 2, 1, 5, 3, 7, 2, 8]
    print("Original data:", sample_data)
    print("Cleaned data:", remove_duplicates(sample_data))
    print("Cleaned with threshold 4:", clean_data_with_threshold(sample_data, 4))