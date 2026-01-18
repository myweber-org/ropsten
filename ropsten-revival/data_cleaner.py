
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
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data.copy()

def zscore_normalize(data, column):
    """
    Normalize data using Z-score normalization.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        DataFrame with normalized column added as '{column}_normalized'
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    normalized_col = f"{column}_normalized"
    data[normalized_col] = stats.zscore(data[column])
    return data

def minmax_normalize(data, column, feature_range=(0, 1)):
    """
    Normalize data using Min-Max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
        feature_range: tuple of (min, max) for scaled range
    
    Returns:
        DataFrame with normalized column added as '{column}_scaled'
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        raise ValueError(f"Column '{column}' has constant values")
    
    scaled_col = f"{column}_scaled"
    data_min, data_max = feature_range
    
    data[scaled_col] = (data[column] - min_val) / (max_val - min_val)
    data[scaled_col] = data[scaled_col] * (data_max - data_min) + data_min
    
    return data

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in specified columns.
    
    Args:
        data: pandas DataFrame
        strategy: imputation strategy ('mean', 'median', 'mode', 'constant')
        columns: list of columns to process (None processes all numeric columns)
    
    Returns:
        DataFrame with missing values handled
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    data_clean = data.copy()
    
    for col in columns:
        if col not in data.columns:
            continue
            
        if data[col].isnull().any():
            if strategy == 'mean':
                fill_value = data[col].mean()
            elif strategy == 'median':
                fill_value = data[col].median()
            elif strategy == 'mode':
                fill_value = data[col].mode()[0]
            elif strategy == 'constant':
                fill_value = 0
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            data_clean[col] = data[col].fillna(fill_value)
    
    return data_clean

def clean_dataset(data, numeric_columns=None, outlier_factor=1.5, normalize_method='zscore'):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric columns to process
        outlier_factor: IQR factor for outlier removal
        normalize_method: normalization method ('zscore' or 'minmax')
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    # Handle missing values
    cleaned_data = handle_missing_values(cleaned_data, strategy='mean', columns=numeric_columns)
    
    # Remove outliers for each numeric column
    for col in numeric_columns:
        if col in cleaned_data.columns:
            cleaned_data = remove_outliers_iqr(cleaned_data, col, factor=outlier_factor)
    
    # Normalize numeric columns
    for col in numeric_columns:
        if col in cleaned_data.columns:
            if normalize_method == 'zscore':
                cleaned_data = zscore_normalize(cleaned_data, col)
            elif normalize_method == 'minmax':
                cleaned_data = minmax_normalize(cleaned_data, col)
    
    return cleaned_data

def validate_data(data, required_columns=None, numeric_columns=None):
    """
    Validate data quality and structure.
    
    Args:
        data: pandas DataFrame
        required_columns: list of columns that must be present
        numeric_columns: list of columns that must be numeric
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'is_valid': True,
        'missing_columns': [],
        'non_numeric_columns': [],
        'missing_values': {},
        'constant_columns': []
    }
    
    # Check required columns
    if required_columns:
        for col in required_columns:
            if col not in data.columns:
                validation_results['missing_columns'].append(col)
                validation_results['is_valid'] = False
    
    # Check numeric columns
    if numeric_columns:
        for col in numeric_columns:
            if col in data.columns:
                if not np.issubdtype(data[col].dtype, np.number):
                    validation_results['non_numeric_columns'].append(col)
                    validation_results['is_valid'] = False
                
                # Check for constant values
                if data[col].nunique() == 1:
                    validation_results['constant_columns'].append(col)
    
    # Check for missing values
    for col in data.columns:
        missing_count = data[col].isnull().sum()
        if missing_count > 0:
            validation_results['missing_values'][col] = missing_count
    
    return validation_results
def remove_duplicates(input_list):
    """
    Remove duplicate items from a list while preserving order.
    Returns a new list with unique elements.
    """
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def clean_numeric_strings(string_list):
    """
    Clean a list of numeric strings by converting to integers,
    removing invalid entries, and returning sorted unique values.
    """
    cleaned = []
    for s in string_list:
        try:
            num = int(s.strip())
            cleaned.append(num)
        except ValueError:
            continue
    return sorted(set(cleaned))

if __name__ == "__main__":
    # Example usage
    sample_data = [1, 2, 2, 3, 4, 4, 5, 1, 6]
    cleaned = remove_duplicates(sample_data)
    print(f"Original: {sample_data}")
    print(f"Cleaned: {cleaned}")
    
    numeric_strings = ["10", "5", "abc", "20", "5", "15", "invalid"]
    cleaned_nums = clean_numeric_strings(numeric_strings)
    print(f"\nNumeric strings: {numeric_strings}")
    print(f"Cleaned numbers: {cleaned_nums}")