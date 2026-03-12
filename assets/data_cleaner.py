import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas Series using the IQR method.
    Returns a filtered Series.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    Returns a filtered DataFrame.
    """
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data to range [0, 1] using min-max scaling.
    Returns a new Series with normalized values.
    """
    min_val = data[column].min()
    max_val = data[column].max()
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def normalize_zscore(data, column):
    """
    Standardize data using Z-score normalization (mean=0, std=1).
    Returns a new Series with standardized values.
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(df, numeric_columns, method='iqr', normalize=False):
    """
    Main cleaning function to process multiple numeric columns.
    Supports 'iqr' or 'zscore' outlier removal methods.
    Optionally applies min-max normalization after cleaning.
    Returns a cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col not in cleaned_df.columns:
            continue
            
        if method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        elif method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
        else:
            raise ValueError("Method must be 'iqr' or 'zscore'")
    
    if normalize:
        for col in numeric_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = normalize_minmax(cleaned_df, col)
    
    return cleaned_df.reset_index(drop=True)

def validate_data(df, required_columns, check_missing=True):
    """
    Validate dataset structure and content.
    Returns boolean indicating validity and list of issues.
    """
    issues = []
    
    for col in required_columns:
        if col not in df.columns:
            issues.append(f"Missing required column: {col}")
    
    if check_missing:
        missing_counts = df.isnull().sum()
        for col, count in missing_counts.items():
            if count > 0:
                issues.append(f"Column '{col}' has {count} missing values")
    
    is_valid = len(issues) == 0
    return is_valid, issues