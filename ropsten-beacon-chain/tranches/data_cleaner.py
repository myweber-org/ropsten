
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using Interquartile Range method.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to process
    multiplier (float): IQR multiplier
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data.copy()

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to normalize
    
    Returns:
    pd.Series: Normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(data), index=data.index)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def z_score_normalize(data, column):
    """
    Normalize data using Z-score method.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to normalize
    
    Returns:
    pd.Series: Z-score normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    z_scores = (data[column] - mean_val) / std_val
    return z_scores

def detect_skewness(data, column, threshold=0.5):
    """
    Detect skewness in data column.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to check
    threshold (float): Absolute skewness threshold
    
    Returns:
    tuple: (skewness_value, is_skewed)
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    skewness = stats.skew(data[column].dropna())
    is_skewed = abs(skewness) > threshold
    
    return skewness, is_skewed

def create_clean_dataframe(data, numeric_columns, outlier_multiplier=1.5):
    """
    Create cleaned dataframe with outlier removal and normalization.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    numeric_columns (list): List of numeric column names to process
    outlier_multiplier (float): IQR multiplier for outlier removal
    
    Returns:
    pd.DataFrame: Cleaned dataframe with normalized values
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column in cleaned_data.columns:
            original_len = len(cleaned_data)
            cleaned_data = remove_outliers_iqr(cleaned_data, column, outlier_multiplier)
            removed_count = original_len - len(cleaned_data)
            
            if removed_count > 0:
                normalized_col = normalize_minmax(cleaned_data, column)
                cleaned_data[f"{column}_normalized"] = normalized_col
    
    return cleaned_data

def summarize_cleaning(data, cleaned_data, numeric_columns):
    """
    Generate summary statistics for cleaning process.
    
    Parameters:
    data (pd.DataFrame): Original dataframe
    cleaned_data (pd.DataFrame): Cleaned dataframe
    numeric_columns (list): List of processed numeric columns
    
    Returns:
    dict: Summary statistics
    """
    summary = {
        'original_rows': len(data),
        'cleaned_rows': len(cleaned_data),
        'removed_rows': len(data) - len(cleaned_data),
        'removed_percentage': ((len(data) - len(cleaned_data)) / len(data)) * 100,
        'columns_processed': []
    }
    
    for column in numeric_columns:
        if column in data.columns and column in cleaned_data.columns:
            col_summary = {
                'column': column,
                'original_mean': data[column].mean(),
                'cleaned_mean': cleaned_data[column].mean(),
                'original_std': data[column].std(),
                'cleaned_std': cleaned_data[column].std(),
                'skewness_original': stats.skew(data[column].dropna()),
                'skewness_cleaned': stats.skew(cleaned_data[column].dropna())
            }
            summary['columns_processed'].append(col_summary)
    
    return summary

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.uniform(0, 1, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    numeric_cols = ['feature_a', 'feature_b', 'feature_c']
    
    cleaned = create_clean_dataframe(sample_data, numeric_cols)
    summary_stats = summarize_cleaning(sample_data, cleaned, numeric_cols)
    
    print(f"Original data shape: {sample_data.shape}")
    print(f"Cleaned data shape: {cleaned.shape}")
    print(f"Rows removed: {summary_stats['removed_rows']}")
    print(f"Removed percentage: {summary_stats['removed_percentage']:.2f}%")
    
    for col_summary in summary_stats['columns_processed']:
        print(f"\nColumn: {col_summary['column']}")
        print(f"  Original mean: {col_summary['original_mean']:.2f}")
        print(f"  Cleaned mean: {col_summary['cleaned_mean']:.2f}")
        print(f"  Skewness change: {col_summary['skewness_original']:.2f} -> {col_summary['skewness_cleaned']:.2f}")