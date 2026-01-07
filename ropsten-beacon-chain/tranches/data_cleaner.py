
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
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    return filtered_data

def normalize_minmax(data, columns=None):
    """
    Normalize data using min-max scaling
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    normalized_data = data.copy()
    for col in columns:
        if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
            min_val = data[col].min()
            max_val = data[col].max()
            if max_val != min_val:
                normalized_data[col] = (data[col] - min_val) / (max_val - min_val)
    
    return normalized_data

def normalize_zscore(data, columns=None):
    """
    Normalize data using Z-score standardization
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    standardized_data = data.copy()
    for col in columns:
        if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
            mean_val = data[col].mean()
            std_val = data[col].std()
            if std_val != 0:
                standardized_data[col] = (data[col] - mean_val) / std_val
    
    return standardized_data

def clean_dataset(data, outlier_method='iqr', normalization_method='minmax', 
                  outlier_columns=None, norm_columns=None, outlier_params=None):
    """
    Main function to clean dataset with outlier removal and normalization
    """
    cleaned_data = data.copy()
    
    if outlier_columns is None:
        outlier_columns = data.select_dtypes(include=[np.number]).columns
    
    if outlier_method == 'iqr':
        factor = outlier_params.get('factor', 1.5) if outlier_params else 1.5
        for col in outlier_columns:
            if col in data.columns:
                cleaned_data = remove_outliers_iqr(cleaned_data, col, factor)
    elif outlier_method == 'zscore':
        threshold = outlier_params.get('threshold', 3) if outlier_params else 3
        for col in outlier_columns:
            if col in data.columns:
                cleaned_data = remove_outliers_zscore(cleaned_data, col, threshold)
    
    if normalization_method == 'minmax':
        cleaned_data = normalize_minmax(cleaned_data, norm_columns)
    elif normalization_method == 'zscore':
        cleaned_data = normalize_zscore(cleaned_data, norm_columns)
    
    return cleaned_data

def validate_data(data, check_missing=True, check_duplicates=True):
    """
    Validate data quality
    """
    validation_report = {}
    
    if check_missing:
        missing_values = data.isnull().sum()
        missing_percentage = (missing_values / len(data)) * 100
        validation_report['missing_values'] = missing_values[missing_values > 0]
        validation_report['missing_percentage'] = missing_percentage[missing_percentage > 0]
    
    if check_duplicates:
        duplicate_count = data.duplicated().sum()
        validation_report['duplicate_rows'] = duplicate_count
    
    return validation_report
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column (str): Column name to process.
    
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
    
    return filtered_df

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column (str): Column name to analyze.
    
    Returns:
    dict: Dictionary containing summary statistics.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def main():
    """
    Example usage of the data cleaning functions.
    """
    np.random.seed(42)
    data = pd.DataFrame({
        'values': np.concatenate([
            np.random.normal(100, 15, 1000),
            np.random.normal(300, 50, 50)
        ])
    })
    
    print("Original data shape:", data.shape)
    print("Original statistics:", calculate_summary_statistics(data, 'values'))
    
    cleaned_data = remove_outliers_iqr(data, 'values')
    
    print("\nCleaned data shape:", cleaned_data.shape)
    print("Cleaned statistics:", calculate_summary_statistics(cleaned_data, 'values'))
    
    return cleaned_data

if __name__ == "__main__":
    cleaned = main()