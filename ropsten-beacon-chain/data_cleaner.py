import numpy as np
import pandas as pd
from scipy import stats

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using IQR method
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def remove_outliers_zscore(data, column, z_threshold=3):
    """
    Remove outliers using Z-score method
    """
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < z_threshold]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    min_val = data[column].min()
    max_val = data[column].max()
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def clean_dataset(df, numeric_columns, outlier_method='zscore', normalize=True):
    """
    Main cleaning function for datasets
    """
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if outlier_method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
        elif outlier_method == 'iqr':
            outliers, _, _ = detect_outliers_iqr(cleaned_df, col)
            cleaned_df = cleaned_df[~cleaned_df.index.isin(outliers.index)]
        
        if normalize:
            cleaned_df[col + '_normalized'] = normalize_minmax(cleaned_df, col)
    
    return cleaned_df

def validate_data(df, required_columns, check_missing=True):
    """
    Validate dataset structure and completeness
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    if check_missing:
        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            print(f"Warning: Dataset contains {missing_values} missing values")
    
    return True

def export_clean_data(df, output_path, format='csv'):
    """
    Export cleaned data to file
    """
    if format == 'csv':
        df.to_csv(output_path, index=False)
    elif format == 'excel':
        df.to_excel(output_path, index=False)
    elif format == 'json':
        df.to_json(output_path, orient='records')
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Data exported successfully to {output_path}")