
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    outliers_removed = len(data) - len(filtered_data)
    
    return filtered_data, outliers_removed

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    outliers_removed = len(data) - len(filtered_data)
    
    return filtered_data, outliers_removed

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

def clean_dataset(data, numeric_columns, outlier_method='iqr', normalization_method='zscore'):
    """
    Comprehensive data cleaning pipeline
    """
    cleaned_data = data.copy()
    cleaning_report = {}
    
    for col in numeric_columns:
        if col not in cleaned_data.columns:
            continue
            
        original_count = len(cleaned_data)
        
        if outlier_method == 'iqr':
            cleaned_data, outliers_removed = remove_outliers_iqr(cleaned_data, col)
        elif outlier_method == 'zscore':
            cleaned_data, outliers_removed = remove_outliers_zscore(cleaned_data, col)
        else:
            outliers_removed = 0
        
        if normalization_method == 'minmax':
            cleaned_data[f'{col}_normalized'] = normalize_minmax(cleaned_data, col)
        elif normalization_method == 'zscore':
            cleaned_data[f'{col}_normalized'] = normalize_zscore(cleaned_data, col)
        
        cleaning_report[col] = {
            'original_rows': original_count,
            'cleaned_rows': len(cleaned_data),
            'outliers_removed': outliers_removed,
            'outlier_percentage': (outliers_removed / original_count * 100) if original_count > 0 else 0
        }
    
    return cleaned_data, cleaning_report

def validate_data(data, required_columns, numeric_columns):
    """
    Validate data structure and content
    """
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    validation_report = {
        'total_rows': len(data),
        'total_columns': len(data.columns),
        'missing_values': data.isnull().sum().to_dict(),
        'numeric_stats': {}
    }
    
    for col in numeric_columns:
        if col in data.columns:
            validation_report['numeric_stats'][col] = {
                'mean': data[col].mean(),
                'std': data[col].std(),
                'min': data[col].min(),
                'max': data[col].max(),
                'median': data[col].median()
            }
    
    return validation_report

def example_usage():
    """
    Example usage of the data cleaning utilities
    """
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'id': range(100),
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })
    
    print("Original data shape:", sample_data.shape)
    
    validation_report = validate_data(
        sample_data, 
        required_columns=['id', 'feature_a', 'feature_b', 'category'],
        numeric_columns=['feature_a', 'feature_b']
    )
    
    print("\nValidation Report:")
    for key, value in validation_report.items():
        if key != 'numeric_stats':
            print(f"{key}: {value}")
    
    cleaned_data, cleaning_report = clean_dataset(
        sample_data,
        numeric_columns=['feature_a', 'feature_b'],
        outlier_method='iqr',
        normalization_method='zscore'
    )
    
    print(f"\nCleaned data shape: {cleaned_data.shape}")
    print("\nCleaning Report:")
    for col, report in cleaning_report.items():
        print(f"{col}: {report}")

if __name__ == "__main__":
    example_usage()