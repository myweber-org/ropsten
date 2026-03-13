
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
    
    if min_val == max_val:
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
    
    normalized = (data[column] - mean_val) / std_val
    return normalized

def clean_dataset(data, numeric_columns, outlier_method='iqr', normalize_method='zscore'):
    """
    Main function to clean dataset by removing outliers and normalizing numeric columns
    """
    cleaned_data = data.copy()
    stats_report = {}
    
    for col in numeric_columns:
        if col not in cleaned_data.columns:
            print(f"Warning: Column '{col}' not found, skipping")
            continue
        
        original_count = len(cleaned_data)
        
        if outlier_method == 'iqr':
            cleaned_data, outliers_removed = remove_outliers_iqr(cleaned_data, col)
        elif outlier_method == 'zscore':
            cleaned_data, outliers_removed = remove_outliers_zscore(cleaned_data, col)
        else:
            raise ValueError("outlier_method must be 'iqr' or 'zscore'")
        
        if normalize_method == 'minmax':
            cleaned_data[f"{col}_normalized"] = normalize_minmax(cleaned_data, col)
        elif normalize_method == 'zscore':
            cleaned_data[f"{col}_normalized"] = normalize_zscore(cleaned_data, col)
        else:
            raise ValueError("normalize_method must be 'minmax' or 'zscore'")
        
        stats_report[col] = {
            'original_samples': original_count,
            'cleaned_samples': len(cleaned_data),
            'outliers_removed': outliers_removed,
            'outlier_percentage': (outliers_removed / original_count) * 100
        }
    
    return cleaned_data, stats_report

def generate_cleaning_summary(stats_report):
    """
    Generate a summary report of the cleaning process
    """
    summary = []
    summary.append("Data Cleaning Summary")
    summary.append("=" * 50)
    
    total_original = 0
    total_cleaned = 0
    total_outliers = 0
    
    for col, stats in stats_report.items():
        summary.append(f"Column: {col}")
        summary.append(f"  Original samples: {stats['original_samples']}")
        summary.append(f"  Cleaned samples: {stats['cleaned_samples']}")
        summary.append(f"  Outliers removed: {stats['outliers_removed']}")
        summary.append(f"  Outlier percentage: {stats['outlier_percentage']:.2f}%")
        summary.append("-" * 30)
        
        total_original += stats['original_samples']
        total_cleaned += stats['cleaned_samples']
        total_outliers += stats['outliers_removed']
    
    if stats_report:
        avg_outlier_percentage = (total_outliers / total_original) * 100
        summary.append(f"\nOverall Statistics:")
        summary.append(f"Total original samples: {total_original}")
        summary.append(f"Total cleaned samples: {total_cleaned}")
        summary.append(f"Total outliers removed: {total_outliers}")
        summary.append(f"Average outlier percentage: {avg_outlier_percentage:.2f}%")
    
    return "\n".join(summary)