
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers using the Interquartile Range method.
    """
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    return filtered_df

def remove_outliers_zscore(dataframe, column, threshold=3):
    """
    Remove outliers using Z-score method.
    """
    z_scores = np.abs(stats.zscore(dataframe[column]))
    filtered_df = dataframe[z_scores < threshold]
    return filtered_df

def normalize_minmax(dataframe, column):
    """
    Normalize data using Min-Max scaling.
    """
    min_val = dataframe[column].min()
    max_val = dataframe[column].max()
    dataframe[column + '_normalized'] = (dataframe[column] - min_val) / (max_val - min_val)
    return dataframe

def normalize_zscore(dataframe, column):
    """
    Normalize data using Z-score standardization.
    """
    mean_val = dataframe[column].mean()
    std_val = dataframe[column].std()
    dataframe[column + '_standardized'] = (dataframe[column] - mean_val) / std_val
    return dataframe

def clean_dataset(dataframe, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    """
    Main function to clean dataset by removing outliers and normalizing numeric columns.
    """
    cleaned_df = dataframe.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            if outlier_method == 'iqr':
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
            elif outlier_method == 'zscore':
                cleaned_df = remove_outliers_zscore(cleaned_df, col)
            
            if normalize_method == 'minmax':
                cleaned_df = normalize_minmax(cleaned_df, col)
            elif normalize_method == 'zscore':
                cleaned_df = normalize_zscore(cleaned_df, col)
    
    return cleaned_df

def validate_dataframe(dataframe, required_columns):
    """
    Validate if dataframe contains all required columns.
    """
    missing_columns = [col for col in required_columns if col not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    return True