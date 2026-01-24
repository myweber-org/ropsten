import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(df, columns):
    """
    Remove outliers using the Interquartile Range method.
    """
    cleaned_df = df.copy()
    for col in columns:
        if col in cleaned_df.columns:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
    return cleaned_df

def remove_outliers_zscore(df, columns, threshold=3):
    """
    Remove outliers using Z-score method.
    """
    cleaned_df = df.copy()
    for col in columns:
        if col in cleaned_df.columns:
            z_scores = np.abs(stats.zscore(cleaned_df[col]))
            cleaned_df = cleaned_df[z_scores < threshold]
    return cleaned_df

def normalize_minmax(df, columns):
    """
    Normalize specified columns using Min-Max scaling.
    """
    normalized_df = df.copy()
    for col in columns:
        if col in normalized_df.columns:
            min_val = normalized_df[col].min()
            max_val = normalized_df[col].max()
            normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
    return normalized_df

def normalize_zscore(df, columns):
    """
    Normalize specified columns using Z-score standardization.
    """
    normalized_df = df.copy()
    for col in columns:
        if col in normalized_df.columns:
            mean_val = normalized_df[col].mean()
            std_val = normalized_df[col].std()
            normalized_df[col] = (normalized_df[col] - mean_val) / std_val
    return normalized_df

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in specified columns.
    """
    filled_df = df.copy()
    if columns is None:
        columns = filled_df.columns
    
    for col in columns:
        if col in filled_df.columns and filled_df[col].isnull().any():
            if strategy == 'mean':
                fill_value = filled_df[col].mean()
            elif strategy == 'median':
                fill_value = filled_df[col].median()
            elif strategy == 'mode':
                fill_value = filled_df[col].mode()[0]
            elif strategy == 'drop':
                filled_df = filled_df.dropna(subset=[col])
                continue
            else:
                fill_value = 0
            
            filled_df[col] = filled_df[col].fillna(fill_value)
    
    return filled_df

def clean_dataset(df, outlier_method='iqr', normalize_method='minmax', missing_strategy='mean'):
    """
    Comprehensive data cleaning pipeline.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if outlier_method == 'iqr':
        df_cleaned = remove_outliers_iqr(df, numeric_cols)
    elif outlier_method == 'zscore':
        df_cleaned = remove_outliers_zscore(df, numeric_cols)
    else:
        df_cleaned = df.copy()
    
    df_cleaned = handle_missing_values(df_cleaned, strategy=missing_strategy, columns=numeric_cols)
    
    if normalize_method == 'minmax':
        df_cleaned = normalize_minmax(df_cleaned, numeric_cols)
    elif normalize_method == 'zscore':
        df_cleaned = normalize_zscore(df_cleaned, numeric_cols)
    
    return df_cleaned