import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(df, columns):
    """
    Remove outliers using the Interquartile Range method.
    Returns a cleaned DataFrame.
    """
    df_clean = df.copy()
    for col in columns:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean.reset_index(drop=True)

def remove_outliers_zscore(df, columns, threshold=3):
    """
    Remove outliers using Z-score method.
    Returns a cleaned DataFrame.
    """
    df_clean = df.copy()
    for col in columns:
        if col in df_clean.columns:
            z_scores = np.abs(stats.zscore(df_clean[col], nan_policy='omit'))
            df_clean = df_clean[z_scores < threshold]
    return df_clean.reset_index(drop=True)

def normalize_minmax(df, columns):
    """
    Normalize specified columns using Min-Max scaling.
    Returns DataFrame with normalized columns.
    """
    df_norm = df.copy()
    for col in columns:
        if col in df_norm.columns:
            min_val = df_norm[col].min()
            max_val = df_norm[col].max()
            if max_val != min_val:
                df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
    return df_norm

def normalize_zscore(df, columns):
    """
    Normalize specified columns using Z-score standardization.
    Returns DataFrame with normalized columns.
    """
    df_norm = df.copy()
    for col in columns:
        if col in df_norm.columns:
            mean_val = df_norm[col].mean()
            std_val = df_norm[col].std()
            if std_val > 0:
                df_norm[col] = (df_norm[col] - mean_val) / std_val
    return df_norm

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    """
    Main function to clean dataset by removing outliers and normalizing.
    """
    if outlier_method == 'iqr':
        df_clean = remove_outliers_iqr(df, numeric_columns)
    elif outlier_method == 'zscore':
        df_clean = remove_outliers_zscore(df, numeric_columns)
    else:
        df_clean = df.copy()

    if normalize_method == 'minmax':
        df_final = normalize_minmax(df_clean, numeric_columns)
    elif normalize_method == 'zscore':
        df_final = normalize_zscore(df_clean, numeric_columns)
    else:
        df_final = df_clean

    return df_final