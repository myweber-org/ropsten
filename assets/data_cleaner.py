
import numpy as np
import pandas as pd
from scipy import stats

def normalize_data(data, method='zscore'):
    if method == 'zscore':
        return (data - np.mean(data)) / np.std(data)
    elif method == 'minmax':
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    else:
        raise ValueError("Method must be 'zscore' or 'minmax'")

def remove_outliers_iqr(data, multiplier=1.5):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    return data[(data >= lower_bound) & (data <= upper_bound)]

def clean_dataset(df, column, normalize=True, remove_outliers=True):
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame")
    
    cleaned_data = df[column].copy()
    
    if remove_outliers:
        cleaned_data = remove_outliers_iqr(cleaned_data)
    
    if normalize:
        cleaned_data = normalize_data(cleaned_data)
    
    return cleaned_data

def process_numerical_columns(df, columns=None, normalize=True, remove_outliers=True):
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    cleaned_df = pd.DataFrame()
    for col in columns:
        try:
            cleaned_df[col] = clean_dataset(df, col, normalize, remove_outliers)
        except Exception as e:
            print(f"Error processing column {col}: {e}")
    
    return cleaned_df