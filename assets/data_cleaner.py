import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def remove_outliers_zscore(df, column, threshold=3):
    z_scores = np.abs(stats.zscore(df[column]))
    return df[z_scores < threshold]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def normalize_zscore(df, column):
    mean_val = df[column].mean()
    std_val = df[column].std()
    df[column + '_standardized'] = (df[column] - mean_val) / std_val
    return df

def clean_dataset(df, numeric_columns, method='iqr', normalize=True):
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        elif method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
    
    if normalize:
        for col in numeric_columns:
            cleaned_df = normalize_zscore(cleaned_df, col)
    
    return cleaned_df

def summarize_cleaning(df_before, df_after, numeric_columns):
    summary = {}
    for col in numeric_columns:
        summary[col] = {
            'original_count': len(df_before),
            'cleaned_count': len(df_after),
            'removed_percentage': ((len(df_before) - len(df_after)) / len(df_before)) * 100,
            'original_mean': df_before[col].mean(),
            'cleaned_mean': df_after[col].mean(),
            'original_std': df_before[col].std(),
            'cleaned_std': df_after[col].std()
        }
    return pd.DataFrame(summary).Timport numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def standardize_zscore(df, column):
    mean_val = df[column].mean()
    std_val = df[column].std()
    df[column + '_standardized'] = (df[column] - mean_val) / std_val
    return df

def handle_missing_values(df, strategy='mean'):
    if strategy == 'mean':
        return df.fillna(df.mean())
    elif strategy == 'median':
        return df.fillna(df.median())
    elif strategy == 'mode':
        return df.fillna(df.mode().iloc[0])
    else:
        return df.dropna()

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = normalize_minmax(cleaned_df, col)
    cleaned_df = handle_missing_values(cleaned_df, strategy='mean')
    return cleaned_df
import pandas as pd

def clean_dataset(df, id_column='id'):
    """
    Remove duplicate rows based on an ID column and standardize column names.
    """
    if df.empty:
        return df

    df_clean = df.copy()

    if id_column in df_clean.columns:
        df_clean = df_clean.drop_duplicates(subset=[id_column], keep='first')
    else:
        df_clean = df_clean.drop_duplicates()

    df_clean.columns = df_clean.columns.str.strip().str.lower().str.replace(' ', '_')
    df_clean = df_clean.reset_index(drop=True)

    return df_clean

def validate_numeric_columns(df, numeric_columns):
    """
    Ensure specified columns contain only numeric values, coercing errors to NaN.
    """
    df_valid = df.copy()
    for col in numeric_columns:
        if col in df_valid.columns:
            df_valid[col] = pd.to_numeric(df_valid[col], errors='coerce')
    return df_valid

if __name__ == "__main__":
    sample_data = {
        'ID': [1, 2, 2, 3, 4],
        'Customer Name': ['Alice', 'Bob', 'Bob', 'Charlie', 'David'],
        'Total Amount': ['100', '150', '150', '200', 'two-fifty']
    }
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)

    cleaned_df = clean_dataset(df, id_column='ID')
    cleaned_df = validate_numeric_columns(cleaned_df, ['Total Amount'])
    print("\nCleaned DataFrame:")
    print(cleaned_df)