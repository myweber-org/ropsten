
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    cleaned_df = cleaned_df.reset_index(drop=True)
    return cleaned_df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': np.random.normal(100, 15, 200),
        'B': np.random.exponential(50, 200),
        'C': np.random.uniform(0, 1, 200)
    })
    sample_data.loc[10, 'A'] = 500
    sample_data.loc[20, 'B'] = 1000
    
    numeric_cols = ['A', 'B']
    result = clean_dataset(sample_data, numeric_cols)
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {result.shape}")
    print("Outliers removed successfully.")
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(df, columns=None, factor=1.5):
    """
    Remove outliers using IQR method
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
        df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def remove_outliers_zscore(df, columns=None, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    for col in columns:
        z_scores = np.abs(stats.zscore(df[col].dropna()))
        mask = z_scores < threshold
        df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def normalize_minmax(df, columns=None):
    """
    Normalize data using Min-Max scaling
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_normalized = df.copy()
    for col in columns:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val != min_val:
            df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
    
    return df_normalized

def normalize_zscore(df, columns=None):
    """
    Normalize data using Z-score standardization
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_normalized = df.copy()
    for col in columns:
        mean_val = df[col].mean()
        std_val = df[col].std()
        if std_val != 0:
            df_normalized[col] = (df[col] - mean_val) / std_val
    
    return df_normalized

def clean_dataset(df, outlier_method='iqr', normalize_method='minmax', 
                  outlier_params=None, normalize_params=None):
    """
    Main function to clean dataset with outlier removal and normalization
    """
    if outlier_params is None:
        outlier_params = {}
    if normalize_params is None:
        normalize_params = {}
    
    # Remove outliers
    if outlier_method == 'iqr':
        df_clean = remove_outliers_iqr(df, **outlier_params)
    elif outlier_method == 'zscore':
        df_clean = remove_outliers_zscore(df, **outlier_params)
    else:
        df_clean = df.copy()
    
    # Normalize data
    if normalize_method == 'minmax':
        df_final = normalize_minmax(df_clean, **normalize_params)
    elif normalize_method == 'zscore':
        df_final = normalize_zscore(df_clean, **normalize_params)
    else:
        df_final = df_clean
    
    return df_final

def get_cleaning_summary(original_df, cleaned_df):
    """
    Generate summary statistics before and after cleaning
    """
    summary = {
        'original_rows': len(original_df),
        'cleaned_rows': len(cleaned_df),
        'rows_removed': len(original_df) - len(cleaned_df),
        'removal_percentage': (len(original_df) - len(cleaned_df)) / len(original_df) * 100
    }
    
    numeric_cols = original_df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        summary[f'{col}_original_mean'] = original_df[col].mean()
        summary[f'{col}_cleaned_mean'] = cleaned_df[col].mean()
        summary[f'{col}_original_std'] = original_df[col].std()
        summary[f'{col}_cleaned_std'] = cleaned_df[col].std()
    
    return summary