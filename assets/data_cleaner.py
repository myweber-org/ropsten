import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to remove duplicate rows
        fill_missing (bool): Whether to fill missing values
        fill_value: Value to use for filling missing data
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def remove_outliers(df, column, method='iqr', threshold=1.5):
    """
    Remove outliers from a specific column using specified method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to process
        method (str): 'iqr' for interquartile range or 'zscore' for standard deviation
        threshold (float): Threshold for outlier detection
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    elif method == 'zscore':
        mean = df[column].mean()
        std = df[column].std()
        z_scores = (df[column] - mean) / std
        filtered_df = df[abs(z_scores) <= threshold]
    
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    return filtered_df

def normalize_columns(df, columns=None, method='minmax'):
    """
    Normalize specified columns in DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of column names to normalize, None for all numeric columns
        method (str): 'minmax' for min-max scaling or 'standard' for standardization
    
    Returns:
        pd.DataFrame: DataFrame with normalized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    normalized_df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if method == 'minmax':
            col_min = df[col].min()
            col_max = df[col].max()
            if col_max != col_min:
                normalized_df[col] = (df[col] - col_min) / (col_max - col_min)
        
        elif method == 'standard':
            col_mean = df[col].mean()
            col_std = df[col].std()
            if col_std != 0:
                normalized_df[col] = (df[col] - col_mean) / col_std
    
    return normalized_df

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, 5, None, 7],
        'B': [10, 20, 20, 40, 50, 60, 70],
        'C': [100, 200, 300, 400, 500, 600, 700]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df)
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    normalized = normalize_columns(cleaned, method='minmax')
    print("\nNormalized DataFrame:")
    print(normalized)
import pandas as pd
import numpy as np
from scipy import stats

def load_dataset(filepath):
    return pd.read_csv(filepath)

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
    df[column] = (df[column] - min_val) / (max_val - min_val)
    return df

def normalize_zscore(df, column):
    mean_val = df[column].mean()
    std_val = df[column].std()
    df[column] = (df[column] - mean_val) / std_val
    return df

def handle_missing_values(df, strategy='mean'):
    if strategy == 'mean':
        return df.fillna(df.mean())
    elif strategy == 'median':
        return df.fillna(df.median())
    elif strategy == 'mode':
        return df.fillna(df.mode().iloc[0])
    elif strategy == 'drop':
        return df.dropna()
    else:
        raise ValueError("Invalid strategy. Choose from 'mean', 'median', 'mode', or 'drop'.")

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method='minmax', missing_strategy='mean'):
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if outlier_method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        elif outlier_method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
        
        if normalize_method == 'minmax':
            cleaned_df = normalize_minmax(cleaned_df, col)
        elif normalize_method == 'zscore':
            cleaned_df = normalize_zscore(cleaned_df, col)
    
    cleaned_df = handle_missing_values(cleaned_df, strategy=missing_strategy)
    return cleaned_df

def save_cleaned_data(df, output_path):
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    numeric_cols = ['age', 'salary', 'score']
    
    raw_data = load_dataset(input_file)
    cleaned_data = clean_dataset(raw_data, numeric_cols)
    save_cleaned_data(cleaned_data, output_file)
    
    print(f"Original shape: {raw_data.shape}")
    print(f"Cleaned shape: {cleaned_data.shape}")
    print(f"Cleaned data saved to {output_file}")