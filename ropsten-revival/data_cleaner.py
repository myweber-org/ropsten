
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def clean_dataset(df, columns_to_clean=None):
    """
    Clean multiple columns in a DataFrame using IQR method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns_to_clean (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns_to_clean is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns_to_clean = list(numeric_cols)
    
    cleaned_df = df.copy()
    
    for column in columns_to_clean:
        if column in cleaned_df.columns:
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, column)
            except Exception as e:
                print(f"Error cleaning column {column}: {e}")
    
    return cleaned_df

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    print(f"Original shape: {df.shape}")
    
    cleaned_df = clean_dataset(df, ['A', 'B'])
    print(f"Cleaned shape: {cleaned_df.shape}")
    print(f"Removed {len(df) - len(cleaned_df)} outliers")
import numpy as np

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def calculate_summary_statistics(data, column):
    mean_val = data[column].mean()
    median_val = data[column].median()
    std_val = data[column].std()
    return {
        'mean': mean_val,
        'median': median_val,
        'std': std_val
    }import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def zscore_normalize(data, column):
    """
    Normalize data using z-score normalization
    """
    mean = data[column].mean()
    std = data[column].std()
    data[column + '_normalized'] = (data[column] - mean) / std
    return data

def minmax_normalize(data, column, feature_range=(0, 1)):
    """
    Normalize data using min-max scaling
    """
    min_val = data[column].min()
    max_val = data[column].max()
    a, b = feature_range
    data[column + '_scaled'] = a + ((data[column] - min_val) * (b - a)) / (max_val - min_val)
    return data

def detect_missing_patterns(data, threshold=0.3):
    """
    Detect columns with high percentage of missing values
    """
    missing_percent = data.isnull().sum() / len(data)
    high_missing_cols = missing_percent[missing_percent > threshold].index.tolist()
    return high_missing_cols

def clean_dataset(data, numeric_columns, outlier_factor=1.5, normalize_method='zscore'):
    """
    Main cleaning pipeline for numeric columns
    """
    cleaned_data = data.copy()
    
    for col in numeric_columns:
        if col in cleaned_data.columns:
            cleaned_data = remove_outliers_iqr(cleaned_data, col, outlier_factor)
            
            if normalize_method == 'zscore':
                cleaned_data = zscore_normalize(cleaned_data, col)
            elif normalize_method == 'minmax':
                cleaned_data = minmax_normalize(cleaned_data, col)
    
    high_missing = detect_missing_patterns(cleaned_data)
    if high_missing:
        print(f"Warning: Columns with >30% missing values: {high_missing}")
    
    return cleaned_data

def validate_data(data, required_columns):
    """
    Validate that required columns exist and have valid data
    """
    missing_cols = [col for col in required_columns if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    for col in required_columns:
        if data[col].isnull().all():
            raise ValueError(f"Column {col} contains only null values")
    
    return True