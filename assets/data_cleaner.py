
import pandas as pd

def clean_dataset(df):
    """
    Clean the dataset by removing null values and duplicate rows.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to be cleaned.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    # Remove rows with any null values
    df_cleaned = df.dropna()
    
    # Remove duplicate rows
    df_cleaned = df_cleaned.drop_duplicates()
    
    # Reset index after cleaning
    df_cleaned = df_cleaned.reset_index(drop=True)
    
    return df_cleaned

def filter_by_threshold(df, column, threshold):
    """
    Filter rows where the specified column value is above a threshold.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column (str): Column name to apply the threshold.
    threshold (float): Threshold value.
    
    Returns:
    pd.DataFrame: Filtered DataFrame.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    filtered_df = df[df[column] > threshold]
    return filtered_df

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, None, 4, 4, 5],
        'B': [10, 20, 30, 40, 40, 50],
        'C': [100, 200, 300, 400, 400, 500]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_dataset(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    filtered_df = filter_by_threshold(cleaned_df, 'B', 25)
    print("\nFiltered DataFrame (B > 25):")
    print(filtered_df)
def remove_duplicates_preserve_order(seq):
    seen = set()
    result = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd
import numpy as np
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

def clean_dataset(file_path):
    df = pd.read_csv(file_path)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        df = remove_outliers_iqr(df, col)
    
    for col in numeric_columns:
        df = normalize_minmax(df, col)
    
    df = df.dropna()
    
    return df

def save_cleaned_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    try:
        cleaned_df = clean_dataset(input_file)
        save_cleaned_data(cleaned_df, output_file)
        print(f"Original shape: {pd.read_csv(input_file).shape}")
        print(f"Cleaned shape: {cleaned_df.shape}")
    except FileNotFoundError:
        print(f"Error: File {input_file} not found")
    except Exception as e:
        print(f"Error during cleaning: {str(e)}")
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
def remove_duplicates(data_list):
    seen = set()
    unique_data = []
    for item in data_list:
        if item not in seen:
            seen.add(item)
            unique_data.append(item)
    return unique_data

def clean_data_with_key(data_list, key_func=None):
    if key_func is None:
        return remove_duplicates(data_list)
    seen = set()
    unique_data = []
    for item in data_list:
        key = key_func(item)
        if key not in seen:
            seen.add(key)
            unique_data.append(item)
    return unique_data

if __name__ == "__main__":
    sample_data = [1, 2, 2, 3, 4, 4, 5]
    print(remove_duplicates(sample_data))
    
    sample_objects = [{"id": 1}, {"id": 2}, {"id": 1}, {"id": 3}]
    print(clean_data_with_key(sample_objects, key_func=lambda x: x["id"]))
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
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
    
    return filtered_df.reset_index(drop=True)

def calculate_statistics(df, column):
    """
    Calculate basic statistics for a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def normalize_column(df, column, method='minmax'):
    """
    Normalize a DataFrame column using specified method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to normalize
    method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
    pd.DataFrame: DataFrame with normalized column
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_copy = df.copy()
    
    if method == 'minmax':
        min_val = df_copy[column].min()
        max_val = df_copy[column].max()
        if max_val != min_val:
            df_copy[f'{column}_normalized'] = (df_copy[column] - min_val) / (max_val - min_val)
        else:
            df_copy[f'{column}_normalized'] = 0.5
    
    elif method == 'zscore':
        mean_val = df_copy[column].mean()
        std_val = df_copy[column].std()
        if std_val > 0:
            df_copy[f'{column}_normalized'] = (df_copy[column] - mean_val) / std_val
        else:
            df_copy[f'{column}_normalized'] = 0
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    return df_copy

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'values': [10, 12, 12, 13, 12, 11, 14, 13, 15, 100, 12, 11, 10, 9, 8, 12, 13, 14]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nOriginal Statistics:")
    print(calculate_statistics(df, 'values'))
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print("\nCleaned Statistics:")
    print(calculate_statistics(cleaned_df, 'values'))
    
    normalized_df = normalize_column(cleaned_df, 'values', method='minmax')
    print("\nNormalized DataFrame:")
    print(normalized_df)
import pandas as pd
import numpy as np

def clean_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        strategy (str): Strategy for imputation ('mean', 'median', 'mode', 'drop')
        columns (list): List of columns to process, None for all columns
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        columns = df.columns
    
    df_clean = df.copy()
    
    for col in columns:
        if df[col].isnull().any():
            if strategy == 'mean':
                df_clean[col].fillna(df[col].mean(), inplace=True)
            elif strategy == 'median':
                df_clean[col].fillna(df[col].median(), inplace=True)
            elif strategy == 'mode':
                df_clean[col].fillna(df[col].mode()[0], inplace=True)
            elif strategy == 'drop':
                df_clean = df_clean.dropna(subset=[col])
    
    return df_clean

def remove_outliers_iqr(df, columns=None, threshold=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of columns to process, None for all numeric columns
        threshold (float): IQR multiplier for outlier detection
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    return df_clean

def normalize_data(df, columns=None, method='minmax'):
    """
    Normalize data to a common scale.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of columns to normalize, None for all numeric columns
        method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
        pd.DataFrame: Normalized DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_norm = df.copy()
    
    for col in columns:
        if method == 'minmax':
            min_val = df_norm[col].min()
            max_val = df_norm[col].max()
            if max_val != min_val:
                df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
        elif method == 'zscore':
            mean_val = df_norm[col].mean()
            std_val = df_norm[col].std()
            if std_val != 0:
                df_norm[col] = (df_norm[col] - mean_val) / std_val
    
    return df_norm

def clean_dataset(df, missing_strategy='mean', outlier_threshold=1.5, normalize=False):
    """
    Complete data cleaning pipeline.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        missing_strategy (str): Strategy for handling missing values
        outlier_threshold (float): IQR threshold for outlier removal
        normalize (bool): Whether to normalize numeric columns
    
    Returns:
        pd.DataFrame: Cleaned and processed DataFrame
    """
    print(f"Original shape: {df.shape}")
    
    # Handle missing values
    df_clean = clean_missing_values(df, strategy=missing_strategy)
    print(f"After missing value handling: {df_clean.shape}")
    
    # Remove outliers
    df_clean = remove_outliers_iqr(df_clean, threshold=outlier_threshold)
    print(f"After outlier removal: {df_clean.shape}")
    
    # Normalize if requested
    if normalize:
        df_clean = normalize_data(df_clean)
        print("Data normalized")
    
    return df_clean

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, np.nan, 4, 5, 100],
        'B': [10, 20, 30, 40, 50, 60],
        'C': [100, 200, 300, 400, 500, 600]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_dataset(df, missing_strategy='mean', outlier_threshold=1.5)
    print("\nCleaned DataFrame:")
    print(cleaned_df)