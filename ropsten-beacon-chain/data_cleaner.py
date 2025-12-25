
import pandas as pd
import re

def clean_dataframe(df, columns_to_clean=None):
    """
    Clean a pandas DataFrame by removing duplicate rows and normalizing string columns.
    """
    df_clean = df.copy()
    
    # Remove duplicate rows
    initial_rows = df_clean.shape[0]
    df_clean = df_clean.drop_duplicates()
    removed_duplicates = initial_rows - df_clean.shape[0]
    
    # If specific columns are provided, clean only those; otherwise, clean all object columns
    if columns_to_clean is None:
        columns_to_clean = df_clean.select_dtypes(include=['object']).columns
    else:
        columns_to_clean = [col for col in columns_to_clean if col in df_clean.columns]
    
    for column in columns_to_clean:
        df_clean[column] = df_clean[column].apply(normalize_string)
    
    return df_clean, removed_duplicates

def normalize_string(value):
    """
    Normalize a string: convert to lowercase, remove extra whitespace, and strip.
    """
    if isinstance(value, str):
        value = value.lower()
        value = re.sub(r'\s+', ' ', value)
        value = value.strip()
    return value

if __name__ == "__main__":
    sample_data = {
        'Name': ['Alice', 'Bob', 'Alice', '  Charlie  ', 'bob'],
        'Age': [25, 30, 25, 35, 30],
        'City': ['New York', 'Los Angeles', 'new york', 'Chicago', 'los angeles']
    }
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    cleaned_df, duplicates_removed = clean_dataframe(df)
    print(f"Removed {duplicates_removed} duplicate row(s).")
    print("Cleaned DataFrame:")
    print(cleaned_df)
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
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

if __name__ == "__main__":
    cleaned_data = clean_dataset('raw_data.csv')
    cleaned_data.to_csv('cleaned_data.csv', index=False)
    print(f"Data cleaning complete. Original shape: {pd.read_csv('raw_data.csv').shape}, Cleaned shape: {cleaned_data.shape}")
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Args:
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
    
    return filtered_df

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to analyze
    
    Returns:
        dict: Dictionary containing summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def process_dataframe(df, columns_to_clean):
    """
    Process multiple columns for outlier removal and return cleaned DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns_to_clean (list): List of column names to clean
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    for column in columns_to_clean:
        if column in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
    
    return cleaned_df

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'temperature': [22, 23, 24, 25, 26, 27, 28, 29, 30, 100],
        'humidity': [45, 46, 47, 48, 49, 50, 51, 52, 53, 200],
        'pressure': [1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 500]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    columns_to_process = ['temperature', 'humidity', 'pressure']
    cleaned_df = process_dataframe(df, columns_to_process)
    
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print()
    
    for column in columns_to_process:
        stats = calculate_summary_statistics(cleaned_df, column)
        print(f"Statistics for {column}:")
        for key, value in stats.items():
            print(f"  {key}: {value:.2f}")
        print()