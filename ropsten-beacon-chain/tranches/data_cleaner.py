import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    Returns a filtered DataFrame.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_column_minmax(data, column):
    """
    Normalize a column using min-max scaling to range [0, 1].
    Returns a new DataFrame with the normalized column.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        normalized = 0.5
    else:
        normalized = (data[column] - min_val) / (max_val - min_val)
    
    result = data.copy()
    result[f'{column}_normalized'] = normalized
    return result

def calculate_basic_stats(data, column):
    """
    Calculate basic statistics for a column.
    Returns a dictionary with mean, median, std, min, and max.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': data[column].mean(),
        'median': data[column].median(),
        'std': data[column].std(),
        'min': data[column].min(),
        'max': data[column].max(),
        'count': data[column].count()
    }
    return stats

def clean_missing_values(data, strategy='mean'):
    """
    Handle missing values in numeric columns.
    Strategy can be 'mean', 'median', or 'drop'.
    Returns a cleaned DataFrame.
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        cleaned_data = data.dropna(subset=numeric_cols)
    elif strategy == 'mean':
        cleaned_data = data.copy()
        for col in numeric_cols:
            cleaned_data[col].fillna(data[col].mean(), inplace=True)
    elif strategy == 'median':
        cleaned_data = data.copy()
        for col in numeric_cols:
            cleaned_data[col].fillna(data[col].median(), inplace=True)
    else:
        raise ValueError("Strategy must be 'mean', 'median', or 'drop'")
    
    return cleaned_data

def validate_dataframe(data):
    """
    Basic validation of DataFrame structure.
    Returns True if valid, raises exceptions otherwise.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if data.empty:
        raise ValueError("DataFrame is empty")
    
    if data.isnull().all().any():
        raise ValueError("Some columns contain only null values")
    
    return Trueimport numpy as np
import pandas as pd

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
    if max_val == min_val:
        return df[column].apply(lambda x: 0.5)
    return df[column].apply(lambda x: (x - min_val) / (max_val - min_val))

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    return cleaned_df.reset_index(drop=True)

def validate_dataframe(df):
    required_checks = [
        (lambda: not df.empty, "DataFrame is empty"),
        (lambda: df.isnull().sum().sum() == 0, "DataFrame contains null values"),
        (lambda: all(df.dtypes != object), "DataFrame contains non-numeric columns")
    ]
    for check, message in required_checks:
        if not check():
            raise ValueError(f"Validation failed: {message}")
    return True

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 200),
        'feature_b': np.random.exponential(scale=2.0, size=200),
        'feature_c': np.random.uniform(0, 1, 200)
    })
    print("Original shape:", sample_data.shape)
    cleaned = clean_dataset(sample_data, ['feature_a', 'feature_b', 'feature_c'])
    print("Cleaned shape:", cleaned.shape)
    try:
        validate_dataframe(cleaned)
        print("Data validation passed")
    except ValueError as e:
        print(f"Data validation error: {e}")import pandas as pd
import sys

def clean_data(input_file, output_file):
    try:
        df = pd.read_csv(input_file)
        print(f"Original shape: {df.shape}")
        
        df_cleaned = df.drop_duplicates()
        print(f"After removing duplicates: {df_cleaned.shape}")
        
        df_cleaned = df_cleaned.dropna()
        print(f"After removing missing values: {df_cleaned.shape}")
        
        df_cleaned.to_csv(output_file, index=False)
        print(f"Cleaned data saved to {output_file}")
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python data_cleaner.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    clean_data(input_file, output_file)import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range (IQR) method.
    
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
    
    return filtered_df

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
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

def process_dataframe(df, numeric_columns):
    """
    Process multiple numeric columns to remove outliers and return cleaned DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of column names to process
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[0:50, 'A'] = 500
    
    print("Original DataFrame shape:", df.shape)
    print("Original summary for column A:")
    print(calculate_summary_statistics(df, 'A'))
    
    cleaned_df = process_dataframe(df, ['A', 'B', 'C'])
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Cleaned summary for column A:")
    print(calculate_summary_statistics(cleaned_df, 'A'))