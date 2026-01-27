
def remove_duplicates(seq):
    seen = set()
    result = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return resultimport pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_strategy='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    original_shape = df.shape
    
    if drop_duplicates:
        df = df.drop_duplicates()
        print(f"Removed {original_shape[0] - df.shape[0]} duplicate rows.")
    
    if fill_missing:
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            if fill_strategy == 'mean':
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            elif fill_strategy == 'median':
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            elif fill_strategy == 'mode':
                for col in df.columns:
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else None)
            elif fill_strategy == 'drop':
                df = df.dropna()
            print(f"Handled {missing_count} missing values using {fill_strategy} strategy.")
    
    print(f"Dataset cleaned. Original shape: {original_shape}, New shape: {df.shape}")
    return df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the dataset structure and content.
    """
    if df.empty:
        raise ValueError("Dataset is empty.")
    
    if len(df) < min_rows:
        raise ValueError(f"Dataset has fewer than {min_rows} rows.")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 3, None, 5],
        'B': [10, None, 10, 30, 40, 50],
        'C': ['x', 'y', 'x', 'y', 'z', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\n")
    
    cleaned_df = clean_dataset(df.copy(), fill_strategy='mean')
    print("\nCleaned dataset:")
    print(cleaned_df)
    
    try:
        validate_data(cleaned_df, required_columns=['A', 'B'], min_rows=3)
        print("Data validation passed.")
    except ValueError as e:
        print(f"Data validation failed: {e}")
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
    Calculate summary statistics for a DataFrame column.
    
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
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def normalize_column(df, column, method='minmax'):
    """
    Normalize a DataFrame column using specified method.
    
    Args:
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
            df_copy[column] = (df_copy[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df_copy[column].mean()
        std_val = df_copy[column].std()
        if std_val != 0:
            df_copy[column] = (df_copy[column] - mean_val) / std_val
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    return df_copy

if __name__ == "__main__":
    sample_data = {
        'values': [10, 12, 12, 13, 12, 11, 14, 13, 15, 100, 12, 13, 12, 11, 14]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    print("DataFrame after outlier removal:")
    print(cleaned_df)
    print()
    
    stats = calculate_summary_statistics(df, 'values')
    print("Summary Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")
    print()
    
    normalized_df = normalize_column(df, 'values', method='minmax')
    print("DataFrame after min-max normalization:")
    print(normalized_df)