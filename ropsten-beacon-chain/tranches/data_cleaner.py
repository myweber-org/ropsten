
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
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
    
    return filtered_df.reset_index(drop=True)

def calculate_summary_stats(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    stats = {
        'original_count': len(df),
        'cleaned_count': len(df[column].dropna()),
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max()
    }
    return stats

def clean_dataset(df, columns_to_clean=None):
    """
    Clean multiple columns in a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns_to_clean (list): List of column names to clean. If None, clean all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    dict: Dictionary of cleaning statistics for each column
    """
    if columns_to_clean is None:
        columns_to_clean = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    stats_dict = {}
    
    for column in columns_to_clean:
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            original_len = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_len - len(cleaned_df)
            
            stats_dict[column] = {
                'outliers_removed': removed_count,
                'percentage_removed': (removed_count / original_len * 100) if original_len > 0 else 0,
                'summary_stats': calculate_summary_stats(cleaned_df, column)
            }
    
    return cleaned_df, stats_dict

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    sample_data = {
        'temperature': np.random.normal(25, 5, 1000),
        'humidity': np.random.normal(60, 15, 1000),
        'pressure': np.random.normal(1013, 10, 1000)
    }
    
    # Add some outliers
    sample_data['temperature'][:50] = np.random.uniform(100, 150, 50)
    sample_data['humidity'][:30] = np.random.uniform(200, 300, 30)
    
    df = pd.DataFrame(sample_data)
    
    print("Original dataset shape:", df.shape)
    print("\nOriginal summary statistics:")
    print(df.describe())
    
    cleaned_df, stats = clean_dataset(df)
    
    print("\nCleaned dataset shape:", cleaned_df.shape)
    print("\nCleaning statistics:")
    for col, col_stats in stats.items():
        print(f"\n{col}:")
        print(f"  Outliers removed: {col_stats['outliers_removed']}")
        print(f"  Percentage removed: {col_stats['percentage_removed']:.2f}%")import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using min-max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using z-score normalization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def handle_missing_values(data, strategy='mean'):
    """
    Handle missing values in numerical columns
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    if strategy == 'mean':
        for col in numeric_cols:
            data[col].fillna(data[col].mean(), inplace=True)
    elif strategy == 'median':
        for col in numeric_cols:
            data[col].fillna(data[col].median(), inplace=True)
    elif strategy == 'mode':
        for col in numeric_cols:
            data[col].fillna(data[col].mode()[0], inplace=True)
    elif strategy == 'drop':
        data.dropna(subset=numeric_cols, inplace=True)
    
    return data

def clean_dataset(data, outlier_columns=None, normalize_columns=None, 
                  standardize_columns=None, missing_strategy='mean'):
    """
    Comprehensive data cleaning pipeline
    """
    cleaned_data = data.copy()
    
    cleaned_data = handle_missing_values(cleaned_data, strategy=missing_strategy)
    
    if outlier_columns:
        for col in outlier_columns:
            if col in cleaned_data.columns:
                cleaned_data = remove_outliers_iqr(cleaned_data, col)
    
    if normalize_columns:
        for col in normalize_columns:
            if col in cleaned_data.columns:
                cleaned_data[f'{col}_normalized'] = normalize_minmax(cleaned_data, col)
    
    if standardize_columns:
        for col in standardize_columns:
            if col in cleaned_data.columns:
                cleaned_data[f'{col}_standardized'] = standardize_zscore(cleaned_data, col)
    
    return cleaned_dataimport pandas as pd
import numpy as np
from typing import Optional

def clean_csv_data(
    input_path: str,
    output_path: str,
    missing_strategy: str = 'mean',
    numeric_columns: Optional[list] = None
) -> pd.DataFrame:
    """
    Clean CSV data by handling missing values and converting data types.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save cleaned CSV file
        missing_strategy: Strategy for handling missing values ('mean', 'median', 'drop', 'zero')
        numeric_columns: List of column names to treat as numeric
    
    Returns:
        Cleaned DataFrame
    """
    
    df = pd.read_csv(input_path)
    
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            if missing_strategy == 'mean':
                df[col].fillna(df[col].mean(), inplace=True)
            elif missing_strategy == 'median':
                df[col].fillna(df[col].median(), inplace=True)
            elif missing_strategy == 'zero':
                df[col].fillna(0, inplace=True)
            elif missing_strategy == 'drop':
                df.dropna(subset=[col], inplace=True)
    
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
    print(f"Original shape: {df.shape}, Missing values handled: {df.isnull().sum().sum()}")
    
    return df

def validate_dataframe(df: pd.DataFrame) -> dict:
    """
    Validate DataFrame structure and content.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'column_types': df.dtypes.to_dict(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
    }
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5],
        'value': [10.5, None, 15.2, 20.1, None],
        'category': ['A', 'B', 'A', 'C', 'B']
    }
    
    df_sample = pd.DataFrame(sample_data)
    df_sample.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data(
        input_path='sample_data.csv',
        output_path='cleaned_data.csv',
        missing_strategy='mean',
        numeric_columns=['value']
    )
    
    validation = validate_dataframe(cleaned_df)
    print("Validation results:", validation)