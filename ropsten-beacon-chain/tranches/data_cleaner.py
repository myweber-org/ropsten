
import pandas as pd
import numpy as np
from pathlib import Path

def clean_csv_data(input_path, output_path=None, missing_strategy='mean'):
    """
    Load and clean CSV data by handling missing values.
    
    Parameters:
    input_path (str): Path to input CSV file
    output_path (str, optional): Path for cleaned output CSV
    missing_strategy (str): Strategy for handling missing values
                           ('mean', 'median', 'drop', 'zero')
    
    Returns:
    pandas.DataFrame: Cleaned DataFrame
    """
    
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    df = pd.read_csv(input_path)
    
    original_rows = len(df)
    original_cols = len(df.columns)
    
    print(f"Original data: {original_rows} rows, {original_cols} columns")
    print(f"Missing values per column:")
    print(df.isnull().sum())
    
    if missing_strategy == 'drop':
        df_cleaned = df.dropna()
    elif missing_strategy == 'mean':
        df_cleaned = df.fillna(df.mean(numeric_only=True))
    elif missing_strategy == 'median':
        df_cleaned = df.fillna(df.median(numeric_only=True))
    elif missing_strategy == 'zero':
        df_cleaned = df.fillna(0)
    else:
        raise ValueError(f"Unknown strategy: {missing_strategy}")
    
    cleaned_rows = len(df_cleaned)
    rows_removed = original_rows - cleaned_rows
    
    print(f"Cleaned data: {cleaned_rows} rows")
    print(f"Rows removed: {rows_removed}")
    
    if output_path:
        df_cleaned.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
    
    return df_cleaned

def detect_outliers_iqr(df, column, threshold=1.5):
    """
    Detect outliers using Interquartile Range method.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame
    column (str): Column name to check for outliers
    threshold (float): IQR multiplier threshold
    
    Returns:
    pandas.Series: Boolean mask of outliers
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    
    outlier_count = outliers.sum()
    print(f"Outliers detected in '{column}': {outlier_count}")
    
    return outliers

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 5, 100],
        'B': [10, np.nan, 30, 40, 50, 60],
        'C': [100, 200, 300, np.nan, 500, 600]
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned_df = clean_csv_data('test_data.csv', 
                               'cleaned_data.csv',
                               missing_strategy='mean')
    
    outliers = detect_outliers_iqr(cleaned_df, 'A')
    print(f"Outlier indices: {cleaned_df[outliers].index.tolist()}")import pandas as pd

def clean_dataframe(df, drop_na=True, rename_columns=True):
    """
    Clean a pandas DataFrame by removing null values and standardizing column names.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_na (bool): If True, drop rows with any null values.
        rename_columns (bool): If True, rename columns to lowercase with underscores.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_na:
        cleaned_df = cleaned_df.dropna()
    
    if rename_columns:
        cleaned_df.columns = (
            cleaned_df.columns
            .str.lower()
            .str.replace(' ', '_')
            .str.replace(r'[^\w_]', '', regex=True)
        )
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and required columns.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        subset (list): Columns to consider for duplicates.
        keep (str): Which duplicates to keep ('first', 'last', False).
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def convert_column_types(df, column_types):
    """
    Convert specified columns to given data types.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column_types (dict): Dictionary mapping column names to target types.
    
    Returns:
        pd.DataFrame: DataFrame with converted column types.
    """
    converted_df = df.copy()
    
    for column, dtype in column_types.items():
        if column in converted_df.columns:
            try:
                converted_df[column] = converted_df[column].astype(dtype)
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not convert column '{column}' to {dtype}: {e}")
    
    return converted_df

def filter_outliers(df, column, method='iqr', threshold=1.5):
    """
    Filter outliers from a DataFrame column.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to filter.
        method (str): Method for outlier detection ('iqr' or 'zscore').
        threshold (float): Threshold for outlier detection.
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        return df
    
    data = df[column].dropna()
    
    if method == 'iqr':
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)
    
    elif method == 'zscore':
        mean = data.mean()
        std = data.std()
        if std > 0:
            z_scores = abs((df[column] - mean) / std)
            mask = z_scores <= threshold
        else:
            mask = pd.Series(True, index=df.index)
    
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    return df[mask]