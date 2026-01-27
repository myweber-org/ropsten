
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
    print(f"Outlier indices: {cleaned_df[outliers].index.tolist()}")