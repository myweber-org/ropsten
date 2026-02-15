
import pandas as pd
import numpy as np
from pathlib import Path

def clean_csv_data(input_path, output_path=None, missing_strategy='mean'):
    """
    Load and clean CSV data by handling missing values.
    
    Parameters:
    input_path (str): Path to input CSV file
    output_path (str, optional): Path for cleaned CSV output
    missing_strategy (str): Strategy for handling missing values
                           ('mean', 'median', 'drop', 'zero')
    
    Returns:
    pandas.DataFrame: Cleaned dataframe
    """
    
    # Validate input file exists
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Load data
    df = pd.read_csv(input_path)
    
    # Report initial status
    print(f"Original data shape: {df.shape}")
    print(f"Missing values per column:")
    print(df.isnull().sum())
    
    # Handle missing values based on strategy
    if missing_strategy == 'drop':
        df_clean = df.dropna()
    elif missing_strategy == 'mean':
        df_clean = df.fillna(df.mean(numeric_only=True))
    elif missing_strategy == 'median':
        df_clean = df.fillna(df.median(numeric_only=True))
    elif missing_strategy == 'zero':
        df_clean = df.fillna(0)
    else:
        raise ValueError(f"Unknown strategy: {missing_strategy}")
    
    # Remove duplicate rows
    df_clean = df_clean.drop_duplicates()
    
    # Reset index after cleaning
    df_clean = df_clean.reset_index(drop=True)
    
    # Report cleaning results
    print(f"Cleaned data shape: {df_clean.shape}")
    print(f"Remaining missing values: {df_clean.isnull().sum().sum()}")
    
    # Save cleaned data if output path provided
    if output_path:
        output_file = Path(output_path)
        df_clean.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
    
    return df_clean

def validate_numeric_columns(df, columns=None):
    """
    Validate that specified columns contain only numeric data.
    
    Parameters:
    df (pandas.DataFrame): Dataframe to validate
    columns (list, optional): List of column names to check
    
    Returns:
    dict: Validation results for each column
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    results = {}
    for col in columns:
        if col in df.columns:
            non_numeric = pd.to_numeric(df[col], errors='coerce').isna().sum()
            results[col] = {
                'total_values': len(df[col]),
                'non_numeric_count': non_numeric,
                'is_valid': non_numeric == 0
            }
    
    return results

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [10.5, np.nan, 30.2, 40.1, 50.0],
        'C': ['x', 'y', 'z', 'x', 'y']
    }
    
    df_sample = pd.DataFrame(sample_data)
    df_sample.to_csv('sample_data.csv', index=False)
    
    # Clean the sample data
    cleaned_df = clean_csv_data('sample_data.csv', 
                               'cleaned_sample.csv',
                               missing_strategy='mean')
    
    # Validate numeric columns
    validation = validate_numeric_columns(cleaned_df)
    print("\nColumn validation:")
    for col, stats in validation.items():
        print(f"{col}: {stats}")