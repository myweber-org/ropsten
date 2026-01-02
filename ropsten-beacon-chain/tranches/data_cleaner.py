import pandas as pd
import numpy as np

def clean_csv_data(filepath, fill_strategy='mean', drop_threshold=0.5):
    """
    Load and clean CSV data by handling missing values.
    
    Args:
        filepath (str): Path to the CSV file.
        fill_strategy (str): Strategy for filling missing values.
            Options: 'mean', 'median', 'mode', 'zero', 'ffill'.
        drop_threshold (float): Drop columns with missing ratio above this threshold.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    
    original_shape = df.shape
    print(f"Original data shape: {original_shape}")
    
    # Calculate missing ratio per column
    missing_ratio = df.isnull().sum() / len(df)
    
    # Drop columns with high missing ratio
    columns_to_drop = missing_ratio[missing_ratio > drop_threshold].index
    df = df.drop(columns=columns_to_drop)
    
    if len(columns_to_drop) > 0:
        print(f"Dropped columns with >{drop_threshold*100}% missing values: {list(columns_to_drop)}")
    
    # Fill missing values based on strategy
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    
    if fill_strategy == 'mean':
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].mean())
    elif fill_strategy == 'median':
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
    elif fill_strategy == 'mode':
        for col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else None)
    elif fill_strategy == 'zero':
        df = df.fillna(0)
    elif fill_strategy == 'ffill':
        df = df.fillna(method='ffill').fillna(method='bfill')
    else:
        raise ValueError(f"Unknown fill strategy: {fill_strategy}")
    
    # Fill any remaining missing values in categorical columns with 'Unknown'
    for col in categorical_cols:
        df[col] = df[col].fillna('Unknown')
    
    # Remove duplicate rows
    duplicates = df.duplicated().sum()
    df = df.drop_duplicates()
    
    print(f"Removed {duplicates} duplicate rows")
    print(f"Final data shape: {df.shape}")
    print(f"Missing values after cleaning: {df.isnull().sum().sum()}")
    
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
    
    Returns:
        dict: Validation results.
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    if not isinstance(df, pd.DataFrame):
        validation_results['is_valid'] = False
        validation_results['errors'].append('Input is not a pandas DataFrame')
        return validation_results
    
    if df.empty:
        validation_results['warnings'].append('DataFrame is empty')
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f'Missing required columns: {missing_columns}')
    
    # Check for infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.isinf(df[col]).any():
            validation_results['warnings'].append(f'Column {col} contains infinite values')
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, np.nan, np.nan, 10, 11],
        'C': ['X', 'Y', np.nan, 'Z', 'X'],
        'D': [100, 200, 300, np.nan, 500]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data('sample_data.csv', fill_strategy='mean', drop_threshold=0.6)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    validation = validate_dataframe(cleaned_df, required_columns=['A', 'C', 'D'])
    print("\nValidation results:")
    print(validation)