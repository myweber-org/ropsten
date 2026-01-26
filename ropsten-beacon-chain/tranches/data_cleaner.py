import pandas as pd
import numpy as np

def clean_csv_data(filepath, fill_strategy='mean', drop_threshold=0.5):
    """
    Clean CSV data by handling missing values and removing low-quality columns.
    
    Parameters:
    filepath (str): Path to the CSV file
    fill_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', 'zero')
    drop_threshold (float): Threshold for dropping columns with too many missing values (0.0-1.0)
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    
    # Read CSV file
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except Exception as e:
        raise ValueError(f"Error reading CSV: {str(e)}")
    
    # Store original shape for reporting
    original_shape = df.shape
    
    # Remove columns with too many missing values
    missing_ratio = df.isnull().sum() / len(df)
    columns_to_drop = missing_ratio[missing_ratio > drop_threshold].index
    df = df.drop(columns=columns_to_drop)
    
    # Fill missing values based on strategy
    if fill_strategy == 'mean':
        df = df.fillna(df.mean(numeric_only=True))
    elif fill_strategy == 'median':
        df = df.fillna(df.median(numeric_only=True))
    elif fill_strategy == 'mode':
        df = df.fillna(df.mode().iloc[0])
    elif fill_strategy == 'zero':
        df = df.fillna(0)
    else:
        raise ValueError(f"Unknown fill strategy: {fill_strategy}")
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Reset index after cleaning
    df = df.reset_index(drop=True)
    
    # Generate cleaning report
    final_shape = df.shape
    print(f"Data cleaning completed:")
    print(f"  Original shape: {original_shape}")
    print(f"  Final shape: {final_shape}")
    print(f"  Columns removed: {len(columns_to_drop)}")
    print(f"  Duplicates removed: {original_shape[0] - final_shape[0]}")
    
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate dataframe structure and content.
    
    Parameters:
    df (pd.DataFrame): Dataframe to validate
    required_columns (list): List of required column names
    
    Returns:
    dict: Validation results
    """
    
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Check if dataframe is empty
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['errors'].append('Dataframe is empty')
    
    # Check required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f'Missing required columns: {missing_columns}')
    
    # Check for infinite values
    if np.any(np.isinf(df.select_dtypes(include=[np.number]))):
        validation_results['warnings'].append('Dataframe contains infinite values')
    
    # Check data types
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        validation_results['warnings'].append('No numeric columns found')
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, np.nan, np.nan, 10, 11],
        'C': [7, 8, 9, 10, 11],
        'D': [12, 13, 14, 15, 16]
    }
    
    # Create test dataframe
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    # Clean the data
    cleaned_df = clean_csv_data('test_data.csv', fill_strategy='mean', drop_threshold=0.5)
    
    # Validate the cleaned data
    validation = validate_dataframe(cleaned_df, required_columns=['A', 'C', 'D'])
    
    print(f"\nValidation results:")
    print(f"  Is valid: {validation['is_valid']}")
    print(f"  Errors: {validation['errors']}")
    print(f"  Warnings: {validation['warnings']}")