
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_strategy='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to remove duplicate rows
        fill_missing (bool): Whether to fill missing values
        fill_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', 'zero')
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing and cleaned_df.isnull().sum().any():
        for column in cleaned_df.select_dtypes(include=[np.number]).columns:
            if cleaned_df[column].isnull().any():
                if fill_strategy == 'mean':
                    fill_value = cleaned_df[column].mean()
                elif fill_strategy == 'median':
                    fill_value = cleaned_df[column].median()
                elif fill_strategy == 'mode':
                    fill_value = cleaned_df[column].mode()[0]
                elif fill_strategy == 'zero':
                    fill_value = 0
                else:
                    fill_value = cleaned_df[column].mean()
                
                cleaned_df[column] = cleaned_df[column].fillna(fill_value)
                print(f"Filled missing values in column '{column}' with {fill_strategy}: {fill_value}")
        
        for column in cleaned_df.select_dtypes(include=['object']).columns:
            if cleaned_df[column].isnull().any():
                cleaned_df[column] = cleaned_df[column].fillna('Unknown')
                print(f"Filled missing values in column '{column}' with 'Unknown'")
    
    print(f"Data cleaning complete. Original shape: {df.shape}, Cleaned shape: {cleaned_df.shape}")
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for basic integrity checks.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of columns that must be present
    
    Returns:
        dict: Dictionary with validation results
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
    
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        validation_results['warnings'].append(f'Found {duplicate_count} duplicate rows')
    
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        validation_results['warnings'].append(f'Found {missing_values} missing values')
    
    return validation_results

def sample_usage():
    """
    Example usage of the data cleaning functions.
    """
    np.random.seed(42)
    
    data = {
        'id': [1, 2, 3, 3, 4, 5, 6, 7, 8, 9],
        'name': ['Alice', 'Bob', 'Charlie', 'Charlie', 'David', 'Eve', 'Frank', None, 'Helen', 'Ivan'],
        'age': [25, 30, 35, 35, 28, None, 40, 45, 32, 29],
        'score': [85.5, 92.0, 78.5, 78.5, 88.0, 95.5, 82.0, 79.5, 91.0, None],
        'department': ['Sales', 'IT', 'HR', 'HR', 'IT', 'Sales', 'IT', 'HR', 'IT', 'Sales']
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    validation = validate_dataframe(df, required_columns=['id', 'name', 'age'])
    print("Validation Results:")
    for key, value in validation.items():
        print(f"{key}: {value}")
    
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_strategy='median')
    print("\nCleaned DataFrame:")
    print(cleaned_df)

if __name__ == "__main__":
    sample_usage()