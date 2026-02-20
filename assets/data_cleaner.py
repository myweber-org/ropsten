import pandas as pd
import numpy as np

def clean_csv_data(file_path, output_path=None, fill_strategy='mean'):
    """
    Clean a CSV file by handling missing values and removing duplicates.
    
    Args:
        file_path (str): Path to the input CSV file.
        output_path (str, optional): Path to save cleaned CSV. If None, returns DataFrame.
        fill_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', 'drop').
    
    Returns:
        pd.DataFrame or None: Cleaned DataFrame if output_path is None, else None.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None
    
    original_shape = df.shape
    print(f"Original data shape: {original_shape}")
    
    df = df.drop_duplicates()
    print(f"Removed {original_shape[0] - df.shape[0]} duplicate rows.")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if fill_strategy == 'drop':
        df = df.dropna()
        print(f"Removed rows with missing values. New shape: {df.shape}")
    elif fill_strategy in ['mean', 'median'] and numeric_cols:
        for col in numeric_cols:
            if df[col].isnull().any():
                if fill_strategy == 'mean':
                    fill_value = df[col].mean()
                else:
                    fill_value = df[col].median()
                df[col].fillna(fill_value, inplace=True)
                print(f"Filled missing values in '{col}' with {fill_strategy}: {fill_value:.2f}")
    elif fill_strategy == 'mode':
        for col in df.columns:
            if df[col].isnull().any():
                fill_value = df[col].mode()[0] if not df[col].mode().empty else 'UNKNOWN'
                df[col].fillna(fill_value, inplace=True)
                print(f"Filled missing values in '{col}' with mode: {fill_value}")
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
        return None
    else:
        return df

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for basic integrity checks.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list, optional): List of columns that must be present.
    
    Returns:
        dict: Dictionary with validation results.
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'summary': {}
    }
    
    if df is None or df.empty:
        validation_results['is_valid'] = False
        validation_results['errors'].append("DataFrame is None or empty.")
        return validation_results
    
    validation_results['summary']['shape'] = df.shape
    validation_results['summary']['columns'] = df.columns.tolist()
    validation_results['summary']['dtypes'] = df.dtypes.to_dict()
    
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        validation_results['warnings'].append(f"Found {missing_values} missing values in the dataset.")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing required columns: {missing_cols}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        validation_results['summary']['numeric_stats'] = df[numeric_cols].describe().to_dict()
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'value': [10.5, 20.3, np.nan, 40.1, 50.0, 60.7, np.nan, 80.2, 90.9, 100.0],
        'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
        'score': [85, 92, 78, np.nan, 88, 95, 76, 89, 91, 94]
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned_df = clean_csv_data('test_data.csv', fill_strategy='mean')
    
    if cleaned_df is not None:
        validation = validate_dataframe(cleaned_df, required_columns=['id', 'value', 'category'])
        print("\nValidation Results:")
        print(f"Is Valid: {validation['is_valid']}")
        print(f"Shape: {validation['summary']['shape']}")
        
        if validation['warnings']:
            print(f"Warnings: {validation['warnings']}")
        
        if validation['errors']:
            print(f"Errors: {validation['errors']}")
    
    import os
    if os.path.exists('test_data.csv'):
        os.remove('test_data.csv')