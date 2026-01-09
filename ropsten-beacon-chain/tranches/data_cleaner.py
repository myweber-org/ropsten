
import pandas as pd
import numpy as np

def clean_dataframe(df, drop_na=True, rename_columns=True):
    """
    Clean a pandas DataFrame by handling missing values and standardizing column names.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    drop_na (bool): Whether to drop rows with null values
    rename_columns (bool): Whether to standardize column names
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    
    df_clean = df.copy()
    
    if drop_na:
        df_clean = df_clean.dropna()
    
    if rename_columns:
        df_clean.columns = df_clean.columns.str.strip().str.lower().str.replace(' ', '_')
    
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    dict: Dictionary with validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['errors'].append('DataFrame is empty')
        return validation_results
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f'Missing required columns: {missing_cols}')
    
    null_counts = df.isnull().sum()
    if null_counts.any():
        validation_results['warnings'].append(f'Null values found: {null_counts[null_counts > 0].to_dict()}')
    
    return validation_results

def sample_data_processing():
    """
    Example function demonstrating data cleaning workflow.
    """
    sample_data = {
        'Customer ID': [1, 2, 3, None, 5],
        'Order Value': [100.50, None, 75.25, 200.00, 150.75],
        'Order Date': ['2023-01-01', '2023-01-02', None, '2023-01-04', '2023-01-05']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nDataFrame Info:")
    print(df.info())
    
    cleaned_df = clean_dataframe(df, drop_na=True, rename_columns=True)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    validation = validate_dataframe(cleaned_df, required_columns=['customer_id', 'order_value'])
    print("\nValidation Results:")
    print(validation)
    
    return cleaned_df

if __name__ == "__main__":
    processed_data = sample_data_processing()