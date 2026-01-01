
import pandas as pd
import numpy as np

def clean_dataset(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by handling missing values, removing duplicates,
    and normalizing text columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column_mapping (dict): Optional column renaming dictionary
    drop_duplicates (bool): Whether to remove duplicate rows
    normalize_text (bool): Whether to normalize text columns
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    
    df_clean = df.copy()
    
    if column_mapping:
        df_clean = df_clean.rename(columns=column_mapping)
    
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    
    for col in df_clean.select_dtypes(include=['object']).columns:
        df_clean[col] = df_clean[col].astype(str)
        
        if normalize_text:
            df_clean[col] = df_clean[col].str.strip()
            df_clean[col] = df_clean[col].str.lower()
            df_clean[col] = df_clean[col].replace('', np.nan)
    
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
    dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing columns: {missing_columns}")
    
    if df.empty:
        validation_results['warnings'].append("DataFrame is empty")
    
    null_counts = df.isnull().sum()
    if null_counts.any():
        validation_results['warnings'].append(f"Null values found: {null_counts[null_counts > 0].to_dict()}")
    
    return validation_results

def sample_data_processing():
    """
    Example usage of the data cleaning functions.
    """
    sample_data = {
        'Name': ['John Doe', 'Jane Smith', 'John Doe', 'Bob Johnson', ''],
        'Age': [25, 30, 25, 35, None],
        'Email': ['JOHN@example.com', 'jane@example.com', 'john@example.com', 'bob@example.com', None],
        'Score': ['85', '92', '85', '78', 'N/A']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataset(df, drop_duplicates=True, normalize_text=True)
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\n" + "="*50 + "\n")
    
    validation = validate_dataframe(cleaned_df, required_columns=['Name', 'Age', 'Email'])
    print("Validation Results:")
    print(validation)

if __name__ == "__main__":
    sample_data_processing()