
import pandas as pd
import numpy as np

def clean_dataset(df, column_mapping=None, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by handling missing values, duplicates, and standardizing columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    column_mapping (dict): Optional dictionary to rename columns
    drop_duplicates (bool): Whether to remove duplicate rows
    fill_missing (str): Strategy for filling missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    
    df_clean = df.copy()
    
    if column_mapping:
        df_clean = df_clean.rename(columns=column_mapping)
    
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates()
    
    if fill_missing == 'drop':
        df_clean = df_clean.dropna()
    else:
        for column in df_clean.select_dtypes(include=[np.number]).columns:
            if fill_missing == 'mean':
                fill_value = df_clean[column].mean()
            elif fill_missing == 'median':
                fill_value = df_clean[column].median()
            elif fill_missing == 'mode':
                fill_value = df_clean[column].mode()[0] if not df_clean[column].mode().empty else 0
            else:
                fill_value = 0
            
            df_clean[column] = df_clean[column].fillna(fill_value)
    
    for column in df_clean.select_dtypes(include=['object']).columns:
        df_clean[column] = df_clean[column].fillna('Unknown')
        df_clean[column] = df_clean[column].str.strip().str.title()
    
    return df_clean

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate that a DataFrame meets basic requirements.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of column names that must be present
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (bool, str) indicating validation result and error message
    """
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "Validation passed"

def sample_data():
    """
    Create a sample DataFrame for testing.
    
    Returns:
    pd.DataFrame: Sample DataFrame with mixed data
    """
    
    data = {
        'name': ['Alice', 'Bob', 'Charlie', 'Alice', None, 'Eve'],
        'age': [25, 30, None, 25, 35, 28],
        'score': [85.5, 92.0, 78.5, 85.5, 95.0, None],
        'city': ['new york', 'los angeles', 'chicago', 'new york', 'boston', 'seattle']
    }
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = sample_data()
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned_df = clean_dataset(df, fill_missing='mean')
    print(cleaned_df)
    
    is_valid, message = validate_dataframe(cleaned_df, required_columns=['name', 'age', 'score'])
    print(f"\nValidation: {message}")