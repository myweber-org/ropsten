
import pandas as pd

def clean_dataframe(df, drop_na=True, rename_columns=None):
    """
    Clean a pandas DataFrame by removing null values and standardizing column names.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_na (bool): If True, drop rows with any null values. Default is True.
    rename_columns (dict): Dictionary mapping old column names to new ones. Default is None.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_na:
        cleaned_df = cleaned_df.dropna()
    
    if rename_columns:
        cleaned_df = cleaned_df.rename(columns=rename_columns)
    
    cleaned_df.columns = cleaned_df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for required columns and data types.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present. Default is None.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    if df.empty:
        print("DataFrame is empty.")
        return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'Name': ['Alice', 'Bob', None, 'David'],
        'Age': [25, 30, 35, None],
        'City': ['New York', 'Los Angeles', 'Chicago', 'Miami']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataframe(df, drop_na=True)
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid = validate_dataframe(cleaned, required_columns=['name', 'age'])
    print(f"\nDataFrame validation result: {is_valid}")