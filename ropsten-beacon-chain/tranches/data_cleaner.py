
import pandas as pd

def clean_dataset(df, remove_duplicates=True):
    """
    Clean a pandas DataFrame by removing null values and optionally duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    remove_duplicates (bool): Whether to remove duplicate rows. Default is True.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.dropna()
    
    if remove_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    return cleaned_df

def validate_data(df, required_columns):
    """
    Validate that the DataFrame contains all required columns.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    bool: True if all required columns are present, False otherwise.
    """
    return all(col in df.columns for col in required_columns)

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, None, 4, 1],
        'B': [5, 6, 7, None, 5],
        'C': ['x', 'y', 'z', 'x', 'x']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df)
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    required = ['A', 'B']
    is_valid = validate_data(df, required)
    print(f"\nDataFrame has required columns {required}: {is_valid}")