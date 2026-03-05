
import pandas as pd

def clean_dataset(df, drop_na=True, column_case='lower'):
    """
    Clean a pandas DataFrame by handling null values and standardizing column names.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_na (bool): If True, drop rows with any null values. Default is True.
    column_case (str): Target case for column names ('lower', 'upper', 'title'). Default is 'lower'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_na:
        cleaned_df = cleaned_df.dropna()
    else:
        cleaned_df = cleaned_df.fillna(0)
    
    if column_case == 'lower':
        cleaned_df.columns = cleaned_df.columns.str.lower()
    elif column_case == 'upper':
        cleaned_df.columns = cleaned_df.columns.str.upper()
    elif column_case == 'title':
        cleaned_df.columns = cleaned_df.columns.str.title()
    
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate that the DataFrame meets basic requirements.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    tuple: (bool, str) indicating validation success and message.
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"

if __name__ == "__main__":
    sample_data = {
        'Name': ['Alice', 'Bob', None, 'David'],
        'Age': [25, None, 35, 40],
        'Score': [85.5, 92.0, 78.5, None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned = clean_dataset(df, drop_na=True, column_case='lower')
    print(cleaned)
    
    is_valid, message = validate_dataframe(cleaned, required_columns=['name', 'age'])
    print(f"\nValidation: {is_valid} - {message}")