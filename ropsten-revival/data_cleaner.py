
def deduplicate_list(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd

def clean_dataframe(df, drop_na=True, column_case='lower'):
    """
    Clean a pandas DataFrame by handling null values and standardizing column names.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_na (bool): If True, drop rows with any null values. Default is True.
        column_case (str): Target case for column names. Options: 'lower', 'upper', 'title'. Default is 'lower'.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    if drop_na:
        df_clean = df_clean.dropna()
    
    if column_case == 'lower':
        df_clean.columns = df_clean.columns.str.lower()
    elif column_case == 'upper':
        df_clean.columns = df_clean.columns.str.upper()
    elif column_case == 'title':
        df_clean.columns = df_clean.columns.str.title()
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for required columns and basic structure.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
    
    Returns:
        tuple: (bool, str) indicating validation success and message.
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame validation passed"

if __name__ == "__main__":
    sample_data = {
        'Name': ['Alice', 'Bob', None, 'David'],
        'Age': [25, 30, 35, None],
        'City': ['NYC', 'LA', 'Chicago', 'Boston']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned_df = clean_dataframe(df, drop_na=True, column_case='lower')
    print(cleaned_df)
    
    is_valid, message = validate_dataframe(cleaned_df, required_columns=['name', 'age'])
    print(f"\nValidation: {is_valid}, Message: {message}")