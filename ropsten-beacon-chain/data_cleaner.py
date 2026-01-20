
import pandas as pd

def clean_dataset(df, columns_to_check=None, fill_missing=True, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        columns_to_check (list, optional): Columns to check for duplicates. 
            If None, checks all columns. Defaults to None.
        fill_missing (bool, optional): Whether to fill missing values. 
            Defaults to True.
        fill_value (any, optional): Value to fill missing entries with. 
            Defaults to 0.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Remove duplicates
    if columns_to_check is None:
        cleaned_df = cleaned_df.drop_duplicates()
    else:
        cleaned_df = cleaned_df.drop_duplicates(subset=columns_to_check)
    
    # Handle missing values
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list, optional): List of required column names.
    
    Returns:
        tuple: (is_valid, message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "Data validation passed"

# Example usage (commented out for production)
# if __name__ == "__main__":
#     sample_data = {
#         'id': [1, 2, 2, 3, 4, None],
#         'value': [10, 20, 20, None, 40, 50],
#         'category': ['A', 'B', 'B', 'C', None, 'D']
#     }
#     
#     df = pd.DataFrame(sample_data)
#     print("Original DataFrame:")
#     print(df)
#     
#     cleaned = clean_dataset(df, columns_to_check=['id'], fill_value='unknown')
#     print("\nCleaned DataFrame:")
#     print(cleaned)
#     
#     is_valid, message = validate_data(cleaned, required_columns=['id', 'value'])
#     print(f"\nValidation: {is_valid} - {message}")