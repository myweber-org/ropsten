
import pandas as pd
import numpy as np

def clean_dataset(df, columns_to_check=None, fill_strategy='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        columns_to_check (list, optional): Specific columns to check for duplicates. 
                                          If None, checks all columns.
        fill_strategy (str): Strategy for filling missing values. 
                            Options: 'mean', 'median', 'mode', 'drop', or 'zero'
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove duplicates
    if columns_to_check:
        cleaned_df = cleaned_df.drop_duplicates(subset=columns_to_check)
    else:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Handle missing values
    for column in cleaned_df.columns:
        if cleaned_df[column].isnull().any():
            if fill_strategy == 'mean' and pd.api.types.is_numeric_dtype(cleaned_df[column]):
                cleaned_df[column].fillna(cleaned_df[column].mean(), inplace=True)
            elif fill_strategy == 'median' and pd.api.types.is_numeric_dtype(cleaned_df[column]):
                cleaned_df[column].fillna(cleaned_df[column].median(), inplace=True)
            elif fill_strategy == 'mode':
                cleaned_df[column].fillna(cleaned_df[column].mode()[0], inplace=True)
            elif fill_strategy == 'drop':
                cleaned_df = cleaned_df.dropna(subset=[column])
            elif fill_strategy == 'zero':
                cleaned_df[column].fillna(0, inplace=True)
            else:
                # For non-numeric columns with mean/median strategy, use mode instead
                cleaned_df[column].fillna(cleaned_df[column].mode()[0], inplace=True)
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate that a DataFrame meets basic requirements.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list, optional): List of columns that must be present
        min_rows (int): Minimum number of rows required
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"

# Example usage (commented out for production)
# if __name__ == "__main__":
#     # Create sample data
#     data = {
#         'id': [1, 2, 2, 3, 4, 5],
#         'name': ['Alice', 'Bob', 'Bob', 'Charlie', None, 'Eve'],
#         'age': [25, 30, 30, None, 28, 35],
#         'score': [85.5, 92.0, 92.0, 78.5, 88.0, None]
#     }
#     
#     df = pd.DataFrame(data)
#     print("Original DataFrame:")
#     print(df)
#     print("\nCleaned DataFrame (mean fill):")
#     cleaned = clean_dataset(df, fill_strategy='mean')
#     print(cleaned)
#     
#     # Validate the cleaned data
#     is_valid, message = validate_dataframe(cleaned, required_columns=['id', 'name', 'age'])
#     print(f"\nValidation: {message}")