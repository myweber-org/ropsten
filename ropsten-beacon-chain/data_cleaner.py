
import pandas as pd

def clean_dataset(df, columns_to_check=None, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicate rows and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        columns_to_check (list, optional): List of column names to check for duplicates.
            If None, checks all columns. Defaults to None.
        fill_missing (str or value): Method to fill missing values.
            Options: 'mean', 'median', 'mode', or a specific value. Defaults to 'mean'.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    if columns_to_check is None:
        cleaned_df = cleaned_df.drop_duplicates()
    else:
        cleaned_df = cleaned_df.drop_duplicates(subset=columns_to_check)
    
    # Handle missing values
    for column in cleaned_df.columns:
        if cleaned_df[column].isnull().any():
            if fill_missing == 'mean':
                if pd.api.types.is_numeric_dtype(cleaned_df[column]):
                    cleaned_df[column].fillna(cleaned_df[column].mean(), inplace=True)
            elif fill_missing == 'median':
                if pd.api.types.is_numeric_dtype(cleaned_df[column]):
                    cleaned_df[column].fillna(cleaned_df[column].median(), inplace=True)
            elif fill_missing == 'mode':
                cleaned_df[column].fillna(cleaned_df[column].mode()[0], inplace=True)
            else:
                cleaned_df[column].fillna(fill_missing, inplace=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for basic integrity checks.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list, optional): List of required column names.
    
    Returns:
        dict: Dictionary with validation results.
    """
    validation_results = {
        'is_dataframe': isinstance(df, pd.DataFrame),
        'has_data': not df.empty if isinstance(df, pd.DataFrame) else False,
        'row_count': len(df) if isinstance(df, pd.DataFrame) else 0,
        'column_count': len(df.columns) if isinstance(df, pd.DataFrame) else 0,
        'missing_values': df.isnull().sum().sum() if isinstance(df, pd.DataFrame) else 0,
        'duplicate_rows': df.duplicated().sum() if isinstance(df, pd.DataFrame) else 0
    }
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        validation_results['missing_required_columns'] = missing_columns
        validation_results['all_required_columns_present'] = len(missing_columns) == 0
    
    return validation_results

# Example usage (commented out for production)
# if __name__ == "__main__":
#     # Create sample data
#     data = {
#         'id': [1, 2, 2, 3, 4],
#         'name': ['Alice', 'Bob', 'Bob', None, 'Eve'],
#         'age': [25, 30, 30, 35, None],
#         'score': [85.5, 92.0, 92.0, 78.5, 88.0]
#     }
#     
#     df = pd.DataFrame(data)
#     print("Original DataFrame:")
#     print(df)
#     print("\nValidation Results:")
#     print(validate_dataframe(df))
#     
#     cleaned = clean_dataset(df, columns_to_check=['id', 'name'], fill_missing='mean')
#     print("\nCleaned DataFrame:")
#     print(cleaned)
#     print("\nValidation Results after cleaning:")
#     print(validate_dataframe(cleaned))