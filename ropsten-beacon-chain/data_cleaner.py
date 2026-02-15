
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, convert_types=True):
    """
    Clean a pandas DataFrame by removing duplicates and converting data types.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to remove duplicate rows.
    convert_types (bool): Whether to convert columns to optimal data types.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows.")
    
    if convert_types:
        for col in cleaned_df.select_dtypes(include=['object']).columns:
            try:
                cleaned_df[col] = pd.to_datetime(cleaned_df[col])
                print(f"Converted column '{col}' to datetime.")
            except (ValueError, TypeError):
                try:
                    cleaned_df[col] = pd.to_numeric(cleaned_df[col])
                    print(f"Converted column '{col}' to numeric.")
                except (ValueError, TypeError):
                    pass
    
    cleaned_df = cleaned_df.reset_index(drop=True)
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    dict: Dictionary with validation results.
    """
    validation_result = {
        'is_valid': True,
        'missing_columns': [],
        'null_counts': {},
        'dtypes': {}
    }
    
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            validation_result['is_valid'] = False
            validation_result['missing_columns'] = missing
    
    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            validation_result['null_counts'][col] = null_count
        
        validation_result['dtypes'][col] = str(df[col].dtype)
    
    return validation_result

def sample_data_processing():
    """
    Example function demonstrating data cleaning workflow.
    """
    data = {
        'id': [1, 2, 2, 3, 4, 4],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', 'David', 'David'],
        'score': ['85', '92', '92', '78', '95', '95'],
        'date': ['2023-01-01', '2023-01-02', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-04']
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\nDataFrame info:")
    print(df.info())
    
    cleaned = clean_dataset(df)
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    validation = validate_dataframe(cleaned, required_columns=['id', 'name', 'score'])
    print("\nValidation results:")
    for key, value in validation.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    sample_data_processing()
import pandas as pd

def clean_dataframe(df, column, threshold, keep_above=True):
    """
    Filters a DataFrame based on a numeric threshold in a specified column.
    Returns a new DataFrame.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    if keep_above:
        filtered_df = df[df[column] > threshold].copy()
    else:
        filtered_df = df[df[column] <= threshold].copy()

    filtered_df.reset_index(drop=True, inplace=True)
    return filtered_df

def remove_duplicates_by_column(df, column):
    """
    Removes duplicate rows based on values in a specified column,
    keeping the first occurrence.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    cleaned_df = df.drop_duplicates(subset=[column], keep='first').copy()
    cleaned_df.reset_index(drop=True, inplace=True)
    return cleaned_df