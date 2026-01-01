
import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    subset (list, optional): Columns to consider for duplicates
    keep (str): Which duplicates to keep - 'first', 'last', or False
    
    Returns:
    pd.DataFrame: DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    if subset is not None:
        if not all(col in df.columns for col in subset):
            raise ValueError("All subset columns must exist in DataFrame")
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(df) - len(cleaned_df)
    print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df.reset_index(drop=True)

def clean_missing_values(df, strategy='drop', fill_value=None):
    """
    Handle missing values in DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): 'drop' to remove rows, 'fill' to replace values
    fill_value: Value to fill missing values with
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if strategy == 'drop':
        cleaned_df = df.dropna()
        print(f"Removed {len(df) - len(cleaned_df)} rows with missing values")
    elif strategy == 'fill':
        if fill_value is None:
            fill_value = df.mean(numeric_only=True)
        cleaned_df = df.fillna(fill_value)
        print(f"Filled missing values with {fill_value}")
    else:
        raise ValueError("Strategy must be 'drop' or 'fill'")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    bool: True if validation passes
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return True
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"DataFrame validated successfully: {len(df)} rows, {len(df.columns)} columns")
    return True

def main():
    """
    Example usage of data cleaning functions.
    """
    sample_data = {
        'id': [1, 2, 3, 2, 4, 5, 3],
        'name': ['Alice', 'Bob', 'Charlie', 'Bob', 'David', 'Eve', 'Charlie'],
        'score': [85, 90, None, 90, 95, 88, 78],
        'department': ['HR', 'IT', 'IT', 'IT', 'HR', 'IT', 'IT']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    validate_dataframe(df, required_columns=['id', 'name'])
    
    df_no_duplicates = remove_duplicates(df, subset=['id', 'name'])
    print("\nAfter removing duplicates:")
    print(df_no_duplicates)
    
    df_cleaned = clean_missing_values(df_no_duplicates, strategy='fill', fill_value=0)
    print("\nAfter handling missing values:")
    print(df_cleaned)

if __name__ == "__main__":
    main()