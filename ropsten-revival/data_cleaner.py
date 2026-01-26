import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (bool): Whether to fill missing values.
    fill_value: Value to use for filling missing data.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def validate_numeric_columns(df, columns):
    """
    Validate that specified columns contain only numeric values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns (list): List of column names to validate.
    
    Returns:
    dict: Dictionary with validation results for each column.
    """
    results = {}
    
    for col in columns:
        if col in df.columns:
            non_numeric = pd.to_numeric(df[col], errors='coerce').isna().sum()
            results[col] = {
                'total_rows': len(df),
                'non_numeric_count': non_numeric,
                'is_valid': non_numeric == 0
            }
    
    return results

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, 2, None, 5],
        'B': [10, 20, 20, 40, 50],
        'C': ['x', 'y', 'y', 'z', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned = clean_dataset(df, fill_value='unknown')
    print(cleaned)
    
    # Validate numeric columns
    validation = validate_numeric_columns(df, ['A', 'B'])
    print("\nColumn Validation:")
    for col, result in validation.items():
        print(f"{col}: {result}")