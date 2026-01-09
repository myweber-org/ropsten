
import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df: pandas DataFrame
        subset: column label or sequence of labels to consider for duplicates
        keep: 'first', 'last', or False to drop all duplicates
    
    Returns:
        Cleaned DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(df) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df

def clean_numeric_column(df, column_name):
    """
    Clean a numeric column by removing non-numeric values.
    
    Args:
        df: pandas DataFrame
        column_name: name of the column to clean
    
    Returns:
        DataFrame with cleaned numeric column
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    original_dtype = df[column_name].dtype
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    
    nan_count = df[column_name].isna().sum()
    if nan_count > 0:
        print(f"Converted {nan_count} non-numeric values to NaN in column '{column_name}'")
    
    if df[column_name].dtype != original_dtype:
        print(f"Changed dtype of column '{column_name}' from {original_dtype} to {df[column_name].dtype}")
    
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: list of column names that must be present
    
    Returns:
        Boolean indicating if validation passed
    """
    if df.empty:
        print("Warning: DataFrame is empty")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    print(f"DataFrame validation passed: {len(df)} rows, {len(df.columns)} columns")
    return True

def main():
    """
    Example usage of data cleaning functions.
    """
    sample_data = {
        'id': [1, 2, 2, 3, 4, 4],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', 'David', 'David'],
        'score': ['95', '88', '88', '92', 'invalid', '85']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    df_clean = remove_duplicates(df, subset=['id', 'name'])
    print("After removing duplicates:")
    print(df_clean)
    print()
    
    df_clean = clean_numeric_column(df_clean, 'score')
    print("After cleaning numeric column:")
    print(df_clean)
    print()
    
    is_valid = validate_dataframe(df_clean, required_columns=['id', 'name', 'score'])
    print(f"DataFrame is valid: {is_valid}")

if __name__ == "__main__":
    main()