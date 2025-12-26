import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing=False, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    df_clean = df.copy()
    
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates()
    
    if fill_missing:
        df_clean = df_clean.fillna(fill_value)
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate that the DataFrame contains required columns and has no infinite values.
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    if df.isin([float('inf'), float('-inf')]).any().any():
        raise ValueError("DataFrame contains infinite values")
    
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4],
        'value': [10.5, None, 15.2, 20.1, None],
        'category': ['A', 'B', 'B', 'C', 'A']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_dataframe(df, drop_duplicates=True, fill_missing=True, fill_value=0)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    try:
        validate_dataframe(cleaned_df, required_columns=['id', 'value', 'category'])
        print("\nData validation passed")
    except ValueError as e:
        print(f"\nData validation failed: {e}")