import pandas as pd
import numpy as np

def clean_dataset(df):
    """
    Cleans a pandas DataFrame by removing duplicate rows and
    handling missing values in numeric columns.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()
    
    # Fill missing numeric values with column median
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_cleaned[col].isnull().any():
            median_val = df_cleaned[col].median()
            df_cleaned[col].fillna(median_val, inplace=True)
    
    # Drop rows where non-numeric essential columns are null
    essential_cols = ['id', 'timestamp']  # Example essential columns
    existing_essential = [col for col in essential_cols if col in df_cleaned.columns]
    if existing_essential:
        df_cleaned.dropna(subset=existing_essential, inplace=True)
    
    return df_cleaned

def validate_data(df, required_columns):
    """
    Validates that the DataFrame contains all required columns
    and has no null values in required fields.
    """
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    null_counts = df[required_columns].isnull().sum()
    if null_counts.any():
        raise ValueError(f"Null values found in required columns:\n{null_counts[null_counts > 0]}")
    
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'timestamp': ['2023-01-01', '2023-01-02', '2023-01-02', None, '2023-01-04', '2023-01-05'],
        'value': [10.5, 20.3, 20.3, 15.7, None, 30.1],
        'category': ['A', 'B', 'B', 'C', 'A', 'B']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned_df = clean_dataset(df)
    print(cleaned_df)
    
    try:
        validate_data(cleaned_df, ['id', 'timestamp'])
        print("\nData validation passed.")
    except ValueError as e:
        print(f"\nData validation failed: {e}")