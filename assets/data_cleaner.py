
import pandas as pd
import numpy as np

def clean_dataframe(df):
    """
    Clean a pandas DataFrame by handling missing values and standardizing text.
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Fill numeric nulls with column mean
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
    
    # Fill categorical nulls with 'Unknown'
    categorical_cols = cleaned_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        cleaned_df[col] = cleaned_df[col].fillna('Unknown')
    
    # Standardize text columns: trim whitespace and convert to lowercase
    for col in categorical_cols:
        cleaned_df[col] = cleaned_df[col].str.strip().str.lower()
    
    # Remove duplicate rows
    cleaned_df = cleaned_df.drop_duplicates()
    
    return cleaned_df

def validate_dataframe(df):
    """
    Validate that DataFrame meets basic quality requirements.
    """
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    # Check for remaining null values
    null_counts = df.isnull().sum()
    if null_counts.any():
        problematic_cols = null_counts[null_counts > 0].index.tolist()
        print(f"Warning: Columns with null values: {problematic_cols}")
    
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'name': ['Alice', 'Bob', None, 'Charlie', 'Alice'],
        'age': [25, None, 30, 35, 25],
        'city': ['New York', '  los angeles  ', 'Chicago', None, 'New York']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned = clean_dataframe(df)
    print(cleaned)
    
    try:
        validate_dataframe(cleaned)
        print("\nData validation passed")
    except ValueError as e:
        print(f"\nData validation failed: {e}")