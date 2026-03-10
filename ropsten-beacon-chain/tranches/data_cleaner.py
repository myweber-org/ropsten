
def remove_duplicates_preserve_order(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd
import numpy as np

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, fill_missing=True):
    """
    Clean a pandas DataFrame by standardizing columns, removing duplicates,
    and handling missing values.
    """
    cleaned_df = df.copy()
    
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype in [np.float64, np.int64]:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
            elif cleaned_df[col].dtype == 'object':
                cleaned_df[col] = cleaned_df[col].fillna('Unknown')
    
    cleaned_df = cleaned_df.reset_index(drop=True)
    return cleaned_df

def standardize_text_columns(df, columns):
    """
    Standardize text columns by converting to lowercase and stripping whitespace.
    """
    df_copy = df.copy()
    for col in columns:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].astype(str).str.lower().str.strip()
    return df_copy

def validate_dataframe(df, required_columns=None, unique_columns=None):
    """
    Validate DataFrame structure and content.
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    if unique_columns:
        for col in unique_columns:
            if col in df.columns and df[col].duplicated().any():
                raise ValueError(f"Column '{col}' contains duplicate values")
    
    return True

if __name__ == "__main__":
    sample_data = {
        'Name': ['Alice', 'Bob', 'Alice', None, 'Charlie'],
        'Age': [25, 30, 25, 35, None],
        'City': ['New York', 'los angeles', 'new york', 'Chicago', 'Boston']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaned = clean_dataframe(df)
    cleaned = standardize_text_columns(cleaned, ['City'])
    
    print("Cleaned DataFrame:")
    print(cleaned)
    
    try:
        validate_dataframe(cleaned, required_columns=['Name', 'Age'])
        print("\nData validation passed")
    except ValueError as e:
        print(f"\nData validation failed: {e}")