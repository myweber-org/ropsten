
import pandas as pd
import re

def clean_dataframe(df, columns_to_clean=None):
    """
    Clean a pandas DataFrame by removing duplicate rows and normalizing string columns.
    """
    df_clean = df.copy()
    
    # Remove duplicate rows
    initial_rows = df_clean.shape[0]
    df_clean = df_clean.drop_duplicates()
    removed_duplicates = initial_rows - df_clean.shape[0]
    
    # If specific columns are provided, clean only those; otherwise, clean all object columns
    if columns_to_clean is None:
        columns_to_clean = df_clean.select_dtypes(include=['object']).columns
    else:
        columns_to_clean = [col for col in columns_to_clean if col in df_clean.columns]
    
    for column in columns_to_clean:
        df_clean[column] = df_clean[column].apply(normalize_string)
    
    return df_clean, removed_duplicates

def normalize_string(value):
    """
    Normalize a string: convert to lowercase, remove extra whitespace, and strip.
    """
    if isinstance(value, str):
        value = value.lower()
        value = re.sub(r'\s+', ' ', value)
        value = value.strip()
    return value

if __name__ == "__main__":
    sample_data = {
        'Name': ['Alice', 'Bob', 'Alice', '  Charlie  ', 'bob'],
        'Age': [25, 30, 25, 35, 30],
        'City': ['New York', 'Los Angeles', 'new york', 'Chicago', 'los angeles']
    }
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    cleaned_df, duplicates_removed = clean_dataframe(df)
    print(f"Removed {duplicates_removed} duplicate row(s).")
    print("Cleaned DataFrame:")
    print(cleaned_df)