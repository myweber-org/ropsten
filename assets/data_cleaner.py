import pandas as pd
import re

def clean_text_column(df, column_name):
    """
    Standardize text by converting to lowercase and removing extra whitespace.
    """
    if column_name in df.columns:
        df[column_name] = df[column_name].astype(str).str.lower()
        df[column_name] = df[column_name].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
    return df

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from the DataFrame.
    """
    return df.drop_duplicates(subset=subset, keep='first')

def process_dataframe(df, text_columns=None, dedupe_subset=None):
    """
    Main function to clean text columns and remove duplicates.
    """
    if text_columns:
        for col in text_columns:
            df = clean_text_column(df, col)
    
    if dedupe_subset:
        df = remove_duplicates(df, subset=dedupe_subset)
    else:
        df = remove_duplicates(df)
    
    return df

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Alice', 'Charlie', 'bob'],
        'email': ['alice@test.com', 'bob@test.com', 'alice@test.com', 'charlie@test.com', 'Bob@Test.Com']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    processed_df = process_dataframe(
        df, 
        text_columns=['name', 'email'], 
        dedupe_subset=['email']
    )
    
    print("\nProcessed DataFrame:")
    print(processed_df)
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
        fill_missing (bool): Whether to fill missing values. Default is True.
        fill_value: Value to use for filling missing values. Default is 0.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate that a DataFrame meets basic requirements.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
    
    Returns:
        bool: True if validation passes, False otherwise.
    """
    if df.empty:
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4],
        'value': [10, 20, 20, None, 40],
        'category': ['A', 'B', 'B', 'C', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned = clean_dataset(df)
    print(cleaned)
    print(f"\nData validation result: {validate_dataframe(cleaned, ['id', 'value'])}")
def remove_duplicates_preserve_order(seq):
    seen = set()
    result = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result