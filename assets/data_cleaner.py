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