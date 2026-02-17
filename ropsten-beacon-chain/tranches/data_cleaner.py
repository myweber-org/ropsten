
import pandas as pd
import re

def clean_text_column(df, column_name):
    """
    Standardize text by converting to lowercase, removing extra whitespace,
    and stripping special characters except basic punctuation.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    df[column_name] = df[column_name].astype(str).str.lower()
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'\s+', ' ', x))
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'[^\w\s.,!?-]', '', x))
    df[column_name] = df[column_name].str.strip()
    return df

def remove_duplicate_rows(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame.
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def validate_email_column(df, column_name):
    """
    Validate email format in specified column and return boolean mask.
    """
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return df[column_name].astype(str).str.match(email_pattern)

def sample_data_cleaning_pipeline():
    """
    Example pipeline demonstrating the data cleaning functions.
    """
    data = {
        'name': ['John Doe', 'Jane Smith', 'john doe', 'Bob Johnson  ', 'Alice Brown'],
        'email': ['john@example.com', 'jane@test.org', 'invalid-email', 'bob@company.net', 'alice@domain.co'],
        'notes': ['Important client!', 'Needs follow-up.', '  Regular   customer  ', 'VIP; special handling', 'New prospect.']
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50)
    
    df = clean_text_column(df, 'name')
    df = clean_text_column(df, 'notes')
    df = remove_duplicate_rows(df, subset=['name'])
    
    df['valid_email'] = validate_email_column(df, 'email')
    
    print("\nCleaned DataFrame:")
    print(df)
    return df

if __name__ == "__main__":
    cleaned_df = sample_data_cleaning_pipeline()