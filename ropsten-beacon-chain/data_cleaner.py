
import pandas as pd
import re

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        column_mapping (dict, optional): Dictionary mapping old column names to new ones
        drop_duplicates (bool): Whether to remove duplicate rows
        normalize_text (bool): Whether to normalize text columns (strip, lower case)
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    # Rename columns if mapping provided
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    # Remove duplicate rows
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    
    # Normalize text columns
    if normalize_text:
        for col in cleaned_df.select_dtypes(include=['object']).columns:
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip().str.lower()
    
    return cleaned_df

def validate_email(email_series):
    """
    Validate email addresses in a pandas Series.
    
    Args:
        email_series (pd.Series): Series containing email addresses
    
    Returns:
        pd.Series: Boolean series indicating valid emails
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return email_series.str.match(pattern, na=False)

def remove_special_characters(text_series, keep_chars=''):
    """
    Remove special characters from text, keeping only alphanumeric and specified characters.
    
    Args:
        text_series (pd.Series): Series containing text to clean
        keep_chars (str): Additional characters to keep (e.g., ' ._-')
    
    Returns:
        pd.Series: Cleaned text series
    """
    pattern = f'[^a-zA-Z0-9{re.escape(keep_chars)}]'
    return text_series.str.replace(pattern, '', regex=True)

def main():
    # Example usage
    sample_data = {
        'name': ['  John Doe  ', 'Jane Smith', 'john doe', 'Bob Johnson  '],
        'email': ['john@example.com', 'invalid-email', 'JANE@EXAMPLE.COM', 'bob@test.org'],
        'phone': ['123-456-7890', '987.654.3210', '555 123 4567', 'invalid'],
        'value': [100, 200, 100, 300]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    # Clean the dataframe
    cleaned = clean_dataframe(df, normalize_text=True, drop_duplicates=True)
    print("Cleaned DataFrame:")
    print(cleaned)
    print()
    
    # Validate emails
    valid_emails = validate_email(cleaned['email'])
    print("Valid emails:")
    print(valid_emails)
    print()
    
    # Clean phone numbers
    cleaned_phones = remove_special_characters(cleaned['phone'], keep_chars='- .')
    print("Cleaned phone numbers:")
    print(cleaned_phones)

if __name__ == "__main__":
    main()