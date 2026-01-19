
import pandas as pd
import re

def clean_string_column(series, case='lower', strip=True, remove_special=True):
    """
    Standardize string values in a pandas Series.
    
    Args:
        series (pd.Series): Input series containing string data.
        case (str): Desired case transformation. Options: 'lower', 'upper', 'title', None.
        strip (bool): Whether to strip leading/trailing whitespace.
        remove_special (bool): Whether to remove special characters (keeping alphanumeric and spaces).
    
    Returns:
        pd.Series: Cleaned series.
    """
    if not pd.api.types.is_string_dtype(series):
        series = series.astype(str)
    
    result = series.copy()
    
    if strip:
        result = result.str.strip()
    
    if remove_special:
        result = result.apply(lambda x: re.sub(r'[^A-Za-z0-9\s]', '', x) if pd.notnull(x) else x)
    
    if case == 'lower':
        result = result.str.lower()
    elif case == 'upper':
        result = result.str.upper()
    elif case == 'title':
        result = result.str.title()
    
    return result

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame with additional logging.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        subset (list, optional): Columns to consider for duplicates.
        keep (str): Which duplicates to keep. Options: 'first', 'last', False.
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    initial_count = len(df)
    df_clean = df.drop_duplicates(subset=subset, keep=keep)
    final_count = len(df_clean)
    
    duplicates_removed = initial_count - final_count
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate rows.")
    
    return df_clean

def validate_email_format(series):
    """
    Validate email format in a pandas Series.
    
    Args:
        series (pd.Series): Series containing email addresses.
    
    Returns:
        pd.Series: Boolean series indicating valid emails.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return series.str.match(pattern, na=False)

def main():
    """
    Example usage of data cleaning functions.
    """
    sample_data = {
        'name': ['  John Doe  ', 'Jane Smith', 'ALICE WONDER', 'bob@example'],
        'email': ['john@example.com', 'invalid-email', 'alice@company.co.uk', 'bob@test.org'],
        'value': [100, 200, 100, 300]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    df['name_clean'] = clean_string_column(df['name'], case='title', strip=True, remove_special=True)
    df['email_valid'] = validate_email_format(df['email'])
    
    print("After cleaning:")
    print(df)
    print()
    
    df_no_dupes = remove_duplicates(df, subset=['value'], keep='first')
    print("After removing duplicates by 'value' column:")
    print(df_no_dupes)

if __name__ == "__main__":
    main()