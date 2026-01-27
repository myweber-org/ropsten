
import pandas as pd
import re

def clean_dataframe(df, columns_to_clean=None):
    """
    Clean a pandas DataFrame by removing duplicate rows and normalizing
    specified string columns (strip whitespace, lowercase).
    """
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    initial_rows = cleaned_df.shape[0]
    cleaned_df = cleaned_df.drop_duplicates()
    removed_duplicates = initial_rows - cleaned_df.shape[0]
    
    # Normalize string columns
    if columns_to_clean is None:
        # Auto-detect string columns
        string_columns = cleaned_df.select_dtypes(include=['object']).columns
    else:
        # Use specified columns
        string_columns = [col for col in columns_to_clean if col in cleaned_df.columns]
    
    for col in string_columns:
        cleaned_df[col] = cleaned_df[col].apply(
            lambda x: re.sub(r'\s+', ' ', str(x).strip().lower()) if pd.notnull(x) else x
        )
    
    return cleaned_df, removed_duplicates

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column and return a boolean mask.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return df[email_column].str.match(email_pattern, na=False)

def save_cleaned_data(df, output_path, format='csv'):
    """
    Save cleaned DataFrame to file in specified format.
    """
    if format == 'csv':
        df.to_csv(output_path, index=False)
    elif format == 'excel':
        df.to_excel(output_path, index=False)
    elif format == 'json':
        df.to_json(output_path, orient='records')
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    return True