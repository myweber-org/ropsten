import pandas as pd
import re

def clean_dataframe(df, text_column):
    """
    Clean a DataFrame by removing duplicate rows and standardizing text in a specified column.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates().reset_index(drop=True)
    
    # Standardize text: lowercase, remove extra whitespace
    if text_column in df_cleaned.columns:
        df_cleaned[text_column] = df_cleaned[text_column].apply(
            lambda x: re.sub(r'\s+', ' ', str(x).strip().lower()) if pd.notnull(x) else x
        )
    
    return df_cleaned

def validate_email(email_series):
    """
    Validate email addresses in a pandas Series.
    Returns a boolean Series indicating valid emails.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return email_series.str.match(pattern, na=False)