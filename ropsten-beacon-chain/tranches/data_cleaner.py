import re

def normalize_string(text):
    """
    Normalize a string by converting to lowercase, removing extra whitespace,
    and stripping non-alphanumeric characters (except basic punctuation).
    """
    if not isinstance(text, str):
        return text

    # Convert to lowercase
    text = text.lower()

    # Remove extra whitespace
    text = ' '.join(text.split())

    # Keep only alphanumeric characters, spaces, and basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)

    return text.strip()

def clean_dataframe_column(df, column_name):
    """
    Apply normalize_string to a specific column in a pandas DataFrame.
    """
    try:
        import pandas as pd
        if isinstance(df, pd.DataFrame) and column_name in df.columns:
            df[column_name] = df[column_name].apply(normalize_string)
        return df
    except ImportError:
        print("Pandas is not installed. Please install pandas to use this function.")
        return df