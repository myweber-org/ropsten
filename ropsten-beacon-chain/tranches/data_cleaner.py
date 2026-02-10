import re
import pandas as pd
from typing import Union, List, Optional

def remove_duplicates(data: Union[List, pd.Series, pd.DataFrame]) -> Union[List, pd.Series, pd.DataFrame]:
    """
    Remove duplicate entries from a list, Series, or DataFrame.
    For DataFrames, it removes duplicate rows.
    """
    if isinstance(data, list):
        seen = set()
        return [x for x in data if not (x in seen or seen.add(x))]
    elif isinstance(data, pd.Series):
        return data.drop_duplicates()
    elif isinstance(data, pd.DataFrame):
        return data.drop_duplicates()
    else:
        raise TypeError("Input must be a list, pandas Series, or pandas DataFrame")

def validate_email(email: str) -> bool:
    """
    Validate an email address using a regular expression.
    Returns True if the email format is valid, False otherwise.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def clean_numeric_string(value: str) -> Optional[float]:
    """
    Clean a numeric string by removing non-numeric characters
    (except decimal point and negative sign) and convert to float.
    Returns None if the cleaned string cannot be converted.
    """
    if not isinstance(value, str):
        return None
    cleaned = re.sub(r'[^\d.-]', '', value)
    try:
        return float(cleaned)
    except ValueError:
        return None

def fill_missing_with_mean(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Fill missing values in a specified column of a DataFrame
    with the mean of that column.
    Returns a new DataFrame with missing values filled.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    df_filled = df.copy()
    mean_val = df_filled[column].mean()
    df_filled[column].fillna(mean_val, inplace=True)
    return df_filled

def standardize_phone_number(phone: str) -> Optional[str]:
    """
    Standardize a phone number to a simple format: only digits.
    Returns None if no digits are found.
    """
    digits = re.sub(r'\D', '', phone)
    return digits if digits else None