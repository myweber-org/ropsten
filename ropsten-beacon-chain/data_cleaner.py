import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to process
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    If no columns specified, clean all numeric columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): List of column names to clean
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
            except Exception as e:
                print(f"Warning: Could not clean column '{col}': {e}")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list, optional): List of required column names
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if not isinstance(df, pd.DataFrame):
        return False
    
    if df.empty:
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'temperature': [22, 23, 24, 25, 26, 100, 27, 28, 29, -10],
        'humidity': [45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
        'pressure': [1013, 1014, 1015, 1016, 1017, 2000, 1018, 1019, 1020, 500]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original data shape:", df.shape)
    print("Original data:")
    print(df)
    
    cleaned_df = clean_numeric_data(df)
    print("\nCleaned data shape:", cleaned_df.shape)
    print("Cleaned data:")
    print(cleaned_df)
    
    is_valid = validate_dataframe(cleaned_df, ['temperature', 'humidity', 'pressure'])
    print(f"\nData validation passed: {is_valid}")import re
from typing import List, Optional

def remove_special_characters(text: str, keep_spaces: bool = True) -> str:
    """
    Remove all non-alphanumeric characters from the input string.

    Args:
        text: The input string to clean.
        keep_spaces: If True, spaces are preserved. If False, spaces are removed.

    Returns:
        The cleaned string containing only alphanumeric characters and optionally spaces.
    """
    if keep_spaces:
        pattern = r'[^A-Za-z0-9\s]+'
    else:
        pattern = r'[^A-Za-z0-9]+'
    return re.sub(pattern, '', text)

def normalize_whitespace(text: str) -> str:
    """
    Replace multiple consecutive whitespace characters with a single space.

    Args:
        text: The input string to normalize.

    Returns:
        The string with normalized whitespace.
    """
    return re.sub(r'\s+', ' ', text).strip()

def clean_text_pipeline(
    text: str,
    remove_special: bool = True,
    normalize_space: bool = True,
    to_lowercase: bool = False
) -> str:
    """
    Apply a series of cleaning operations to the input text.

    Args:
        text: The input string to process.
        remove_special: If True, remove special characters.
        normalize_space: If True, normalize whitespace.
        to_lowercase: If True, convert the text to lowercase.

    Returns:
        The cleaned text after applying the specified operations.
    """
    result = text
    if remove_special:
        result = remove_special_characters(result, keep_spaces=True)
    if normalize_space:
        result = normalize_whitespace(result)
    if to_lowercase:
        result = result.lower()
    return result

def batch_clean_texts(
    texts: List[str],
    remove_special: bool = True,
    normalize_space: bool = True,
    to_lowercase: bool = False
) -> List[str]:
    """
    Apply cleaning operations to a list of text strings.

    Args:
        texts: A list of input strings to clean.
        remove_special: If True, remove special characters from each string.
        normalize_space: If True, normalize whitespace in each string.
        to_lowercase: If True, convert each string to lowercase.

    Returns:
        A list of cleaned strings.
    """
    return [
        clean_text_pipeline(t, remove_special, normalize_space, to_lowercase)
        for t in texts
    ]

def extract_numbers(text: str, as_strings: bool = False) -> List:
    """
    Extract all numbers from the given text.

    Args:
        text: The input string to search for numbers.
        as_strings: If True, return numbers as strings. If False, return as integers or floats.

    Returns:
        A list of extracted numbers.
    """
    numbers = re.findall(r'\b\d+\.?\d*\b', text)
    if as_strings:
        return numbers
    result = []
    for num in numbers:
        try:
            if '.' in num:
                result.append(float(num))
            else:
                result.append(int(num))
        except ValueError:
            continue
    return result

if __name__ == "__main__":
    sample_text = "Hello,   World! This is a test. 123.45 and 678 are numbers."
    print("Original:", sample_text)
    cleaned = clean_text_pipeline(sample_text, to_lowercase=True)
    print("Cleaned:", cleaned)
    numbers = extract_numbers(sample_text)
    print("Numbers found:", numbers)