
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
            elif fill_missing == 'median':
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None, inplace=True)
    
    return cleaned_df

def validate_dataset(df, check_duplicates=True, check_missing=True):
    """
    Validate a DataFrame by checking for duplicates and missing values.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    check_duplicates (bool): Whether to check for duplicate rows.
    check_missing (bool): Whether to check for missing values.
    
    Returns:
    dict: Dictionary containing validation results.
    """
    validation_results = {}
    
    if check_duplicates:
        duplicate_count = df.duplicated().sum()
        validation_results['duplicates'] = duplicate_count
    
    if check_missing:
        missing_counts = df.isnull().sum()
        validation_results['missing_values'] = missing_counts.to_dict()
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, None],
        'B': [5, None, 7, 8, 9],
        'C': ['x', 'y', 'y', 'z', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    validation = validate_dataset(df)
    print("\nValidation Results:")
    print(validation)
    
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)import csv
import os
from typing import List, Dict, Optional

def read_csv(filepath: str) -> List[Dict[str, str]]:
    """Read a CSV file and return a list of dictionaries."""
    data = []
    try:
        with open(filepath, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append(row)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
    except Exception as e:
        print(f"Error reading CSV: {e}")
    return data

def clean_numeric(value: str) -> Optional[float]:
    """Clean and convert a string to float, removing non-numeric characters."""
    if not value:
        return None
    cleaned = ''.join(ch for ch in value if ch.isdigit() or ch == '.')
    try:
        return float(cleaned) if cleaned else None
    except ValueError:
        return None

def remove_duplicates(data: List[Dict], key: str) -> List[Dict]:
    """Remove duplicate rows based on a specified key."""
    seen = set()
    unique_data = []
    for row in data:
        if key in row and row[key] not in seen:
            seen.add(row[key])
            unique_data.append(row)
    return unique_data

def write_csv(data: List[Dict], filepath: str) -> bool:
    """Write data to a CSV file."""
    if not data:
        print("No data to write.")
        return False
    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        return True
    except Exception as e:
        print(f"Error writing CSV: {e}")
        return False

def filter_by_threshold(data: List[Dict], column: str, threshold: float) -> List[Dict]:
    """Filter rows where the numeric value in a column exceeds a threshold."""
    filtered = []
    for row in data:
        if column in row:
            numeric_val = clean_numeric(row[column])
            if numeric_val is not None and numeric_val > threshold:
                filtered.append(row)
    return filtered