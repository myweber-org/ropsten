import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df: pandas DataFrame to clean
        drop_duplicates: If True, remove duplicate rows
        fill_missing: Strategy for handling missing values. 
                     Can be 'mean', 'median', 'mode', or a specific value.
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing is not None:
        missing_count = cleaned_df.isnull().sum().sum()
        if missing_count > 0:
            if fill_missing == 'mean':
                cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
            elif fill_missing == 'median':
                cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
            elif fill_missing == 'mode':
                cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
            else:
                cleaned_df = cleaned_df.fillna(fill_missing)
            print(f"Filled {missing_count} missing values using {fill_missing} strategy")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate that a DataFrame meets basic requirements.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: List of column names that must be present
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"import csv
import re
from typing import List, Optional

def remove_duplicates(data: List[List[str]]) -> List[List[str]]:
    seen = set()
    unique_data = []
    for row in data:
        row_tuple = tuple(row)
        if row_tuple not in seen:
            seen.add(row_tuple)
            unique_data.append(row)
    return unique_data

def normalize_string(value: str) -> str:
    normalized = re.sub(r'\s+', ' ', value.strip())
    return normalized.lower()

def clean_csv(input_path: str, output_path: str, columns_to_clean: Optional[List[int]] = None):
    with open(input_path, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        data = list(reader)
    
    if not data:
        return
    
    headers = data[0]
    rows = data[1:]
    
    cleaned_rows = []
    for row in rows:
        cleaned_row = row.copy()
        if columns_to_clean:
            for col_index in columns_to_clean:
                if col_index < len(cleaned_row):
                    cleaned_row[col_index] = normalize_string(cleaned_row[col_index])
        cleaned_rows.append(cleaned_row)
    
    cleaned_rows = remove_duplicates(cleaned_rows)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(headers)
        writer.writerows(cleaned_rows)
    
    print(f"Cleaned data saved to {output_path}")
    print(f"Removed {len(rows) - len(cleaned_rows)} duplicate rows")

def validate_email(email: str) -> bool:
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

if __name__ == "__main__":
    sample_data = [
        ["Name", "Email", "Age"],
        ["John Doe", "john@example.com", "25"],
        ["Jane Smith", "jane@example.com", "30"],
        ["John Doe", "john@example.com", "25"],
        ["Bob Johnson", "bob@example.com", "35"]
    ]
    
    test_file = "sample_data.csv"
    output_file = "cleaned_data.csv"
    
    with open(test_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(sample_data)
    
    clean_csv(test_file, output_file, columns_to_clean=[0, 1])
    
    test_email = "test@example.com"
    print(f"Email validation for {test_email}: {validate_email(test_email)}")def remove_duplicates(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

if __name__ == "__main__":
    sample_data = [1, 2, 2, 3, 4, 4, 5]
    cleaned = remove_duplicates(sample_data)
    print(f"Original: {sample_data}")
    print(f"Cleaned: {cleaned}")