def remove_duplicates(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
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
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{col}'")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'id': range(1, 101),
        'value': np.random.randn(100) * 10 + 50,
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(sample_data)
    print(f"Original DataFrame shape: {df.shape}")
    
    cleaned_df = clean_numeric_data(df, columns=['value'])
    print(f"Cleaned DataFrame shape: {cleaned_df.shape}")
    
    is_valid = validate_dataframe(cleaned_df, required_columns=['id', 'value', 'category'])
    print(f"DataFrame validation: {is_valid}")import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Original shape: {df.shape}")
        
        for col in df.select_dtypes(include=[np.number]).columns:
            df = remove_outliers_iqr(df, col)
        
        print(f"After outlier removal: {df.shape}")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df = normalize_minmax(df, col)
        
        output_path = file_path.replace('.csv', '_cleaned.csv')
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
        return df
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

if __name__ == "__main__":
    cleaned_data = clean_dataset('sample_data.csv')import csv
import os
from typing import List, Dict, Any, Optional

def read_csv_file(filepath: str) -> List[Dict[str, Any]]:
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
        print(f"Error reading CSV file: {e}")
    return data

def clean_numeric_field(record: Dict[str, Any], field: str, default: Any = 0) -> Dict[str, Any]:
    """Clean a numeric field in a record, converting to float if possible."""
    if field in record:
        try:
            record[field] = float(record[field])
        except (ValueError, TypeError):
            record[field] = default
    return record

def remove_empty_records(data: List[Dict[str, Any]], required_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Remove records that have empty values for required fields."""
    if required_fields is None:
        required_fields = []
    
    cleaned_data = []
    for record in data:
        keep = True
        for field in required_fields:
            if field not in record or not record[field]:
                keep = False
                break
        if keep:
            cleaned_data.append(record)
    return cleaned_data

def write_csv_file(data: List[Dict[str, Any]], filepath: str) -> bool:
    """Write data to a CSV file."""
    if not data:
        print("No data to write.")
        return False
    
    try:
        fieldnames = data[0].keys()
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        return True
    except Exception as e:
        print(f"Error writing CSV file: {e}")
        return False

def clean_csv_data(input_file: str, output_file: str, numeric_fields: List[str] = None, required_fields: List[str] = None) -> None:
    """Main function to clean CSV data."""
    if numeric_fields is None:
        numeric_fields = []
    
    print(f"Reading data from {input_file}")
    data = read_csv_file(input_file)
    
    if not data:
        print("No data loaded. Exiting.")
        return
    
    print(f"Loaded {len(data)} records.")
    
    cleaned_data = []
    for record in data:
        for field in numeric_fields:
            record = clean_numeric_field(record, field)
        cleaned_data.append(record)
    
    cleaned_data = remove_empty_records(cleaned_data, required_fields)
    print(f"After cleaning, {len(cleaned_data)} records remain.")
    
    if write_csv_file(cleaned_data, output_file):
        print(f"Cleaned data written to {output_file}")
    else:
        print("Failed to write cleaned data.")

if __name__ == "__main__":
    input_path = "raw_data.csv"
    output_path = "cleaned_data.csv"
    
    numeric_cols = ["price", "quantity", "rating"]
    required_cols = ["id", "name", "price"]
    
    clean_csv_data(input_path, output_path, numeric_cols, required_cols)