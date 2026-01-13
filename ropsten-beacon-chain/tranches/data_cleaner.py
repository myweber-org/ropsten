import numpy as np
import pandas as pd

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

def standardize_zscore(df, column):
    mean_val = df[column].mean()
    std_val = df[column].std()
    df[column + '_standardized'] = (df[column] - mean_val) / std_val
    return df

def handle_missing_mean(df, column):
    mean_val = df[column].mean()
    df[column].fillna(mean_val, inplace=True)
    return df

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if cleaned_df[col].isnull().sum() > 0:
            cleaned_df = handle_missing_mean(cleaned_df, col)
        cleaned_df = remove_outliers_iqr(cleaned_df, col)
        cleaned_df = normalize_minmax(cleaned_df, col)
        cleaned_df = standardize_zscore(cleaned_df, col)
    return cleaned_df
import csv
import os
from typing import List, Dict, Any

def read_csv_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Read a CSV file and return its contents as a list of dictionaries.
    """
    data = []
    try:
        with open(file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error reading file: {e}")
    return data

def clean_numeric_fields(data: List[Dict[str, Any]], fields: List[str]) -> List[Dict[str, Any]]:
    """
    Clean specified numeric fields by removing non-numeric characters and converting to float.
    """
    cleaned_data = []
    for row in data:
        cleaned_row = row.copy()
        for field in fields:
            if field in cleaned_row:
                value = cleaned_row[field]
                if isinstance(value, str):
                    cleaned_value = ''.join(ch for ch in value if ch.isdigit() or ch == '.')
                    try:
                        cleaned_row[field] = float(cleaned_value) if cleaned_value else 0.0
                    except ValueError:
                        cleaned_row[field] = 0.0
        cleaned_data.append(cleaned_row)
    return cleaned_data

def remove_empty_rows(data: List[Dict[str, Any]], key_field: str) -> List[Dict[str, Any]]:
    """
    Remove rows where the specified key field is empty or None.
    """
    return [row for row in data if row.get(key_field) not in [None, ""]]

def write_csv_file(data: List[Dict[str, Any]], file_path: str) -> bool:
    """
    Write data to a CSV file.
    """
    if not data:
        print("No data to write.")
        return False
    try:
        with open(file_path, mode='w', newline='', encoding='utf-8') as file:
            fieldnames = data[0].keys()
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        return True
    except Exception as e:
        print(f"Error writing file: {e}")
        return False

def process_csv(input_path: str, output_path: str, numeric_fields: List[str], key_field: str) -> None:
    """
    Main function to read, clean, and write CSV data.
    """
    print(f"Processing file: {input_path}")
    data = read_csv_file(input_path)
    if not data:
        return
    print(f"Read {len(data)} rows.")
    data = clean_numeric_fields(data, numeric_fields)
    data = remove_empty_rows(data, key_field)
    print(f"After cleaning: {len(data)} rows.")
    if write_csv_file(data, output_path):
        print(f"Cleaned data written to: {output_path}")
    else:
        print("Failed to write output file.")

if __name__ == "__main__":
    input_file = "input_data.csv"
    output_file = "cleaned_data.csv"
    numeric_columns = ["price", "quantity", "weight"]
    primary_key = "id"
    process_csv(input_file, output_file, numeric_columns, primary_key)