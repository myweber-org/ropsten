import csv
import os

def read_csv_file(file_path):
    """Read a CSV file and return its data as a list of dictionaries."""
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

def clean_numeric_fields(data, fields):
    """Clean specified numeric fields by removing non-numeric characters and converting to float."""
    cleaned_data = []
    for row in data:
        cleaned_row = row.copy()
        for field in fields:
            if field in cleaned_row:
                value = cleaned_row[field]
                if isinstance(value, str):
                    cleaned_value = ''.join(char for char in value if char.isdigit() or char == '.')
                    try:
                        cleaned_row[field] = float(cleaned_value) if cleaned_value else 0.0
                    except ValueError:
                        cleaned_row[field] = 0.0
        cleaned_data.append(cleaned_row)
    return cleaned_data

def write_csv_file(data, file_path):
    """Write data to a CSV file."""
    if not data:
        print("No data to write.")
        return False
    try:
        with open(file_path, mode='w', newline='', encoding='utf-8') as file:
            fieldnames = data[0].keys()
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        print(f"Data successfully written to '{file_path}'.")
        return True
    except Exception as e:
        print(f"Error writing file: {e}")
        return False

def main():
    input_file = "input_data.csv"
    output_file = "cleaned_data.csv"
    numeric_fields = ["price", "quantity", "score"]

    data = read_csv_file(input_file)
    if data:
        cleaned_data = clean_numeric_fields(data, numeric_fields)
        write_csv_file(cleaned_data, output_file)

if __name__ == "__main__":
    main()import pandas as pd
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
    
    return filtered_df.reset_index(drop=True)

def calculate_summary_stats(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to analyze
    
    Returns:
        dict: Dictionary containing summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def process_dataset(file_path, column_to_clean):
    """
    Load a dataset, clean outliers, and return processed data with statistics.
    
    Args:
        file_path (str): Path to CSV file
        column_to_clean (str): Column name to clean
    
    Returns:
        tuple: (cleaned_df, original_stats, cleaned_stats)
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    
    original_stats = calculate_summary_stats(df, column_to_clean)
    cleaned_df = remove_outliers_iqr(df, column_to_clean)
    cleaned_stats = calculate_summary_stats(cleaned_df, column_to_clean)
    
    return cleaned_df, original_stats, cleaned_stats

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'values': np.concatenate([
            np.random.normal(100, 10, 95),
            np.random.normal(300, 10, 5)
        ])
    })
    
    print("Original data shape:", sample_data.shape)
    cleaned_data = remove_outliers_iqr(sample_data, 'values')
    print("Cleaned data shape:", cleaned_data.shape)
    
    original_stats = calculate_summary_stats(sample_data, 'values')
    cleaned_stats = calculate_summary_stats(cleaned_data, 'values')
    
    print("\nOriginal statistics:", original_stats)
    print("Cleaned statistics:", cleaned_stats)