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
import pandas as pd
import numpy as np
from scipy import stats

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

def clean_dataset(file_path, numeric_columns):
    df = pd.read_csv(file_path)
    
    for col in numeric_columns:
        if col in df.columns:
            df = remove_outliers_iqr(df, col)
            df = normalize_minmax(df, col)
    
    df = df.dropna()
    return df

def calculate_statistics(df, column):
    return {
        'mean': np.mean(df[column]),
        'median': np.median(df[column]),
        'std': np.std(df[column]),
        'skewness': stats.skew(df[column])
    }

if __name__ == "__main__":
    cleaned_data = clean_dataset('sample_data.csv', ['age', 'income', 'score'])
    stats_result = calculate_statistics(cleaned_data, 'income')
    print(f"Dataset cleaned. Statistics: {stats_result}")
    cleaned_data.to_csv('cleaned_data.csv', index=False)import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_columns = df.columns.tolist()
    
    def remove_missing(self, threshold=0.3):
        missing_percent = self.df.isnull().sum() / len(self.df)
        columns_to_drop = missing_percent[missing_percent > threshold].index
        self.df = self.df.drop(columns=columns_to_drop)
        return self
    
    def fill_numeric_missing(self, method='median'):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if method == 'mean':
            fill_values = self.df[numeric_cols].mean()
        elif method == 'median':
            fill_values = self.df[numeric_cols].median()
        elif method == 'mode':
            fill_values = self.df[numeric_cols].mode().iloc[0]
        else:
            fill_values = 0
        
        self.df[numeric_cols] = self.df[numeric_cols].fillna(fill_values)
        return self
    
    def remove_outliers_zscore(self, threshold=3):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            z_scores = np.abs(stats.zscore(self.df[col].dropna()))
            outliers = z_scores > threshold
            self.df.loc[outliers, col] = np.nan
        
        return self.fill_numeric_missing('median')
    
    def normalize_numeric(self, method='minmax'):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if method == 'minmax':
            for col in numeric_cols:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
        
        elif method == 'standard':
            for col in numeric_cols:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    self.df[col] = (self.df[col] - mean_val) / std_val
        
        return self
    
    def get_cleaned_data(self):
        return self.df
    
    def get_summary(self):
        summary = {
            'original_columns': len(self.original_columns),
            'current_columns': len(self.df.columns),
            'original_rows': len(self.df),
            'current_rows': len(self.df),
            'missing_values': self.df.isnull().sum().sum(),
            'numeric_columns': len(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(self.df.select_dtypes(include=['object']).columns)
        }
        return summary

def clean_dataset(df, outlier_threshold=3, normalize_method='minmax'):
    cleaner = DataCleaner(df)
    cleaner.remove_missing(0.3)
    cleaner.fill_numeric_missing('median')
    cleaner.remove_outliers_zscore(outlier_threshold)
    cleaner.normalize_numeric(normalize_method)
    return cleaner.get_cleaned_data(), cleaner.get_summary()