
import pandas as pd
import numpy as np

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
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for basic integrity checks.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    tuple: (bool, str) indicating validation result and message.
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame validation passed"

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, None],
        'B': [5, None, 7, 8, 9],
        'C': ['x', 'y', 'y', 'z', None]
    }
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid, message = validate_dataframe(cleaned, required_columns=['A', 'B'])
    print(f"\nValidation: {is_valid}, Message: {message}")
import numpy as np
import pandas as pd

def remove_outliers_iqr(dataframe, column):
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]

def normalize_column(dataframe, column):
    min_val = dataframe[column].min()
    max_val = dataframe[column].max()
    if max_val - min_val == 0:
        return dataframe[column].apply(lambda x: 0)
    return (dataframe[column] - min_val) / (max_val - min_val)

def clean_dataset(dataframe, numeric_columns):
    df_clean = dataframe.copy()
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean = remove_outliers_iqr(df_clean, col)
            df_clean[col] = normalize_column(df_clean, col)
    return df_clean

def calculate_statistics(dataframe, column):
    if column not in dataframe.columns:
        return {}
    series = dataframe[column]
    return {
        'mean': np.mean(series),
        'median': np.median(series),
        'std': np.std(series),
        'min': np.min(series),
        'max': np.max(series)
    }
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column in a dataset using the IQR method.
    
    Parameters:
    data (numpy.ndarray): The dataset.
    column (int): Index of the column to clean.
    
    Returns:
    numpy.ndarray: Dataset with outliers removed from the specified column.
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a numpy array")
    
    if column >= data.shape[1] or column < 0:
        raise IndexError("Column index out of bounds")
    
    col_data = data[:, column]
    q1 = np.percentile(col_data, 25)
    q3 = np.percentile(col_data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    mask = (col_data >= lower_bound) & (col_data <= upper_bound)
    cleaned_data = data[mask]
    
    return cleaned_data
import csv
import sys
from typing import Dict, List, Any, Optional

class DataCleaner:
    def __init__(self, input_file: str, output_file: str):
        self.input_file = input_file
        self.output_file = output_file
        self.data = []
        self.headers = []

    def load_data(self) -> None:
        try:
            with open(self.input_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                self.headers = reader.fieldnames or []
                self.data = [row for row in reader]
        except FileNotFoundError:
            print(f"Error: File '{self.input_file}' not found.")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading data: {e}")
            sys.exit(1)

    def handle_missing_values(self, default_values: Optional[Dict[str, Any]] = None) -> None:
        if default_values is None:
            default_values = {}

        for row in self.data:
            for header in self.headers:
                if row.get(header, '').strip() == '':
                    row[header] = default_values.get(header, 'N/A')

    def convert_types(self, type_map: Dict[str, str]) -> None:
        for row in self.data:
            for header, target_type in type_map.items():
                if header in row:
                    try:
                        if target_type == 'int':
                            row[header] = int(float(row[header])) if row[header] else 0
                        elif target_type == 'float':
                            row[header] = float(row[header]) if row[header] else 0.0
                        elif target_type == 'bool':
                            row[header] = row[header].lower() in ('true', '1', 'yes', 'y')
                    except (ValueError, TypeError):
                        row[header] = None

    def remove_duplicates(self, key_columns: List[str]) -> None:
        seen = set()
        unique_data = []

        for row in self.data:
            key = tuple(row.get(col, '') for col in key_columns)
            if key not in seen:
                seen.add(key)
                unique_data.append(row)

        self.data = unique_data

    def save_data(self) -> None:
        try:
            with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.headers)
                writer.writeheader()
                writer.writerows(self.data)
            print(f"Cleaned data saved to '{self.output_file}'")
        except Exception as e:
            print(f"Error saving data: {e}")
            sys.exit(1)

    def get_summary(self) -> Dict[str, Any]:
        return {
            'total_rows': len(self.data),
            'headers': self.headers,
            'sample_row': self.data[0] if self.data else {}
        }

def main():
    if len(sys.argv) != 3:
        print("Usage: python data_cleaner.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    cleaner = DataCleaner(input_file, output_file)
    cleaner.load_data()

    print(f"Loaded {len(cleaner.data)} rows with headers: {cleaner.headers}")

    cleaner.handle_missing_values({'age': 0, 'salary': 0.0})
    cleaner.convert_types({'age': 'int', 'salary': 'float', 'active': 'bool'})
    cleaner.remove_duplicates(['id', 'email'])

    cleaner.save_data()
    summary = cleaner.get_summary()
    print(f"Processing complete. Final row count: {summary['total_rows']}")

if __name__ == "__main__":
    main()