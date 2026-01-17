
import pandas as pd
import hashlib

def remove_duplicates(input_file, output_file, key_columns=None):
    """
    Load a CSV file, remove duplicate rows based on specified columns,
    and save the cleaned data to a new file.
    """
    try:
        df = pd.read_csv(input_file)
        original_count = len(df)
        
        if key_columns is None:
            key_columns = df.columns.tolist()
        
        df_cleaned = df.drop_duplicates(subset=key_columns, keep='first')
        cleaned_count = len(df_cleaned)
        
        df_cleaned.to_csv(output_file, index=False)
        
        print(f"Original records: {original_count}")
        print(f"Cleaned records: {cleaned_count}")
        print(f"Duplicates removed: {original_count - cleaned_count}")
        print(f"Cleaned data saved to: {output_file}")
        
        return df_cleaned
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def generate_data_hash(df):
    """
    Generate a hash for the dataframe to verify data integrity.
    """
    data_string = df.to_string(index=False).encode('utf-8')
    return hashlib.md5(data_string).hexdigest()

if __name__ == "__main__":
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    
    cleaned_data = remove_duplicates(input_csv, output_csv)
    
    if cleaned_data is not None:
        data_hash = generate_data_hash(cleaned_data)
        print(f"Data integrity hash: {data_hash}")
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to remove duplicate rows.
    fill_missing (str): Method to fill missing values - 'mean', 'median', or 'drop'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    missing_before = cleaned_df.isnull().sum().sum()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    
    missing_after = cleaned_df.isnull().sum().sum()
    print(f"Missing values before: {missing_before}, after: {missing_after}")
    
    return cleaned_df

def validate_dataframe(df):
    """
    Validate DataFrame structure and data quality.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    
    Returns:
    dict: Dictionary containing validation results.
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict(),
        'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(df.select_dtypes(include=['object']).columns)
    }
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, 5, None, 7],
        'B': [10, 20, 20, None, 50, 60, 70],
        'C': ['x', 'y', 'y', 'z', 'x', 'y', 'z']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nValidation results:")
    print(validate_dataframe(df))
    
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    print("\nValidation results after cleaning:")
    print(validate_dataframe(cleaned))
import pandas as pd
import numpy as np
from typing import List, Optional

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_duplicates(self, subset: Optional[List[str]] = None) -> pd.DataFrame:
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset)
        removed = initial_count - len(self.df)
        print(f"Removed {removed} duplicate rows")
        return self.df
    
    def handle_missing_values(self, strategy: str = 'drop', fill_value: Optional[float] = None) -> pd.DataFrame:
        if strategy == 'drop':
            self.df = self.df.dropna()
        elif strategy == 'fill':
            if fill_value is not None:
                self.df = self.df.fillna(fill_value)
            else:
                self.df = self.df.fillna(self.df.mean())
        else:
            raise ValueError("Strategy must be 'drop' or 'fill'")
        
        null_count = self.df.isnull().sum().sum()
        print(f"Missing values after handling: {null_count}")
        return self.df
    
    def normalize_column(self, column: str) -> pd.DataFrame:
        if column not in self.df.columns:
            raise KeyError(f"Column {column} not found in DataFrame")
        
        col_min = self.df[column].min()
        col_max = self.df[column].max()
        
        if col_max == col_min:
            self.df[column] = 0
        else:
            self.df[column] = (self.df[column] - col_min) / (col_max - col_min)
        
        print(f"Normalized column {column} to range [0, 1]")
        return self.df
    
    def get_cleaning_report(self) -> dict:
        final_shape = self.df.shape
        return {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'final_rows': final_shape[0],
            'final_columns': final_shape[1],
            'rows_removed': self.original_shape[0] - final_shape[0],
            'null_values': int(self.df.isnull().sum().sum())
        }

def create_sample_data() -> pd.DataFrame:
    data = {
        'id': [1, 2, 3, 4, 5, 5, 6],
        'value': [10.5, np.nan, 15.2, 20.1, 25.0, 25.0, 30.7],
        'category': ['A', 'B', 'A', 'C', 'B', 'B', 'A']
    }
    return pd.DataFrame(data)

if __name__ == "__main__":
    sample_df = create_sample_data()
    cleaner = DataCleaner(sample_df)
    
    print("Original data:")
    print(sample_df)
    print("\nCleaning process...")
    
    cleaner.remove_duplicates(subset=['id'])
    cleaner.handle_missing_values(strategy='fill', fill_value=0)
    cleaner.normalize_column('value')
    
    print("\nCleaned data:")
    print(cleaner.df)
    
    report = cleaner.get_cleaning_report()
    print("\nCleaning report:")
    for key, value in report.items():
        print(f"{key}: {value}")