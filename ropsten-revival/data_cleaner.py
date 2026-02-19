
import pandas as pd
import numpy as np
from pathlib import Path

class CSVDataCleaner:
    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.df = None
        self.original_shape = None
        
    def load_data(self):
        try:
            self.df = pd.read_csv(self.filepath)
            self.original_shape = self.df.shape
            print(f"Loaded data with shape: {self.original_shape}")
            return True
        except FileNotFoundError:
            print(f"Error: File {self.filepath} not found")
            return False
        except Exception as e:
            print(f"Error loading file: {e}")
            return False
    
    def remove_duplicates(self):
        if self.df is not None:
            initial_rows = len(self.df)
            self.df.drop_duplicates(inplace=True)
            removed = initial_rows - len(self.df)
            print(f"Removed {removed} duplicate rows")
            return removed
        return 0
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if self.df is None:
            print("No data loaded")
            return
        
        if columns is None:
            columns = self.df.columns
        
        for col in columns:
            if col in self.df.columns and self.df[col].isnull().any():
                if strategy == 'mean' and pd.api.types.is_numeric_dtype(self.df[col]):
                    fill_value = self.df[col].mean()
                elif strategy == 'median' and pd.api.types.is_numeric_dtype(self.df[col]):
                    fill_value = self.df[col].median()
                elif strategy == 'mode':
                    fill_value = self.df[col].mode()[0] if not self.df[col].mode().empty else np.nan
                elif strategy == 'drop':
                    self.df = self.df.dropna(subset=[col])
                    print(f"Dropped rows with missing values in column: {col}")
                    continue
                else:
                    fill_value = 0 if pd.api.types.is_numeric_dtype(self.df[col]) else 'Unknown'
                
                missing_count = self.df[col].isnull().sum()
                self.df[col].fillna(fill_value, inplace=True)
                print(f"Filled {missing_count} missing values in '{col}' with {strategy}: {fill_value}")
    
    def normalize_numeric_columns(self, columns=None):
        if self.df is None:
            print("No data loaded")
            return
        
        if columns is None:
            columns = [col for col in self.df.columns if pd.api.types.is_numeric_dtype(self.df[col])]
        
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                col_min = self.df[col].min()
                col_max = self.df[col].max()
                
                if col_max > col_min:
                    self.df[col] = (self.df[col] - col_min) / (col_max - col_min)
                    print(f"Normalized column '{col}' to range [0, 1]")
                else:
                    print(f"Column '{col}' has no variation (min=max={col_min})")
    
    def save_cleaned_data(self, output_path=None):
        if self.df is None:
            print("No data to save")
            return False
        
        if output_path is None:
            output_path = self.filepath.parent / f"cleaned_{self.filepath.name}"
        
        try:
            self.df.to_csv(output_path, index=False)
            print(f"Saved cleaned data to: {output_path}")
            print(f"Original shape: {self.original_shape}, Cleaned shape: {self.df.shape}")
            return True
        except Exception as e:
            print(f"Error saving file: {e}")
            return False
    
    def get_summary(self):
        if self.df is None:
            return "No data loaded"
        
        summary = {
            'original_shape': self.original_shape,
            'current_shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'numeric_columns': [col for col in self.df.columns if pd.api.types.is_numeric_dtype(self.df[col])],
            'categorical_columns': [col for col in self.df.columns if pd.api.types.is_object_dtype(self.df[col])]
        }
        return summary

def clean_csv_file(input_file, output_file=None):
    cleaner = CSVDataCleaner(input_file)
    
    if not cleaner.load_data():
        return None
    
    cleaner.remove_duplicates()
    cleaner.handle_missing_values(strategy='mean')
    cleaner.normalize_numeric_columns()
    
    if output_file:
        cleaner.save_cleaned_data(output_file)
    else:
        cleaner.save_cleaned_data()
    
    return cleaner.get_summary()

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'name': ['Alice', 'Bob', 'Charlie', None, 'Eve', 'Eve'],
        'age': [25, 30, None, 40, 35, 35],
        'score': [85.5, 92.0, 78.5, 88.0, 95.5, 95.5],
        'department': ['HR', 'IT', 'IT', None, 'Finance', 'Finance']
    }
    
    test_df = pd.DataFrame(sample_data)
    test_file = 'test_data.csv'
    test_df.to_csv(test_file, index=False)
    
    print("Testing CSVDataCleaner...")
    summary = clean_csv_file(test_file, 'cleaned_test_data.csv')
    
    import os
    if os.path.exists(test_file):
        os.remove(test_file)
    
    print("\nCleaning complete!")
def clean_data(data):
    """
    Remove duplicate entries from a list and sort the remaining items.
    """
    if not isinstance(data, list):
        raise TypeError("Input must be a list")
    
    unique_data = list(set(data))
    unique_data.sort()
    return unique_data

def validate_data(data, expected_type):
    """
    Validate that all items in the list are of the expected type.
    """
    if not isinstance(data, list):
        raise TypeError("Input must be a list")
    
    for item in data:
        if not isinstance(item, expected_type):
            raise TypeError(f"All items must be of type {expected_type}")
    
    return True

def process_data(raw_data, data_type):
    """
    Main function to clean and validate data.
    """
    try:
        validate_data(raw_data, data_type)
        cleaned_data = clean_data(raw_data)
        return cleaned_data
    except Exception as e:
        print(f"Error processing data: {e}")
        return []import numpy as np
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
    if max_val == min_val:
        return df[column]
    return (df[column] - min_val) / (max_val - min_val)

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    return cleaned_df.reset_index(drop=True)

def validate_cleaning(df_before, df_after, column):
    stats_before = {
        'mean': df_before[column].mean(),
        'std': df_before[column].std(),
        'min': df_before[column].min(),
        'max': df_before[column].max()
    }
    stats_after = {
        'mean': df_after[column].mean(),
        'std': df_after[column].std(),
        'min': df_after[column].min(),
        'max': df_after[column].max()
    }
    return stats_before, stats_after