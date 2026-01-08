
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to drop duplicate rows
        fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if cleaned_df.isnull().sum().sum() > 0:
        print(f"Found {cleaned_df.isnull().sum().sum()} missing values")
        
        if fill_missing == 'drop':
            cleaned_df = cleaned_df.dropna()
            print("Dropped rows with missing values")
        elif fill_missing == 'mean':
            for column in cleaned_df.select_dtypes(include=[np.number]).columns:
                if cleaned_df[column].isnull().sum() > 0:
                    cleaned_df[column].fillna(cleaned_df[column].mean(), inplace=True)
            print("Filled missing values with column means")
        elif fill_missing == 'median':
            for column in cleaned_df.select_dtypes(include=[np.number]).columns:
                if cleaned_df[column].isnull().sum() > 0:
                    cleaned_df[column].fillna(cleaned_df[column].median(), inplace=True)
            print("Filled missing values with column medians")
        elif fill_missing == 'mode':
            for column in cleaned_df.columns:
                if cleaned_df[column].isnull().sum() > 0:
                    cleaned_df[column].fillna(cleaned_df[column].mode()[0], inplace=True)
            print("Filled missing values with column modes")
    
    print(f"Data cleaning complete. Final shape: {cleaned_df.shape}")
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate that a DataFrame meets basic requirements.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of column names that must be present
    
    Returns:
        bool: True if DataFrame passes validation
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if len(df) == 0:
        raise ValueError("DataFrame is empty")
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, 5, np.nan],
        'B': [10, 20, 20, 40, np.nan, 60],
        'C': ['x', 'y', 'y', 'z', 'z', 'x']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaning with mean imputation...")
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
import pandas as pd
import numpy as np
from pathlib import Path

class DataCleaner:
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.df = None
        
    def load_data(self):
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"Loaded data with shape: {self.df.shape}")
            return True
        except Exception as e:
            print(f"Error loading file: {e}")
            return False
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return
        
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col in self.df.columns:
                if self.df[col].isnull().any():
                    if strategy == 'mean':
                        fill_value = self.df[col].mean()
                    elif strategy == 'median':
                        fill_value = self.df[col].median()
                    elif strategy == 'mode':
                        fill_value = self.df[col].mode()[0]
                    elif strategy == 'zero':
                        fill_value = 0
                    else:
                        fill_value = strategy
                    
                    self.df[col].fillna(fill_value, inplace=True)
                    print(f"Filled missing values in column '{col}' using {strategy}")
    
    def remove_duplicates(self, subset=None, keep='first'):
        initial_count = len(self.df)
        self.df.drop_duplicates(subset=subset, keep=keep, inplace=True)
        removed = initial_count - len(self.df)
        print(f"Removed {removed} duplicate rows")
    
    def normalize_column(self, column_name):
        if column_name in self.df.columns and self.df[column_name].dtype in [np.float64, np.int64]:
            col_min = self.df[column_name].min()
            col_max = self.df[column_name].max()
            
            if col_max != col_min:
                self.df[column_name] = (self.df[column_name] - col_min) / (col_max - col_min)
                print(f"Normalized column '{column_name}' to range [0, 1]")
            else:
                print(f"Column '{column_name}' has constant values, skipping normalization")
    
    def save_cleaned_data(self, output_path=None):
        if output_path is None:
            output_path = self.file_path.parent / f"cleaned_{self.file_path.name}"
        
        self.df.to_csv(output_path, index=False)
        print(f"Saved cleaned data to: {output_path}")
        return output_path
    
    def get_summary(self):
        if self.df is not None:
            summary = {
                'original_file': str(self.file_path),
                'rows': len(self.df),
                'columns': len(self.df.columns),
                'missing_values': self.df.isnull().sum().sum(),
                'duplicates': self.df.duplicated().sum(),
                'data_types': self.df.dtypes.to_dict()
            }
            return summary
        return None

def clean_csv_file(input_file, output_file=None):
    cleaner = DataCleaner(input_file)
    
    if cleaner.load_data():
        summary_before = cleaner.get_summary()
        print("Data summary before cleaning:")
        print(f"  Rows: {summary_before['rows']}")
        print(f"  Columns: {summary_before['columns']}")
        print(f"  Missing values: {summary_before['missing_values']}")
        print(f"  Duplicates: {summary_before['duplicates']}")
        
        cleaner.handle_missing_values(strategy='mean')
        cleaner.remove_duplicates()
        
        numeric_cols = cleaner.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:3]:
            cleaner.normalize_column(col)
        
        output_path = cleaner.save_cleaned_data(output_file)
        
        summary_after = cleaner.get_summary()
        print("\nData summary after cleaning:")
        print(f"  Rows: {summary_after['rows']}")
        print(f"  Columns: {summary_after['columns']}")
        print(f"  Missing values: {summary_after['missing_values']}")
        print(f"  Duplicates: {summary_after['duplicates']}")
        
        return output_path
    return None

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'value': [10.5, None, 15.2, 20.1, None, 10.5],
        'category': ['A', 'B', 'A', 'C', 'B', 'A'],
        'score': [85, 92, 78, None, 88, 85]
    }
    
    test_df = pd.DataFrame(sample_data)
    test_file = Path('test_data.csv')
    test_df.to_csv(test_file, index=False)
    
    print("Testing DataCleaner utility...")
    result = clean_csv_file('test_data.csv', 'cleaned_test_data.csv')
    
    if result:
        print(f"\nCleaning completed successfully. Output: {result}")
    
    test_file.unlink(missing_ok=True)