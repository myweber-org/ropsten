
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to drop duplicate rows
        fill_missing (str): Strategy for filling missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
        print("Dropped rows with missing values")
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        print(f"Filled missing numeric values with {fill_missing}")
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 'Unknown')
        print("Filled missing categorical values with mode")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
        min_rows (int): Minimum number of rows required
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if df.empty:
        print("Validation failed: DataFrame is empty")
        return False
    
    if len(df) < min_rows:
        print(f"Validation failed: DataFrame has fewer than {min_rows} rows")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing required columns: {missing_cols}")
            return False
    
    return True

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers from a specific column using IQR method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to process
        multiplier (float): IQR multiplier for outlier detection
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        print(f"Column '{column}' not found in DataFrame")
        return df
    
    if not np.issubdtype(df[column].dtype, np.number):
        print(f"Column '{column}' is not numeric")
        return df
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    initial_count = len(df)
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    removed_count = initial_count - len(filtered_df)
    
    print(f"Removed {removed_count} outliers from column '{column}'")
    return filtered_df
def remove_duplicates(input_list):
    """
    Remove duplicate elements from a list while preserving order.
    
    Args:
        input_list: A list that may contain duplicate elements.
    
    Returns:
        A new list with duplicates removed.
    """
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def clean_data_with_threshold(data, threshold=None):
    """
    Clean data by removing duplicates, optionally filtering by frequency threshold.
    
    Args:
        data: List of items to clean.
        threshold: Minimum frequency count to keep an item (inclusive).
    
    Returns:
        Cleaned list of items.
    """
    from collections import Counter
    
    if not data:
        return []
    
    cleaned = remove_duplicates(data)
    
    if threshold is not None and threshold > 0:
        counts = Counter(data)
        cleaned = [item for item in cleaned if counts[item] >= threshold]
    
    return cleaned

if __name__ == "__main__":
    sample_data = [1, 2, 2, 3, 4, 4, 4, 5, 1, 6]
    print(f"Original data: {sample_data}")
    print(f"Without duplicates: {remove_duplicates(sample_data)}")
    print(f"Threshold >= 2: {clean_data_with_threshold(sample_data, threshold=2)}")
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
            print(f"Loaded data from {self.file_path}")
            return True
        except FileNotFoundError:
            print(f"File not found: {self.file_path}")
            return False
            
    def handle_missing_values(self, strategy='mean', columns=None):
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return
            
        if columns is None:
            columns = self.df.columns
            
        for column in columns:
            if column in self.df.columns:
                if self.df[column].isnull().any():
                    if strategy == 'mean':
                        fill_value = self.df[column].mean()
                    elif strategy == 'median':
                        fill_value = self.df[column].median()
                    elif strategy == 'mode':
                        fill_value = self.df[column].mode()[0]
                    else:
                        fill_value = strategy
                    
                    self.df[column].fillna(fill_value, inplace=True)
                    print(f"Filled missing values in {column} using {strategy}")
                    
    def remove_duplicates(self):
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return
            
        initial_count = len(self.df)
        self.df.drop_duplicates(inplace=True)
        removed = initial_count - len(self.df)
        print(f"Removed {removed} duplicate rows")
        
    def normalize_column(self, column_name):
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return
            
        if column_name in self.df.columns:
            col_min = self.df[column_name].min()
            col_max = self.df[column_name].max()
            
            if col_max != col_min:
                self.df[column_name] = (self.df[column_name] - col_min) / (col_max - col_min)
                print(f"Normalized column {column_name}")
            else:
                print(f"Cannot normalize {column_name}: all values are identical")
                
    def save_cleaned_data(self, output_path=None):
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return
            
        if output_path is None:
            output_path = self.file_path.parent / f"cleaned_{self.file_path.name}"
            
        self.df.to_csv(output_path, index=False)
        print(f"Saved cleaned data to {output_path}")
        return output_path
        
    def get_summary(self):
        if self.df is None:
            return "No data loaded"
            
        summary = {
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'missing_values': self.df.isnull().sum().sum(),
            'duplicates': len(self.df) - len(self.df.drop_duplicates()),
            'data_types': self.df.dtypes.to_dict()
        }
        return summary

def clean_csv_file(input_file, output_file=None):
    cleaner = DataCleaner(input_file)
    
    if cleaner.load_data():
        print("Starting data cleaning process...")
        print(f"Initial summary: {cleaner.get_summary()}")
        
        cleaner.handle_missing_values(strategy='mean')
        cleaner.remove_duplicates()
        
        numeric_columns = cleaner.df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            cleaner.normalize_column(col)
            
        output_path = cleaner.save_cleaned_data(output_file)
        print(f"Final summary: {cleaner.get_summary()}")
        print(f"Cleaning complete. Output saved to: {output_path}")
        
        return True
    return False

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 5, 5],
        'B': [10, 20, 30, np.nan, 50, 50],
        'C': ['x', 'y', 'z', 'x', 'y', 'z']
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    clean_csv_file('test_data.csv', 'cleaned_test_data.csv')