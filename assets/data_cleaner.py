
import pandas as pd
import numpy as np
from pathlib import Path

class DataCleaner:
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.df = None
        
    def load_data(self):
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        if self.file_path.suffix == '.csv':
            self.df = pd.read_csv(self.file_path)
        elif self.file_path.suffix in ['.xlsx', '.xls']:
            self.df = pd.read_excel(self.file_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel files.")
        
        print(f"Loaded {len(self.df)} rows and {len(self.df.columns)} columns")
        return self.df
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col not in self.df.columns:
                continue
                
            missing_count = self.df[col].isnull().sum()
            if missing_count > 0:
                print(f"Column '{col}' has {missing_count} missing values")
                
                if strategy == 'mean':
                    fill_value = self.df[col].mean()
                elif strategy == 'median':
                    fill_value = self.df[col].median()
                elif strategy == 'mode':
                    fill_value = self.df[col].mode()[0]
                elif strategy == 'drop':
                    self.df = self.df.dropna(subset=[col])
                    print(f"Dropped rows with missing values in column '{col}'")
                    continue
                else:
                    fill_value = strategy
                
                self.df[col].fillna(fill_value, inplace=True)
                print(f"Filled missing values in '{col}' with {fill_value}")
        
        return self.df
    
    def remove_duplicates(self, subset=None, keep='first'):
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        removed = initial_rows - len(self.df)
        
        if removed > 0:
            print(f"Removed {removed} duplicate rows")
        
        return self.df
    
    def normalize_columns(self, columns=None, method='minmax'):
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col not in self.df.columns:
                continue
            
            if method == 'minmax':
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
            elif method == 'zscore':
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    self.df[col] = (self.df[col] - mean_val) / std_val
        
        return self.df
    
    def save_cleaned_data(self, output_path=None):
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if output_path is None:
            output_path = self.file_path.parent / f"cleaned_{self.file_path.name}"
        
        if Path(output_path).suffix == '.csv':
            self.df.to_csv(output_path, index=False)
        else:
            self.df.to_excel(output_path, index=False)
        
        print(f"Cleaned data saved to: {output_path}")
        return output_path

def clean_dataset(input_file, output_file=None):
    cleaner = DataCleaner(input_file)
    cleaner.load_data()
    cleaner.handle_missing_values(strategy='mean')
    cleaner.remove_duplicates()
    cleaner.normalize_columns(method='minmax')
    return cleaner.save_cleaned_data(output_file)