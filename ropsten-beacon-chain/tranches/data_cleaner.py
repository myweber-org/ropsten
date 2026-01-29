
import pandas as pd
import numpy as np
from pathlib import Path

class DataCleaner:
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.df = None
        self.load_data()
    
    def load_data(self):
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        if self.file_path.suffix == '.csv':
            self.df = pd.read_csv(self.file_path)
        elif self.file_path.suffix in ['.xlsx', '.xls']:
            self.df = pd.read_excel(self.file_path)
        else:
            raise ValueError("Unsupported file format")
        
        print(f"Loaded data with shape: {self.df.shape}")
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if self.df is None:
            raise ValueError("No data loaded")
        
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
                    raise ValueError(f"Unknown strategy: {strategy}")
                
                self.df[col].fillna(fill_value, inplace=True)
                print(f"Filled missing values in '{col}' with {strategy}: {fill_value}")
    
    def remove_duplicates(self, subset=None, keep='first'):
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        removed = initial_rows - len(self.df)
        print(f"Removed {removed} duplicate rows")
    
    def normalize_numeric(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col in self.df.columns:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
                    print(f"Normalized column '{col}' to range [0, 1]")
    
    def save_cleaned_data(self, output_path=None):
        if output_path is None:
            output_path = self.file_path.parent / f"cleaned_{self.file_path.name}"
        
        if self.file_path.suffix == '.csv':
            self.df.to_csv(output_path, index=False)
        elif self.file_path.suffix in ['.xlsx', '.xls']:
            self.df.to_excel(output_path, index=False)
        
        print(f"Saved cleaned data to: {output_path}")
        return output_path
    
    def get_summary(self):
        summary = {
            'original_file': str(self.file_path),
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'missing_values': self.df.isnull().sum().sum(),
            'duplicates': len(self.df) - len(self.df.drop_duplicates()),
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(include=['object']).columns)
        }
        return summary

def process_csv_file(input_file, output_dir='cleaned_data'):
    cleaner = DataCleaner(input_file)
    
    print("Data Summary:")
    summary = cleaner.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    cleaner.handle_missing_values(strategy='mean')
    cleaner.remove_duplicates()
    cleaner.normalize_numeric()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    output_path = cleaner.save_cleaned_data(output_dir / f"cleaned_{Path(input_file).name}")
    return output_path

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'age': [25, 30, None, 35, 40, 40],
        'salary': [50000, 60000, 70000, None, 90000, 90000],
        'department': ['IT', 'HR', 'IT', 'Finance', 'IT', 'IT']
    }
    
    test_df = pd.DataFrame(sample_data)
    test_file = 'test_data.csv'
    test_df.to_csv(test_file, index=False)
    
    try:
        result = process_csv_file(test_file)
        print(f"\nProcessing complete. Output file: {result}")
        
        cleaned_df = pd.read_csv(result)
        print("\nCleaned data preview:")
        print(cleaned_df.head())
        
    finally:
        if Path(test_file).exists():
            Path(test_file).unlink()
        if Path('cleaned_data').exists():
            for f in Path('cleaned_data').glob('*'):
                f.unlink()
            Path('cleaned_data').rmdir()