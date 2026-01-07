
import pandas as pd
import numpy as np
from typing import Optional

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
    
    def remove_duplicates(self) -> 'DataCleaner':
        self.df = self.df.drop_duplicates()
        return self
    
    def fill_missing_numeric(self, strategy: str = 'mean', fill_value: Optional[float] = None) -> 'DataCleaner':
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if self.df[col].isnull().any():
                if strategy == 'mean':
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                elif strategy == 'median':
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                elif strategy == 'mode':
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                elif strategy == 'custom' and fill_value is not None:
                    self.df[col].fillna(fill_value, inplace=True)
        
        return self
    
    def fill_missing_categorical(self, fill_value: str = 'Unknown') -> 'DataCleaner':
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if self.df[col].isnull().any():
                self.df[col].fillna(fill_value, inplace=True)
        
        return self
    
    def remove_columns_with_high_missing(self, threshold: float = 0.5) -> 'DataCleaner':
        missing_ratios = self.df.isnull().sum() / len(self.df)
        cols_to_drop = missing_ratios[missing_ratios > threshold].index
        self.df = self.df.drop(columns=cols_to_drop)
        return self
    
    def get_cleaned_data(self) -> pd.DataFrame:
        return self.df
    
    def get_cleaning_report(self) -> dict:
        report = {
            'original_shape': self.original_shape,
            'cleaned_shape': self.df.shape,
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'columns_removed': self.original_shape[1] - self.df.shape[1],
            'missing_values_before': self.df.isnull().sum().sum(),
            'missing_values_after': 0
        }
        return report

def clean_csv_file(input_path: str, output_path: str) -> dict:
    try:
        df = pd.read_csv(input_path)
        cleaner = DataCleaner(df)
        
        cleaner.remove_duplicates() \
               .remove_columns_with_high_missing(threshold=0.7) \
               .fill_missing_numeric(strategy='median') \
               .fill_missing_categorical(fill_value='Missing')
        
        cleaned_df = cleaner.get_cleaned_data()
        cleaned_df.to_csv(output_path, index=False)
        
        return cleaner.get_cleaning_report()
    
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        return {}
    except Exception as e:
        print(f"Error during cleaning: {str(e)}")
        return {}