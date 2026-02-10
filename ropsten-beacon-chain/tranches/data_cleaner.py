
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
    
    def handle_missing_values(self, strategy: str = 'mean', columns: Optional[list] = None) -> 'DataCleaner':
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col in self.df.columns:
                if strategy == 'mean':
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                elif strategy == 'median':
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                elif strategy == 'mode':
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                elif strategy == 'drop':
                    self.df = self.df.dropna(subset=[col])
        
        return self
    
    def remove_outliers(self, columns: list, n_std: float = 3) -> 'DataCleaner':
        for col in columns:
            if col in self.df.columns:
                mean = self.df[col].mean()
                std = self.df[col].std()
                self.df = self.df[(self.df[col] >= mean - n_std * std) & 
                                 (self.df[col] <= mean + n_std * std)]
        return self
    
    def normalize_columns(self, columns: list) -> 'DataCleaner':
        for col in columns:
            if col in self.df.columns:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
        return self
    
    def get_cleaned_data(self) -> pd.DataFrame:
        return self.df
    
    def get_cleaning_report(self) -> dict:
        return {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_rows': self.df.shape[0],
            'cleaned_columns': self.df.shape[1],
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'columns_removed': self.original_shape[1] - self.df.shape[1]
        }

def clean_csv_file(input_path: str, output_path: str, **kwargs) -> dict:
    df = pd.read_csv(input_path)
    cleaner = DataCleaner(df)
    
    if 'remove_duplicates' in kwargs and kwargs['remove_duplicates']:
        cleaner.remove_duplicates()
    
    if 'missing_strategy' in kwargs:
        columns = kwargs.get('missing_columns')
        cleaner.handle_missing_values(strategy=kwargs['missing_strategy'], columns=columns)
    
    if 'outlier_columns' in kwargs:
        n_std = kwargs.get('outlier_std', 3)
        cleaner.remove_outliers(columns=kwargs['outlier_columns'], n_std=n_std)
    
    if 'normalize_columns' in kwargs:
        cleaner.normalize_columns(columns=kwargs['normalize_columns'])
    
    cleaned_df = cleaner.get_cleaned_data()
    cleaned_df.to_csv(output_path, index=False)
    
    return cleaner.get_cleaning_report()