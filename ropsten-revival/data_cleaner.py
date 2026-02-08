
import pandas as pd
import numpy as np
from typing import List, Optional

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape

    def remove_duplicates(self, subset: Optional[List[str]] = None) -> 'DataCleaner':
        self.df = self.df.drop_duplicates(subset=subset)
        return self

    def normalize_column(self, column: str, method: str = 'minmax') -> 'DataCleaner':
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")

        if method == 'minmax':
            col_min = self.df[column].min()
            col_max = self.df[column].max()
            if col_max != col_min:
                self.df[column] = (self.df[column] - col_min) / (col_max - col_min)
        elif method == 'zscore':
            col_mean = self.df[column].mean()
            col_std = self.df[column].std()
            if col_std > 0:
                self.df[column] = (self.df[column] - col_mean) / col_std
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return self

    def fill_missing(self, column: str, strategy: str = 'mean') -> 'DataCleaner':
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")

        if strategy == 'mean':
            fill_value = self.df[column].mean()
        elif strategy == 'median':
            fill_value = self.df[column].median()
        elif strategy == 'mode':
            fill_value = self.df[column].mode()[0]
        elif strategy == 'zero':
            fill_value = 0
        else:
            raise ValueError(f"Unknown fill strategy: {strategy}")

        self.df[column] = self.df[column].fillna(fill_value)
        return self

    def get_cleaned_data(self) -> pd.DataFrame:
        return self.df

    def get_stats(self) -> dict:
        return {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_rows': self.df.shape[0],
            'cleaned_columns': self.df.shape[1],
            'rows_removed': self.original_shape[0] - self.df.shape[0]
        }

def clean_dataset(df: pd.DataFrame, 
                  deduplicate: bool = True,
                  normalize_columns: Optional[List[str]] = None,
                  fill_missing_columns: Optional[List[str]] = None) -> pd.DataFrame:
    cleaner = DataCleaner(df)

    if deduplicate:
        cleaner.remove_duplicates()

    if normalize_columns:
        for col in normalize_columns:
            cleaner.normalize_column(col)

    if fill_missing_columns:
        for col in fill_missing_columns:
            cleaner.fill_missing(col)

    return cleaner.get_cleaned_data()