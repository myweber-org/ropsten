
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, columns=None, factor=1.5):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_clean = self.df.copy()
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        self.df = df_clean
        removed_count = self.original_shape[0] - self.df.shape[0]
        print(f"Removed {removed_count} outliers")
        return self
        
    def normalize_data(self, columns=None, method='zscore'):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_normalized = self.df.copy()
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                if method == 'zscore':
                    df_normalized[col] = stats.zscore(self.df[col])
                elif method == 'minmax':
                    df_normalized[col] = (self.df[col] - self.df[col].min()) / (self.df[col].max() - self.df[col].min())
                elif method == 'robust':
                    median = self.df[col].median()
                    iqr = self.df[col].quantile(0.75) - self.df[col].quantile(0.25)
                    df_normalized[col] = (self.df[col] - median) / iqr
        
        self.df = df_normalized
        print(f"Normalized {len(columns)} columns using {method} method")
        return self
        
    def fill_missing(self, columns=None, strategy='mean'):
        if columns is None:
            columns = self.df.columns
        
        df_filled = self.df.copy()
        for col in columns:
            if col in self.df.columns and self.df[col].isnull().any():
                if strategy == 'mean' and pd.api.types.is_numeric_dtype(self.df[col]):
                    df_filled[col] = self.df[col].fillna(self.df[col].mean())
                elif strategy == 'median' and pd.api.types.is_numeric_dtype(self.df[col]):
                    df_filled[col] = self.df[col].fillna(self.df[col].median())
                elif strategy == 'mode':
                    df_filled[col] = self.df[col].fillna(self.df[col].mode()[0])
                elif strategy == 'ffill':
                    df_filled[col] = self.df[col].fillna(method='ffill')
                elif strategy == 'bfill':
                    df_filled[col] = self.df[col].fillna(method='bfill')
        
        self.df = df_filled
        print(f"Filled missing values using {strategy} strategy")
        return self
        
    def get_cleaned_data(self):
        return self.df
        
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'cleaned_rows': self.df.shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_columns': self.df.shape[1],
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'missing_values': self.df.isnull().sum().sum()
        }
        return summary

def clean_dataset(df, outlier_removal=True, normalization=True, fill_missing=True):
    cleaner = DataCleaner(df)
    
    if outlier_removal:
        cleaner.remove_outliers_iqr()
    
    if fill_missing:
        cleaner.fill_missing(strategy='median')
    
    if normalization:
        cleaner.normalize_data(method='zscore')
    
    return cleaner.get_cleaned_data(), cleaner.get_summary()