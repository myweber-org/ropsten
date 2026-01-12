import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_columns = df.columns.tolist()
    
    def remove_missing(self, threshold=0.3):
        missing_percent = self.df.isnull().sum() / len(self.df)
        columns_to_drop = missing_percent[missing_percent > threshold].index
        self.df = self.df.drop(columns=columns_to_drop)
        return self
    
    def fill_numeric_missing(self, method='median'):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if method == 'median':
            fill_values = self.df[numeric_cols].median()
        elif method == 'mean':
            fill_values = self.df[numeric_cols].mean()
        elif method == 'mode':
            fill_values = self.df[numeric_cols].mode().iloc[0]
        else:
            fill_values = 0
        
        self.df[numeric_cols] = self.df[numeric_cols].fillna(fill_values)
        return self
    
    def remove_outliers_zscore(self, threshold=3):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            z_scores = np.abs(stats.zscore(self.df[col].dropna()))
            outliers = z_scores > threshold
            self.df.loc[outliers, col] = np.nan
        
        return self
    
    def normalize_data(self, method='minmax'):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if method == 'minmax':
            for col in numeric_cols:
                col_min = self.df[col].min()
                col_max = self.df[col].max()
                if col_max != col_min:
                    self.df[col] = (self.df[col] - col_min) / (col_max - col_min)
        
        elif method == 'standard':
            for col in numeric_cols:
                col_mean = self.df[col].mean()
                col_std = self.df[col].std()
                if col_std != 0:
                    self.df[col] = (self.df[col] - col_mean) / col_std
        
        return self
    
    def get_cleaned_data(self):
        return self.df
    
    def get_removed_columns(self):
        current_columns = self.df.columns.tolist()
        removed = [col for col in self.original_columns if col not in current_columns]
        return removed

def clean_dataset(df, missing_threshold=0.3, outlier_threshold=3, normalize=True):
    cleaner = DataCleaner(df)
    
    cleaner.remove_missing(missing_threshold)
    cleaner.fill_numeric_missing('median')
    cleaner.remove_outliers_zscore(outlier_threshold)
    cleaner.fill_numeric_missing('median')
    
    if normalize:
        cleaner.normalize_data('standard')
    
    return cleaner.get_cleaned_data()