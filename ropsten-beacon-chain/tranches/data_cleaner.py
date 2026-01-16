
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_missing(self, threshold=0.3):
        missing_percent = self.df.isnull().sum() / len(self.df)
        columns_to_drop = missing_percent[missing_percent > threshold].index
        self.df = self.df.drop(columns=columns_to_drop)
        return self
        
    def fill_numeric_missing(self, method='median'):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if self.df[col].isnull().any():
                if method == 'median':
                    fill_value = self.df[col].median()
                elif method == 'mean':
                    fill_value = self.df[col].mean()
                elif method == 'mode':
                    fill_value = self.df[col].mode()[0]
                else:
                    fill_value = 0
                    
                self.df[col] = self.df[col].fillna(fill_value)
                
        return self
        
    def detect_outliers_zscore(self, threshold=3):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        outlier_mask = pd.Series([False] * len(self.df))
        
        for col in numeric_cols:
            z_scores = np.abs(stats.zscore(self.df[col].fillna(0)))
            outlier_mask = outlier_mask | (z_scores > threshold)
            
        return outlier_mask
        
    def remove_outliers_iqr(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            self.df = self.df[(self.df[col] >= lower_bound) & 
                             (self.df[col] <= upper_bound)]
            
        return self
        
    def normalize_data(self, method='minmax'):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if method == 'minmax':
            for col in numeric_cols:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val != min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
                    
        elif method == 'standard':
            for col in numeric_cols:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val != 0:
                    self.df[col] = (self.df[col] - mean_val) / std_val
                    
        return self
        
    def get_cleaned_data(self):
        return self.df
        
    def get_summary(self):
        original_rows, original_cols = self.original_shape
        cleaned_rows, cleaned_cols = self.df.shape
        
        summary = {
            'original_rows': original_rows,
            'original_columns': original_cols,
            'cleaned_rows': cleaned_rows,
            'cleaned_columns': cleaned_cols,
            'rows_removed': original_rows - cleaned_rows,
            'columns_removed': original_cols - cleaned_cols
        }
        
        return summary

def clean_dataset(df, remove_outliers=True, normalize=True):
    cleaner = DataCleaner(df)
    
    cleaner.remove_missing(threshold=0.3)
    cleaner.fill_numeric_missing(method='median')
    
    if remove_outliers:
        cleaner.remove_outliers_iqr()
        
    if normalize:
        cleaner.normalize_data(method='minmax')
        
    return cleaner.get_cleaned_data(), cleaner.get_summary()