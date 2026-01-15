
import pandas as pd
import numpy as np
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
        
        if method == 'median':
            fill_values = self.df[numeric_cols].median()
        elif method == 'mean':
            fill_values = self.df[numeric_cols].mean()
        else:
            fill_values = 0
            
        self.df[numeric_cols] = self.df[numeric_cols].fillna(fill_values)
        return self
    
    def detect_outliers_zscore(self, threshold=3):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        z_scores = np.abs(stats.zscore(self.df[numeric_cols]))
        outlier_mask = (z_scores > threshold).any(axis=1)
        return outlier_mask
    
    def remove_outliers(self, threshold=3):
        outlier_mask = self.detect_outliers_zscore(threshold)
        self.df = self.df[~outlier_mask]
        return self
    
    def normalize_data(self, method='minmax'):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if method == 'minmax':
            for col in numeric_cols:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
        
        elif method == 'standard':
            for col in numeric_cols:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    self.df[col] = (self.df[col] - mean_val) / std_val
        
        return self
    
    def get_cleaned_data(self):
        return self.df
    
    def get_summary(self):
        cleaned_shape = self.df.shape
        rows_removed = self.original_shape[0] - cleaned_shape[0]
        cols_removed = self.original_shape[1] - cleaned_shape[1]
        
        summary = {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_rows': cleaned_shape[0],
            'cleaned_columns': cleaned_shape[1],
            'rows_removed': rows_removed,
            'columns_removed': cols_removed,
            'missing_values': self.df.isnull().sum().sum()
        }
        return summary

def example_usage():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.uniform(0, 1, 100),
        'feature_c': np.random.exponential(2, 100)
    }
    
    df = pd.DataFrame(data)
    df.iloc[5:10, 0] = np.nan
    df.iloc[15:20, 1] = np.nan
    df.iloc[0, 0] = 500
    
    cleaner = DataCleaner(df)
    cleaned_df = (cleaner
                  .remove_missing(threshold=0.2)
                  .fill_numeric_missing('median')
                  .remove_outliers(threshold=3)
                  .normalize_data('minmax')
                  .get_cleaned_data())
    
    summary = cleaner.get_summary()
    return cleaned_df, summary

if __name__ == "__main__":
    result_df, stats_summary = example_usage()
    print(f"Cleaned data shape: {result_df.shape}")
    print(f"Processing summary: {stats_summary}")