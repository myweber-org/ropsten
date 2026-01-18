
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_columns = df.columns.tolist()
    
    def remove_outliers_iqr(self, column, multiplier=1.5):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        mask = (self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)
        return self.df[mask].reset_index(drop=True)
    
    def remove_outliers_zscore(self, column, threshold=3):
        z_scores = np.abs(stats.zscore(self.df[column]))
        mask = z_scores < threshold
        return self.df[mask].reset_index(drop=True)
    
    def normalize_minmax(self, column):
        min_val = self.df[column].min()
        max_val = self.df[column].max()
        self.df[f'{column}_normalized'] = (self.df[column] - min_val) / (max_val - min_val)
        return self.df
    
    def standardize_zscore(self, column):
        mean_val = self.df[column].mean()
        std_val = self.df[column].std()
        self.df[f'{column}_standardized'] = (self.df[column] - mean_val) / std_val
        return self.df
    
    def handle_missing_mean(self, column):
        mean_val = self.df[column].mean()
        self.df[column] = self.df[column].fillna(mean_val)
        return self.df
    
    def handle_missing_median(self, column):
        median_val = self.df[column].median()
        self.df[column] = self.df[column].fillna(median_val)
        return self.df
    
    def get_summary(self):
        summary = {
            'original_shape': self.df.shape,
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.to_dict(),
            'numeric_columns': self.df.select_dtypes(include=[np.number]).columns.tolist()
        }
        return summary
    
    def save_cleaned_data(self, filename='cleaned_data.csv'):
        self.df.to_csv(filename, index=False)
        return f"Data saved to {filename}"