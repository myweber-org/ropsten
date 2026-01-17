
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
        self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]
        return self
    
    def remove_outliers_zscore(self, column, threshold=3):
        z_scores = np.abs(stats.zscore(self.df[column]))
        self.df = self.df[z_scores < threshold]
        return self
    
    def normalize_minmax(self, column):
        min_val = self.df[column].min()
        max_val = self.df[column].max()
        self.df[column] = (self.df[column] - min_val) / (max_val - min_val)
        return self
    
    def normalize_zscore(self, column):
        mean_val = self.df[column].mean()
        std_val = self.df[column].std()
        self.df[column] = (self.df[column] - mean_val) / std_val
        return self
    
    def fill_missing_mean(self, column):
        self.df[column].fillna(self.df[column].mean(), inplace=True)
        return self
    
    def fill_missing_median(self, column):
        self.df[column].fillna(self.df[column].median(), inplace=True)
        return self
    
    def drop_duplicates(self):
        self.df.drop_duplicates(inplace=True)
        return self
    
    def get_cleaned_data(self):
        return self.df
    
    def summary(self):
        print(f"Original shape: {self.df.shape}")
        print(f"Missing values per column:")
        print(self.df.isnull().sum())
        print(f"Data types:")
        print(self.df.dtypes)

def clean_dataset(df, config):
    cleaner = DataCleaner(df)
    
    if 'outlier_method' in config:
        for col in config.get('outlier_columns', []):
            if col in df.columns:
                if config['outlier_method'] == 'iqr':
                    cleaner.remove_outliers_iqr(col, config.get('iqr_multiplier', 1.5))
                elif config['outlier_method'] == 'zscore':
                    cleaner.remove_outliers_zscore(col, config.get('zscore_threshold', 3))
    
    if 'normalization' in config:
        for col in config.get('normalize_columns', []):
            if col in df.columns:
                if config['normalization'] == 'minmax':
                    cleaner.normalize_minmax(col)
                elif config['normalization'] == 'zscore':
                    cleaner.normalize_zscore(col)
    
    if 'missing_values' in config:
        for col in config.get('missing_columns', []):
            if col in df.columns:
                if config['missing_values'] == 'mean':
                    cleaner.fill_missing_mean(col)
                elif config['missing_values'] == 'median':
                    cleaner.fill_missing_median(col)
    
    if config.get('drop_duplicates', False):
        cleaner.drop_duplicates()
    
    return cleaner.get_cleaned_data()