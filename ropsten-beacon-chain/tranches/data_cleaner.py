
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, column, multiplier=1.5):
        q1 = self.df[column].quantile(0.25)
        q3 = self.df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        mask = (self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)
        self.df = self.df[mask]
        return self.df
    
    def remove_outliers_zscore(self, column, threshold=3):
        z_scores = np.abs(stats.zscore(self.df[column]))
        mask = z_scores < threshold
        self.df = self.df[mask]
        return self.df
    
    def normalize_minmax(self, column):
        min_val = self.df[column].min()
        max_val = self.df[column].max()
        self.df[column + '_normalized'] = (self.df[column] - min_val) / (max_val - min_val)
        return self.df
    
    def normalize_zscore(self, column):
        mean_val = self.df[column].mean()
        std_val = self.df[column].std()
        self.df[column + '_standardized'] = (self.df[column] - mean_val) / std_val
        return self.df
    
    def fill_missing_mean(self, column):
        mean_val = self.df[column].mean()
        self.df[column] = self.df[column].fillna(mean_val)
        return self.df
    
    def fill_missing_median(self, column):
        median_val = self.df[column].median()
        self.df[column] = self.df[column].fillna(median_val)
        return self.df
    
    def get_cleaned_data(self):
        return self.df
    
    def get_removed_count(self):
        return self.original_shape[0] - self.df.shape[0]
    
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'cleaned_rows': self.df.shape[0],
            'removed_rows': self.get_removed_count(),
            'columns': list(self.df.columns)
        }
        return summary

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.uniform(0, 1, 1000)
    }
    df = pd.DataFrame(data)
    df.loc[np.random.choice(df.index, 50), 'feature_a'] = np.nan
    df.loc[np.random.choice(df.index, 30), 'feature_b'] = np.nan
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    cleaner = DataCleaner(sample_df)
    
    print("Original data shape:", cleaner.original_shape)
    print("Missing values before cleaning:")
    print(sample_df.isnull().sum())
    
    cleaner.fill_missing_mean('feature_a')
    cleaner.fill_missing_median('feature_b')
    cleaner.remove_outliers_iqr('feature_a')
    cleaner.normalize_minmax('feature_c')
    
    cleaned_df = cleaner.get_cleaned_data()
    print("\nCleaned data shape:", cleaned_df.shape)
    print("Missing values after cleaning:")
    print(cleaned_df.isnull().sum())
    
    summary = cleaner.get_summary()
    print("\nCleaning summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")