
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, column, multiplier=1.5):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        mask = (self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)
        self.df = self.df[mask]
        return self
        
    def remove_outliers_zscore(self, column, threshold=3):
        z_scores = np.abs(stats.zscore(self.df[column]))
        mask = z_scores < threshold
        self.df = self.df[mask]
        return self
        
    def normalize_minmax(self, columns):
        for col in columns:
            if col in self.df.columns:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val != min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
        return self
        
    def normalize_zscore(self, columns):
        for col in columns:
            if col in self.df.columns:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    self.df[col] = (self.df[col] - mean_val) / std_val
        return self
        
    def fill_missing_mean(self, columns):
        for col in columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(self.df[col].mean())
        return self
        
    def fill_missing_median(self, columns):
        for col in columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(self.df[col].median())
        return self
        
    def get_cleaned_data(self):
        return self.df
        
    def get_removed_count(self):
        return self.original_shape[0] - self.df.shape[0]
        
    def summary(self):
        print(f"Original rows: {self.original_shape[0]}")
        print(f"Cleaned rows: {self.df.shape[0]}")
        print(f"Rows removed: {self.get_removed_count()}")
        print(f"Columns: {self.df.shape[1]}")
        print("\nMissing values per column:")
        print(self.df.isnull().sum())
        return self

def example_usage():
    np.random.seed(42)
    data = {
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'feature3': np.random.uniform(0, 1, 1000)
    }
    df = pd.DataFrame(data)
    df.loc[np.random.choice(df.index, 50), 'feature1'] = np.nan
    df.loc[np.random.choice(df.index, 30), 'feature2'] = np.nan
    
    cleaner = DataCleaner(df)
    cleaned_df = (cleaner
                 .remove_outliers_iqr('feature1')
                 .remove_outliers_zscore('feature2')
                 .fill_missing_mean(['feature1', 'feature2'])
                 .normalize_minmax(['feature1', 'feature2', 'feature3'])
                 .get_cleaned_data())
    
    cleaner.summary()
    return cleaned_df

if __name__ == "__main__":
    result = example_usage()
    print("\nFirst 5 rows of cleaned data:")
    print(result.head())