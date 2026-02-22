import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    def detect_outliers_iqr(self, column, threshold=1.5):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
        return outliers.index.tolist()
    
    def remove_outliers(self, columns=None, threshold=1.5):
        if columns is None:
            columns = self.numeric_columns
        
        outlier_indices = set()
        for col in columns:
            if col in self.numeric_columns:
                indices = self.detect_outliers_iqr(col, threshold)
                outlier_indices.update(indices)
        
        self.df = self.df.drop(index=list(outlier_indices))
        return self
    
    def impute_missing_mean(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        for col in columns:
            if col in self.numeric_columns:
                mean_val = self.df[col].mean()
                self.df[col] = self.df[col].fillna(mean_val)
        
        return self
    
    def impute_missing_median(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        for col in columns:
            if col in self.numeric_columns:
                median_val = self.df[col].median()
                self.df[col] = self.df[col].fillna(median_val)
        
        return self
    
    def standardize_features(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        for col in columns:
            if col in self.numeric_columns:
                mean = self.df[col].mean()
                std = self.df[col].std()
                if std > 0:
                    self.df[col] = (self.df[col] - mean) / std
        
        return self
    
    def get_cleaned_data(self):
        return self.df.copy()
    
    def summary(self):
        missing_values = self.df.isnull().sum()
        numeric_stats = self.df[self.numeric_columns].describe()
        return {
            'missing_values': missing_values,
            'numeric_statistics': numeric_stats,
            'shape': self.df.shape
        }

def example_usage():
    np.random.seed(42)
    data = {
        'feature1': np.random.normal(100, 15, 100),
        'feature2': np.random.exponential(50, 100),
        'feature3': np.random.uniform(0, 1, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(data)
    df.loc[10:15, 'feature1'] = np.nan
    df.loc[5, 'feature2'] = 1000
    
    cleaner = DataCleaner(df)
    cleaner.remove_outliers(['feature2']).impute_missing_mean().standardize_features(['feature1', 'feature2'])
    
    cleaned_df = cleaner.get_cleaned_data()
    summary = cleaner.summary()
    
    print(f"Original shape: {df.shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    print(f"Missing values after cleaning: {cleaned_df.isnull().sum().sum()}")
    
    return cleaned_df

if __name__ == "__main__":
    result = example_usage()
    print("Data cleaning completed successfully.")