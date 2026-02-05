
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    def detect_outliers_iqr(self, column, threshold=1.5):
        if column not in self.numeric_columns:
            raise ValueError(f"Column {column} is not numeric")
        
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
        return outliers
    
    def remove_outliers_zscore(self, column, threshold=3):
        if column not in self.numeric_columns:
            raise ValueError(f"Column {column} is not numeric")
        
        z_scores = np.abs(stats.zscore(self.df[column].dropna()))
        mask = z_scores < threshold
        self.df = self.df[mask]
        return self.df
    
    def normalize_column(self, column, method='minmax'):
        if column not in self.numeric_columns:
            raise ValueError(f"Column {column} is not numeric")
        
        if method == 'minmax':
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            self.df[f'{column}_normalized'] = (self.df[column] - min_val) / (max_val - min_val)
        
        elif method == 'zscore':
            mean_val = self.df[column].mean()
            std_val = self.df[column].std()
            self.df[f'{column}_normalized'] = (self.df[column] - mean_val) / std_val
        
        else:
            raise ValueError("Method must be 'minmax' or 'zscore'")
        
        return self.df
    
    def fill_missing_with_strategy(self, column, strategy='mean'):
        if column not in self.numeric_columns:
            raise ValueError(f"Column {column} is not numeric")
        
        if strategy == 'mean':
            fill_value = self.df[column].mean()
        elif strategy == 'median':
            fill_value = self.df[column].median()
        elif strategy == 'mode':
            fill_value = self.df[column].mode()[0]
        else:
            raise ValueError("Strategy must be 'mean', 'median', or 'mode'")
        
        self.df[column] = self.df[column].fillna(fill_value)
        return self.df
    
    def get_cleaned_data(self):
        return self.df.copy()

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 50),
        'feature_b': np.random.exponential(2, 50),
        'feature_c': np.random.uniform(0, 1, 50)
    }
    df = pd.DataFrame(data)
    df.loc[::10, 'feature_a'] = np.nan
    df.loc[5, 'feature_b'] = 1000
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    cleaner = DataCleaner(sample_df)
    
    print("Original data shape:", sample_df.shape)
    print("\nOutliers in feature_b:")
    print(cleaner.detect_outliers_iqr('feature_b'))
    
    cleaner.remove_outliers_zscore('feature_b')
    cleaner.fill_missing_with_strategy('feature_a', 'mean')
    cleaner.normalize_column('feature_c', 'minmax')
    
    cleaned_df = cleaner.get_cleaned_data()
    print("\nCleaned data shape:", cleaned_df.shape)
    print("\nCleaned data summary:")
    print(cleaned_df.describe())