
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, data):
        self.data = data
        self.original_shape = data.shape
        
    def remove_outliers_iqr(self, columns=None):
        if columns is None:
            columns = self.data.columns if hasattr(self.data, 'columns') else range(self.data.shape[1])
        
        cleaned_data = self.data.copy()
        for col in columns:
            Q1 = cleaned_data[col].quantile(0.25)
            Q3 = cleaned_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            cleaned_data = cleaned_data[(cleaned_data[col] >= lower_bound) & (cleaned_data[col] <= upper_bound)]
        
        self.removed_count = self.original_shape[0] - cleaned_data.shape[0]
        self.data = cleaned_data
        return self
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.data.columns if hasattr(self.data, 'columns') else range(self.data.shape[1])
        
        normalized_data = self.data.copy()
        for col in columns:
            min_val = normalized_data[col].min()
            max_val = normalized_data[col].max()
            if max_val != min_val:
                normalized_data[col] = (normalized_data[col] - min_val) / (max_val - min_val)
        
        self.data = normalized_data
        return self
    
    def standardize_zscore(self, columns=None):
        if columns is None:
            columns = self.data.columns if hasattr(self.data, 'columns') else range(self.data.shape[1])
        
        standardized_data = self.data.copy()
        for col in columns:
            mean_val = standardized_data[col].mean()
            std_val = standardized_data[col].std()
            if std_val > 0:
                standardized_data[col] = (standardized_data[col] - mean_val) / std_val
        
        self.data = standardized_data
        return self
    
    def get_summary(self):
        summary = {
            'original_samples': self.original_shape[0],
            'current_samples': self.data.shape[0],
            'features': self.data.shape[1],
            'removed_outliers': getattr(self, 'removed_count', 0)
        }
        return summary

def create_sample_data(n_samples=1000, n_features=5):
    np.random.seed(42)
    data = np.random.randn(n_samples, n_features)
    data = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(n_features)])
    return data

if __name__ == "__main__":
    sample_data = create_sample_data()
    cleaner = DataCleaner(sample_data)
    
    print("Original data shape:", cleaner.original_shape)
    
    cleaner.remove_outliers_iqr()
    cleaner.normalize_minmax()
    
    summary = cleaner.get_summary()
    print(f"Cleaned data shape: {summary['current_samples']} samples, {summary['features']} features")
    print(f"Removed outliers: {summary['removed_outliers']}")
    
    print("\nFirst 5 rows of cleaned data:")
    print(cleaner.data.head())