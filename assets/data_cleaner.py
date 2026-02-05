import numpy as np
import pandas as pd

class DataCleaner:
    def __init__(self, data):
        self.data = data
        self.cleaned_data = None
        
    def remove_outliers_iqr(self, column):
        Q1 = self.data[column].quantile(0.25)
        Q3 = self.data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return self.data[(self.data[column] >= lower_bound) & (self.data[column] <= upper_bound)]
    
    def normalize_column(self, column, method='minmax'):
        if method == 'minmax':
            min_val = self.data[column].min()
            max_val = self.data[column].max()
            self.data[column] = (self.data[column] - min_val) / (max_val - min_val)
        elif method == 'zscore':
            mean_val = self.data[column].mean()
            std_val = self.data[column].std()
            self.data[column] = (self.data[column] - mean_val) / std_val
        return self.data
    
    def handle_missing_values(self, strategy='mean'):
        for column in self.data.columns:
            if self.data[column].isnull().any():
                if strategy == 'mean':
                    fill_value = self.data[column].mean()
                elif strategy == 'median':
                    fill_value = self.data[column].median()
                elif strategy == 'mode':
                    fill_value = self.data[column].mode()[0]
                else:
                    fill_value = 0
                self.data[column].fillna(fill_value, inplace=True)
        return self.data
    
    def clean(self, outlier_columns=None, normalize_columns=None, missing_strategy='mean'):
        self.handle_missing_values(strategy=missing_strategy)
        
        if outlier_columns:
            for column in outlier_columns:
                self.data = self.remove_outliers_iqr(column)
        
        if normalize_columns:
            for column in normalize_columns:
                self.normalize_column(column)
        
        self.cleaned_data = self.data.copy()
        return self.cleaned_data
    
    def get_summary(self):
        summary = {
            'original_shape': self.data.shape,
            'cleaned_shape': self.cleaned_data.shape if self.cleaned_data is not None else None,
            'missing_values': self.data.isnull().sum().to_dict(),
            'data_types': self.data.dtypes.to_dict()
        }
        return summary

def example_usage():
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 100),
        'feature2': np.random.exponential(50, 100),
        'feature3': np.random.randint(1, 100, 100)
    })
    
    cleaner = DataCleaner(sample_data)
    cleaned = cleaner.clean(
        outlier_columns=['feature1', 'feature2'],
        normalize_columns=['feature1', 'feature2'],
        missing_strategy='mean'
    )
    
    print("Data cleaning completed")
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {cleaned.shape}")
    return cleaned

if __name__ == "__main__":
    example_usage()