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
    
    def normalize_minmax(self, column):
        min_val = self.data[column].min()
        max_val = self.data[column].max()
        self.data[column] = (self.data[column] - min_val) / (max_val - min_val)
        return self.data
    
    def fill_missing_mean(self, column):
        mean_val = self.data[column].mean()
        self.data[column].fillna(mean_val, inplace=True)
        return self.data
    
    def clean_all(self, outlier_columns=None, normalize_columns=None, fill_columns=None):
        if outlier_columns:
            for col in outlier_columns:
                self.data = self.remove_outliers_iqr(col)
        
        if normalize_columns:
            for col in normalize_columns:
                self.data = self.normalize_minmax(col)
        
        if fill_columns:
            for col in fill_columns:
                self.data = self.fill_missing_mean(col)
        
        self.cleaned_data = self.data.copy()
        return self.cleaned_data
    
    def save_cleaned_data(self, filename):
        if self.cleaned_data is not None:
            self.cleaned_data.to_csv(filename, index=False)
            return True
        return False

def example_usage():
    sample_data = pd.DataFrame({
        'age': [25, 30, 35, 200, 40, 45, None, 50],
        'salary': [50000, 60000, 70000, 800000, 90000, 100000, 110000, 120000],
        'score': [85, 90, 78, 92, 88, None, 95, 87]
    })
    
    cleaner = DataCleaner(sample_data)
    cleaned = cleaner.clean_all(
        outlier_columns=['age', 'salary'],
        normalize_columns=['score'],
        fill_columns=['age', 'score']
    )
    
    print("Original data shape:", sample_data.shape)
    print("Cleaned data shape:", cleaned.shape)
    print("\nCleaned data summary:")
    print(cleaned.describe())
    
    cleaner.save_cleaned_data('cleaned_data.csv')
    return cleaned

if __name__ == "__main__":
    example_usage()