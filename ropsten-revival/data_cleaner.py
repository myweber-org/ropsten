
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
import pandas as pd

def clean_dataset(df, column_name):
    """
    Remove duplicate rows and sort the DataFrame by a specified column.
    
    Args:
        df (pd.DataFrame): The input DataFrame to clean.
        column_name (str): The column name to sort by.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame with duplicates removed and sorted.
    """
    if df.empty:
        return df
    
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    df_cleaned = df.drop_duplicates().reset_index(drop=True)
    df_cleaned = df_cleaned.sort_values(by=column_name).reset_index(drop=True)
    
    return df_cleaned

def filter_by_threshold(df, column_name, threshold):
    """
    Filter rows where the column value is greater than a threshold.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The column to apply the filter on.
        threshold (float): The threshold value.
    
    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    if df.empty:
        return df
    
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    filtered_df = df[df[column_name] > threshold].reset_index(drop=True)
    return filtered_df

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 4, 5],
        'value': [10.5, 20.3, 20.3, 15.7, 8.9, 8.9, 30.1],
        'category': ['A', 'B', 'B', 'A', 'C', 'C', 'B']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    cleaned_df = clean_dataset(df, 'value')
    print("Cleaned DataFrame (duplicates removed, sorted by 'value'):")
    print(cleaned_df)
    print()
    
    filtered_df = filter_by_threshold(cleaned_df, 'value', 15.0)
    print("Filtered DataFrame (value > 15.0):")
    print(filtered_df)