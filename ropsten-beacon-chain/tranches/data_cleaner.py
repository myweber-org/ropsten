
def remove_duplicates(input_list):
    """
    Remove duplicate elements from a list while preserving order.
    
    Args:
        input_list: A list containing elements that may have duplicates.
    
    Returns:
        A new list with duplicates removed, preserving the original order.
    """
    seen = set()
    result = []
    
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    
    return result

def clean_data_with_threshold(data_list, threshold=None):
    """
    Clean data by removing duplicates, optionally with a frequency threshold.
    
    Args:
        data_list: List of data elements to clean.
        threshold: Optional integer threshold. If provided, only elements
                  appearing more than threshold times are considered duplicates.
    
    Returns:
        Cleaned list with duplicates removed.
    """
    if threshold is None:
        return remove_duplicates(data_list)
    
    from collections import Counter
    counter = Counter(data_list)
    
    result = []
    seen = set()
    
    for item in data_list:
        if counter[item] > threshold:
            if item not in seen:
                seen.add(item)
                result.append(item)
        else:
            result.append(item)
    
    return result

def validate_input(data):
    """
    Validate that input is a list or convertible to list.
    
    Args:
        data: Input data to validate.
    
    Returns:
        Validated list.
    
    Raises:
        TypeError: If input cannot be converted to list.
    """
    if isinstance(data, list):
        return data
    elif hasattr(data, '__iter__'):
        return list(data)
    else:
        raise TypeError("Input must be iterable")

if __name__ == "__main__":
    # Example usage
    sample_data = [1, 2, 2, 3, 4, 4, 4, 5, 1, 6]
    
    print("Original data:", sample_data)
    print("After basic deduplication:", remove_duplicates(sample_data))
    print("With threshold 2:", clean_data_with_threshold(sample_data, threshold=2))
    
    # Test validation
    try:
        validate_input("not a list")
    except TypeError as e:
        print(f"Validation error: {e}")
def remove_duplicates(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, columns=None, multiplier=1.5):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        clean_df = self.df.copy()
        for col in columns:
            if col in clean_df.columns:
                Q1 = clean_df[col].quantile(0.25)
                Q3 = clean_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                clean_df = clean_df[(clean_df[col] >= lower_bound) & (clean_df[col] <= upper_bound)]
        
        removed_count = self.original_shape[0] - clean_df.shape[0]
        self.df = clean_df
        return removed_count
    
    def normalize_data(self, columns=None, method='minmax'):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        normalized_df = self.df.copy()
        for col in columns:
            if col in normalized_df.columns:
                if method == 'minmax':
                    min_val = normalized_df[col].min()
                    max_val = normalized_df[col].max()
                    if max_val != min_val:
                        normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
                elif method == 'zscore':
                    mean_val = normalized_df[col].mean()
                    std_val = normalized_df[col].std()
                    if std_val != 0:
                        normalized_df[col] = (normalized_df[col] - mean_val) / std_val
        
        self.df = normalized_df
        return self.df
    
    def handle_missing_values(self, strategy='mean', fill_value=None):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if strategy == 'mean':
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
        elif strategy == 'median':
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        elif strategy == 'mode':
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mode().iloc[0])
        elif strategy == 'custom' and fill_value is not None:
            self.df[numeric_cols] = self.df[numeric_cols].fillna(fill_value)
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        self.df[categorical_cols] = self.df[categorical_cols].fillna('Unknown')
        
        return self.df.isnull().sum().sum()
    
    def get_cleaned_data(self):
        return self.df
    
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'current_rows': self.df.shape[0],
            'columns': self.df.shape[1],
            'missing_values': self.df.isnull().sum().sum(),
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(include=['object']).columns)
        }
        return summary

def load_and_clean_data(filepath, cleaning_steps=None):
    df = pd.read_csv(filepath)
    cleaner = DataCleaner(df)
    
    if cleaning_steps:
        for step in cleaning_steps:
            if step == 'handle_missing':
                cleaner.handle_missing_values()
            elif step == 'remove_outliers':
                cleaner.remove_outliers_iqr()
            elif step == 'normalize':
                cleaner.normalize_data()
    
    return cleaner.get_cleaned_data()