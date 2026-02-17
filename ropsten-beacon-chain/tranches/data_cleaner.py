
import pandas as pd
import re

def clean_text_column(df, column_name):
    """
    Standardize text by converting to lowercase, removing extra whitespace,
    and stripping special characters except basic punctuation.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    df[column_name] = df[column_name].astype(str).str.lower()
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'\s+', ' ', x))
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'[^\w\s.,!?-]', '', x))
    df[column_name] = df[column_name].str.strip()
    return df

def remove_duplicate_rows(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame.
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def validate_email_column(df, column_name):
    """
    Validate email format in specified column and return boolean mask.
    """
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return df[column_name].astype(str).str.match(email_pattern)

def sample_data_cleaning_pipeline():
    """
    Example pipeline demonstrating the data cleaning functions.
    """
    data = {
        'name': ['John Doe', 'Jane Smith', 'john doe', 'Bob Johnson  ', 'Alice Brown'],
        'email': ['john@example.com', 'jane@test.org', 'invalid-email', 'bob@company.net', 'alice@domain.co'],
        'notes': ['Important client!', 'Needs follow-up.', '  Regular   customer  ', 'VIP; special handling', 'New prospect.']
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50)
    
    df = clean_text_column(df, 'name')
    df = clean_text_column(df, 'notes')
    df = remove_duplicate_rows(df, subset=['name'])
    
    df['valid_email'] = validate_email_column(df, 'email')
    
    print("\nCleaned DataFrame:")
    print(df)
    return df

if __name__ == "__main__":
    cleaned_df = sample_data_cleaning_pipeline()
import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_missing(self, threshold=0.3):
        missing_percent = self.df.isnull().sum() / len(self.df)
        columns_to_drop = missing_percent[missing_percent > threshold].index
        self.df = self.df.drop(columns=columns_to_drop)
        return self
    
    def fill_numeric_missing(self, method='median'):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if self.df[col].isnull().any():
                if method == 'median':
                    fill_value = self.df[col].median()
                elif method == 'mean':
                    fill_value = self.df[col].mean()
                elif method == 'mode':
                    fill_value = self.df[col].mode()[0]
                else:
                    fill_value = 0
                
                self.df[col] = self.df[col].fillna(fill_value)
        
        return self
    
    def detect_outliers_zscore(self, threshold=3):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        outliers_mask = pd.Series([False] * len(self.df))
        
        for col in numeric_cols:
            z_scores = np.abs(stats.zscore(self.df[col].dropna()))
            col_outliers = (z_scores > threshold)
            outliers_mask = outliers_mask | col_outliers.reindex(self.df.index, fill_value=False)
        
        return outliers_mask
    
    def remove_outliers(self, threshold=3):
        outliers_mask = self.detect_outliers_zscore(threshold)
        self.df = self.df[~outliers_mask]
        return self
    
    def normalize_numeric(self, method='minmax'):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if method == 'minmax':
            for col in numeric_cols:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
        
        elif method == 'standard':
            for col in numeric_cols:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    self.df[col] = (self.df[col] - mean_val) / std_val
        
        return self
    
    def get_cleaned_data(self):
        return self.df
    
    def get_summary(self):
        cleaned_shape = self.df.shape
        rows_removed = self.original_shape[0] - cleaned_shape[0]
        cols_removed = self.original_shape[1] - cleaned_shape[1]
        
        summary = {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_rows': cleaned_shape[0],
            'cleaned_columns': cleaned_shape[1],
            'rows_removed': rows_removed,
            'columns_removed': cols_removed,
            'remaining_missing': self.df.isnull().sum().sum()
        }
        
        return summary

def clean_dataset(df, missing_threshold=0.3, outlier_threshold=3, normalize=True):
    cleaner = DataCleaner(df)
    
    cleaner.remove_missing(missing_threshold)
    cleaner.fill_numeric_missing('median')
    cleaner.remove_outliers(outlier_threshold)
    
    if normalize:
        cleaner.normalize_numeric('minmax')
    
    return cleaner.get_cleaned_data(), cleaner.get_summary()