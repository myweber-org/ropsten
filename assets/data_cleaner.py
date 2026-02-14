
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, columns=None, threshold=1.5):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_clean = self.df.copy()
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        self.df = df_clean
        removed_count = self.original_shape[0] - self.df.shape[0]
        print(f"Removed {removed_count} outliers")
        return self
        
    def normalize_data(self, columns=None, method='minmax'):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_normalized = self.df.copy()
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                if method == 'minmax':
                    min_val = self.df[col].min()
                    max_val = self.df[col].max()
                    if max_val != min_val:
                        df_normalized[col] = (self.df[col] - min_val) / (max_val - min_val)
                elif method == 'zscore':
                    mean_val = self.df[col].mean()
                    std_val = self.df[col].std()
                    if std_val > 0:
                        df_normalized[col] = (self.df[col] - mean_val) / std_val
        
        self.df = df_normalized
        print(f"Normalized {len(columns)} columns using {method} method")
        return self
        
    def handle_missing_values(self, strategy='mean', fill_value=None):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if self.df[col].isnull().any():
                if strategy == 'mean':
                    fill_val = self.df[col].mean()
                elif strategy == 'median':
                    fill_val = self.df[col].median()
                elif strategy == 'mode':
                    fill_val = self.df[col].mode()[0]
                elif strategy == 'constant' and fill_value is not None:
                    fill_val = fill_value
                else:
                    continue
                    
                self.df[col] = self.df[col].fillna(fill_val)
                print(f"Filled missing values in {col} with {strategy}: {fill_val:.4f}")
        
        return self
        
    def get_cleaned_data(self):
        return self.df
        
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'cleaned_rows': self.df.shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_columns': self.df.shape[1],
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'missing_values': self.df.isnull().sum().sum()
        }
        return summary

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'feature_c': np.random.uniform(0, 1, 100)
    }
    
    df = pd.DataFrame(data)
    
    df.loc[10:15, 'feature_a'] = np.nan
    df.loc[5, 'feature_b'] = 1000
    df.loc[95, 'feature_b'] = -500
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    print("Original data shape:", sample_df.shape)
    
    cleaner = DataCleaner(sample_df)
    cleaned_df = (cleaner
                 .handle_missing_values(strategy='median')
                 .remove_outliers_iqr(threshold=1.5)
                 .normalize_data(method='minmax')
                 .get_cleaned_data())
    
    print("\nCleaned data shape:", cleaned_df.shape)
    print("\nSummary:")
    for key, value in cleaner.get_summary().items():
        print(f"{key}: {value}")
import pandas as pd
import re

def clean_dataframe(df, columns_to_clean=None, remove_duplicates=True, case_normalization='lower'):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing string columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    columns_to_clean (list, optional): List of column names to apply string normalization.
                                       If None, all object dtype columns are cleaned.
    remove_duplicates (bool): If True, remove duplicate rows.
    case_normalization (str): One of 'lower', 'upper', or None for case normalization.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    if remove_duplicates:
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)
        removed = initial_rows - len(df_clean)
        print(f"Removed {removed} duplicate rows.")
    
    if columns_to_clean is None:
        columns_to_clean = df_clean.select_dtypes(include=['object']).columns.tolist()
    
    for col in columns_to_clean:
        if col in df_clean.columns and df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].astype(str)
            
            if case_normalization == 'lower':
                df_clean[col] = df_clean[col].str.lower()
            elif case_normalization == 'upper':
                df_clean[col] = df_clean[col].str.upper()
            
            df_clean[col] = df_clean[col].apply(lambda x: re.sub(r'\s+', ' ', x.strip()))
    
    return df_clean

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    email_column (str): Name of the column containing email addresses.
    
    Returns:
    pd.DataFrame: DataFrame with additional 'email_valid' boolean column.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame.")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    df['email_valid'] = df[email_column].str.match(email_pattern, na=False)
    
    valid_count = df['email_valid'].sum()
    total_count = len(df)
    print(f"Found {valid_count} valid emails out of {total_count} rows.")
    
    return df

if __name__ == "__main__":
    sample_data = {
        'name': ['John Doe', 'Jane Smith', 'John Doe', 'ALICE WONDER', '  bob   '],
        'email': ['john@example.com', 'jane@test.org', 'invalid-email', 'alice@company.co.uk', 'bob@domain.com'],
        'age': [25, 30, 25, 28, 35]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaned_df = clean_dataframe(df, columns_to_clean=['name', 'email'], remove_duplicates=True)
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\n")
    
    validated_df = validate_email_column(cleaned_df, 'email')
    print("DataFrame with email validation:")
    print(validated_df[['name', 'email', 'email_valid']])