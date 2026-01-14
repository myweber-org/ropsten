import pandas as pd
import numpy as np
from typing import Optional, Dict, List

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_duplicates(self, subset: Optional[List[str]] = None) -> 'DataCleaner':
        self.df = self.df.drop_duplicates(subset=subset)
        return self
        
    def handle_missing_values(self, 
                             strategy: str = 'mean',
                             columns: Optional[List[str]] = None) -> 'DataCleaner':
        if columns is None:
            columns = self.df.columns
            
        for col in columns:
            if self.df[col].isnull().any():
                if strategy == 'mean' and pd.api.types.is_numeric_dtype(self.df[col]):
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                elif strategy == 'median' and pd.api.types.is_numeric_dtype(self.df[col]):
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                elif strategy == 'mode':
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                elif strategy == 'drop':
                    self.df = self.df.dropna(subset=[col])
                else:
                    self.df[col].fillna(0, inplace=True)
        return self
        
    def convert_types(self, 
                     type_map: Dict[str, str]) -> 'DataCleaner':
        for col, dtype in type_map.items():
            if col in self.df.columns:
                try:
                    if dtype == 'datetime':
                        self.df[col] = pd.to_datetime(self.df[col])
                    elif dtype == 'numeric':
                        self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                    elif dtype == 'category':
                        self.df[col] = self.df[col].astype('category')
                except Exception as e:
                    print(f"Error converting column {col} to {dtype}: {e}")
        return self
        
    def remove_outliers(self, 
                       columns: List[str],
                       method: str = 'iqr',
                       threshold: float = 1.5) -> 'DataCleaner':
        for col in columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                if method == 'iqr':
                    Q1 = self.df[col].quantile(0.25)
                    Q3 = self.df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    self.df = self.df[(self.df[col] >= lower_bound) & 
                                     (self.df[col] <= upper_bound)]
                elif method == 'zscore':
                    z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                    self.df = self.df[z_scores < threshold]
        return self
        
    def normalize_columns(self, 
                         columns: List[str],
                         method: str = 'minmax') -> 'DataCleaner':
        for col in columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                if method == 'minmax':
                    min_val = self.df[col].min()
                    max_val = self.df[col].max()
                    if max_val > min_val:
                        self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
                elif method == 'standard':
                    mean_val = self.df[col].mean()
                    std_val = self.df[col].std()
                    if std_val > 0:
                        self.df[col] = (self.df[col] - mean_val) / std_val
        return self
        
    def get_cleaned_data(self) -> pd.DataFrame:
        return self.df
        
    def get_cleaning_report(self) -> Dict:
        report = {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_rows': self.df.shape[0],
            'cleaned_columns': self.df.shape[1],
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'missing_values': int(self.df.isnull().sum().sum()),
            'duplicates_removed': self.original_shape[0] - self.df.drop_duplicates().shape[0]
        }
        return report

def clean_csv_file(input_path: str,
                  output_path: str,
                  missing_strategy: str = 'mean',
                  type_conversions: Optional[Dict] = None) -> Dict:
    try:
        df = pd.read_csv(input_path)
        cleaner = DataCleaner(df)
        
        cleaner.remove_duplicates()
        cleaner.handle_missing_values(strategy=missing_strategy)
        
        if type_conversions:
            cleaner.convert_types(type_conversions)
            
        cleaned_df = cleaner.get_cleaned_data()
        report = cleaner.get_cleaning_report()
        
        cleaned_df.to_csv(output_path, index=False)
        report['output_file'] = output_path
        
        return report
        
    except Exception as e:
        return {'error': str(e), 'success': False}
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
        df_clean = df_clean.drop_duplicates()
        removed = initial_rows - len(df_clean)
        print(f"Removed {removed} duplicate rows.")
    
    if columns_to_clean is None:
        columns_to_clean = df_clean.select_dtypes(include=['object']).columns.tolist()
    
    for col in columns_to_clean:
        if col in df_clean.columns and df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].astype(str)
            df_clean[col] = df_clean[col].apply(lambda x: re.sub(r'\s+', ' ', x.strip()))
            
            if case_normalization == 'lower':
                df_clean[col] = df_clean[col].str.lower()
            elif case_normalization == 'upper':
                df_clean[col] = df_clean[col].str.upper()
    
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
    print(f"Valid emails: {valid_count}/{total_count} ({valid_count/total_count*100:.1f}%)")
    
    return df

if __name__ == "__main__":
    sample_data = {
        'name': ['John Doe', 'Jane Smith', 'John Doe', 'ALICE WONDER', 'bob   builder  '],
        'email': ['john@example.com', 'jane@test.org', 'invalid-email', 'alice@company.co.uk', 'bob@home.net'],
        'age': [25, 30, 25, 28, 35]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaned_df = clean_dataframe(df, case_normalization='lower')
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\n")
    
    validated_df = validate_email_column(cleaned_df, 'email')
    print("DataFrame with email validation:")
    print(validated_df[['name', 'email', 'email_valid']])