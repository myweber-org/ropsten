
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, columns=None, factor=1.5):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_clean = self.df.copy()
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        self.df = df_clean
        removed_count = self.original_shape[0] - self.df.shape[0]
        print(f"Removed {removed_count} outliers")
        return self
        
    def normalize_data(self, columns=None, method='zscore'):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_normalized = self.df.copy()
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                if method == 'zscore':
                    df_normalized[col] = stats.zscore(self.df[col])
                elif method == 'minmax':
                    df_normalized[col] = (self.df[col] - self.df[col].min()) / (self.df[col].max() - self.df[col].min())
                elif method == 'robust':
                    median = self.df[col].median()
                    iqr = self.df[col].quantile(0.75) - self.df[col].quantile(0.25)
                    df_normalized[col] = (self.df[col] - median) / iqr
        
        self.df = df_normalized
        print(f"Normalized {len(columns)} columns using {method} method")
        return self
        
    def fill_missing(self, columns=None, strategy='mean'):
        if columns is None:
            columns = self.df.columns
        
        df_filled = self.df.copy()
        for col in columns:
            if col in self.df.columns and self.df[col].isnull().any():
                if strategy == 'mean' and pd.api.types.is_numeric_dtype(self.df[col]):
                    df_filled[col] = self.df[col].fillna(self.df[col].mean())
                elif strategy == 'median' and pd.api.types.is_numeric_dtype(self.df[col]):
                    df_filled[col] = self.df[col].fillna(self.df[col].median())
                elif strategy == 'mode':
                    df_filled[col] = self.df[col].fillna(self.df[col].mode()[0])
                elif strategy == 'ffill':
                    df_filled[col] = self.df[col].fillna(method='ffill')
                elif strategy == 'bfill':
                    df_filled[col] = self.df[col].fillna(method='bfill')
        
        self.df = df_filled
        print(f"Filled missing values using {strategy} strategy")
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

def clean_dataset(df, outlier_removal=True, normalization=True, fill_missing=True):
    cleaner = DataCleaner(df)
    
    if outlier_removal:
        cleaner.remove_outliers_iqr()
    
    if fill_missing:
        cleaner.fill_missing(strategy='median')
    
    if normalization:
        cleaner.normalize_data(method='zscore')
    
    return cleaner.get_cleaned_data(), cleaner.get_summary()import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            elif fill_missing == 'median':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate a DataFrame for required columns and data types.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    
    Returns:
    tuple: (bool, str) indicating validation success and message.
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    return True, "Dataset validation passed"

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 3, None],
        'B': [4, None, 6, 7, 8],
        'C': ['x', 'y', 'y', 'z', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid, message = validate_dataset(cleaned, required_columns=['A', 'B'])
    print(f"\nValidation: {message}")
import pandas as pd
import numpy as np
from typing import Optional, Dict, List

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def handle_missing_values(self, strategy: str = 'mean', columns: Optional[List[str]] = None) -> 'DataCleaner':
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col not in self.df.columns:
                continue
                
            if strategy == 'mean':
                self.df[col].fillna(self.df[col].mean(), inplace=True)
            elif strategy == 'median':
                self.df[col].fillna(self.df[col].median(), inplace=True)
            elif strategy == 'mode':
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
            elif strategy == 'drop':
                self.df = self.df.dropna(subset=[col])
            elif isinstance(strategy, (int, float)):
                self.df[col].fillna(strategy, inplace=True)
        
        return self
    
    def convert_types(self, type_mapping: Dict[str, str]) -> 'DataCleaner':
        for col, dtype in type_mapping.items():
            if col in self.df.columns:
                try:
                    if dtype == 'datetime':
                        self.df[col] = pd.to_datetime(self.df[col])
                    elif dtype == 'category':
                        self.df[col] = self.df[col].astype('category')
                    else:
                        self.df[col] = self.df[col].astype(dtype)
                except Exception as e:
                    print(f"Warning: Could not convert column {col} to {dtype}: {e}")
        
        return self
    
    def remove_outliers(self, method: str = 'iqr', threshold: float = 1.5) -> 'DataCleaner':
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                mask = (self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)
                self.df = self.df[mask]
        
        return self
    
    def get_cleaned_data(self) -> pd.DataFrame:
        return self.df
    
    def get_summary(self) -> Dict:
        return {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_rows': self.df.shape[0],
            'cleaned_columns': self.df.shape[1],
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'missing_values': self.df.isnull().sum().sum()
        }

def clean_csv_file(input_path: str, output_path: str, **kwargs) -> Dict:
    try:
        df = pd.read_csv(input_path)
        cleaner = DataCleaner(df)
        
        if 'missing_strategy' in kwargs:
            cleaner.handle_missing_values(strategy=kwargs['missing_strategy'])
        
        if 'type_mapping' in kwargs:
            cleaner.convert_types(kwargs['type_mapping'])
        
        if 'remove_outliers' in kwargs and kwargs['remove_outliers']:
            cleaner.remove_outliers()
        
        cleaned_df = cleaner.get_cleaned_data()
        cleaned_df.to_csv(output_path, index=False)
        
        return {
            'success': True,
            'summary': cleaner.get_summary(),
            'output_file': output_path
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }