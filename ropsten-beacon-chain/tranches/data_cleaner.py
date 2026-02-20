
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
            if col in self.df.columns:
                if strategy == 'mean':
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                elif strategy == 'median':
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                elif strategy == 'mode':
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                elif strategy == 'drop':
                    self.df.dropna(subset=[col], inplace=True)
        
        return self
    
    def convert_types(self, type_mapping: Dict[str, str]) -> 'DataCleaner':
        for col, dtype in type_mapping.items():
            if col in self.df.columns:
                try:
                    if dtype == 'datetime':
                        self.df[col] = pd.to_datetime(self.df[col])
                    else:
                        self.df[col] = self.df[col].astype(dtype)
                except Exception as e:
                    print(f"Error converting column {col} to {dtype}: {e}")
        
        return self
    
    def remove_duplicates(self, subset: Optional[List[str]] = None) -> 'DataCleaner':
        self.df.drop_duplicates(subset=subset, inplace=True, keep='first')
        return self
    
    def normalize_numeric(self, columns: Optional[List[str]] = None) -> 'DataCleaner':
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col in self.df.columns:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
        
        return self
    
    def get_cleaned_data(self) -> pd.DataFrame:
        return self.df
    
    def get_cleaning_report(self) -> Dict:
        report = {
            'original_shape': self.original_shape,
            'cleaned_shape': self.df.shape,
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.to_dict(),
            'numeric_columns': self.df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': self.df.select_dtypes(include=['object']).columns.tolist()
        }
        return report

def load_and_clean_csv(filepath: str, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(filepath, **kwargs)
    cleaner = DataCleaner(df)
    
    cleaner.handle_missing_values(strategy='mean')
    cleaner.remove_duplicates()
    
    numeric_cols = cleaner.df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        cleaner.normalize_numeric(columns=numeric_cols[:3])
    
    return cleaner.get_cleaned_data()def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df.reset_index(drop=True)

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    If no columns specified, clean all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            original_len = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            removed_count = original_len - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{col}'")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[10, 'A'] = 500
    df.loc[20, 'B'] = 1000
    
    print("Original DataFrame shape:", df.shape)
    print("\nOriginal Statistics:")
    for col in df.columns:
        stats = calculate_basic_stats(df, col)
        print(f"{col}: {stats}")
    
    cleaned_df = clean_numeric_data(df)
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("\nCleaned Statistics:")
    for col in cleaned_df.columns:
        stats = calculate_basic_stats(cleaned_df, col)
        print(f"{col}: {stats}")
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas DataFrame column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def calculate_summary_stats(data, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if data.empty:
        return {}
    
    stats = {
        'mean': np.mean(data[column]),
        'median': np.median(data[column]),
        'std': np.std(data[column]),
        'min': np.min(data[column]),
        'max': np.max(data[column]),
        'count': len(data[column])
    }
    return stats

def process_dataset(df, numeric_columns):
    """
    Process dataset by removing outliers from multiple numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of column names to process
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    dict: Dictionary of statistics for each column
    """
    cleaned_df = df.copy()
    all_stats = {}
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            removed_count = original_count - len(cleaned_df)
            
            stats = calculate_summary_stats(cleaned_df, col)
            stats['outliers_removed'] = removed_count
            all_stats[col] = stats
    
    return cleaned_df, all_stats