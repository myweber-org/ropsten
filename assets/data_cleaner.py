
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_strategy='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to remove duplicate rows.
    fill_missing (bool): Whether to fill missing values.
    fill_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', or 'constant').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        for column in cleaned_df.columns:
            if cleaned_df[column].dtype in [np.float64, np.int64]:
                if fill_strategy == 'mean':
                    cleaned_df[column].fillna(cleaned_df[column].mean(), inplace=True)
                elif fill_strategy == 'median':
                    cleaned_df[column].fillna(cleaned_df[column].median(), inplace=True)
                elif fill_strategy == 'constant':
                    cleaned_df[column].fillna(0, inplace=True)
            elif cleaned_df[column].dtype == 'object':
                if fill_strategy == 'mode':
                    cleaned_df[column].fillna(cleaned_df[column].mode()[0], inplace=True)
                elif fill_strategy == 'constant':
                    cleaned_df[column].fillna('Unknown', inplace=True)
    
    return cleaned_df

def validate_dataset(df, check_missing=True, check_types=True):
    """
    Validate a pandas DataFrame for common data quality issues.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    check_missing (bool): Check for missing values.
    check_types (bool): Check for consistent data types.
    
    Returns:
    dict: Dictionary containing validation results.
    """
    validation_results = {}
    
    if check_missing:
        missing_counts = df.isnull().sum()
        validation_results['missing_values'] = missing_counts[missing_counts > 0].to_dict()
    
    if check_types:
        type_summary = {}
        for column in df.columns:
            type_summary[column] = str(df[column].dtype)
        validation_results['data_types'] = type_summary
    
    validation_results['shape'] = df.shape
    validation_results['columns'] = list(df.columns)
    
    return validation_results

def normalize_columns(df, columns=None, method='minmax'):
    """
    Normalize specified columns in a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns (list): List of columns to normalize. If None, normalize all numeric columns.
    method (str): Normalization method ('minmax' or 'zscore').
    
    Returns:
    pd.DataFrame: DataFrame with normalized columns.
    """
    normalized_df = df.copy()
    
    if columns is None:
        numeric_cols = normalized_df.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    for column in columns:
        if column in normalized_df.columns and normalized_df[column].dtype in [np.float64, np.int64]:
            if method == 'minmax':
                min_val = normalized_df[column].min()
                max_val = normalized_df[column].max()
                if max_val > min_val:
                    normalized_df[column] = (normalized_df[column] - min_val) / (max_val - min_val)
            elif method == 'zscore':
                mean_val = normalized_df[column].mean()
                std_val = normalized_df[column].std()
                if std_val > 0:
                    normalized_df[column] = (normalized_df[column] - mean_val) / std_val
    
    return normalized_df

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, 2, 4, 5, None],
        'B': [10.5, 20.3, 20.3, 40.1, 50.0, 60.2],
        'C': ['apple', 'banana', 'banana', 'cherry', None, 'elderberry']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaned = clean_dataset(df, fill_strategy='mean')
    print("Cleaned DataFrame:")
    print(cleaned)
    print("\n")
    
    validation = validate_dataset(cleaned)
    print("Validation Results:")
    print(validation)
    print("\n")
    
    normalized = normalize_columns(cleaned, method='minmax')
    print("Normalized DataFrame:")
    print(normalized)
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    return dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]

def remove_outliers_zscore(dataframe, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    z_scores = np.abs(stats.zscore(dataframe[column]))
    return dataframe[z_scores < threshold]

def normalize_minmax(dataframe, column):
    """
    Normalize column using Min-Max scaling
    """
    min_val = dataframe[column].min()
    max_val = dataframe[column].max()
    
    if max_val == min_val:
        return dataframe[column].apply(lambda x: 0.5)
    
    return (dataframe[column] - min_val) / (max_val - min_val)

def normalize_zscore(dataframe, column):
    """
    Normalize column using Z-score standardization
    """
    mean_val = dataframe[column].mean()
    std_val = dataframe[column].std()
    
    if std_val == 0:
        return dataframe[column].apply(lambda x: 0)
    
    return (dataframe[column] - mean_val) / std_val

def clean_dataset(dataframe, numeric_columns, outlier_method='iqr', normalize_method='zscore'):
    """
    Main cleaning function for numeric columns
    """
    cleaned_df = dataframe.copy()
    
    for column in numeric_columns:
        if column not in cleaned_df.columns:
            continue
            
        # Remove outliers
        if outlier_method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
        elif outlier_method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, column)
        
        # Normalize data
        if normalize_method == 'minmax':
            cleaned_df[f'{column}_normalized'] = normalize_minmax(cleaned_df, column)
        elif normalize_method == 'zscore':
            cleaned_df[f'{column}_normalized'] = normalize_zscore(cleaned_df, column)
    
    return cleaned_df

def validate_dataframe(dataframe, required_columns):
    """
    Validate dataframe structure and content
    """
    missing_columns = [col for col in required_columns if col not in dataframe.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    if dataframe.empty:
        raise ValueError("DataFrame is empty")
    
    return True
import pandas as pd
import numpy as np
from typing import Optional, List

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_duplicates(self, subset: Optional[List[str]] = None, keep: str = 'first') -> 'DataCleaner':
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        return self
        
    def handle_missing_values(self, strategy: str = 'drop', fill_value: Optional[float] = None) -> 'DataCleaner':
        if strategy == 'drop':
            self.df = self.df.dropna()
        elif strategy == 'fill':
            if fill_value is not None:
                self.df = self.df.fillna(fill_value)
            else:
                self.df = self.df.fillna(self.df.mean())
        return self
        
    def normalize_column(self, column: str, method: str = 'minmax') -> 'DataCleaner':
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
            
        if method == 'minmax':
            col_min = self.df[column].min()
            col_max = self.df[column].max()
            if col_max != col_min:
                self.df[column] = (self.df[column] - col_min) / (col_max - col_min)
        elif method == 'zscore':
            col_mean = self.df[column].mean()
            col_std = self.df[column].std()
            if col_std > 0:
                self.df[column] = (self.df[column] - col_mean) / col_std
                
        return self
        
    def get_cleaned_data(self) -> pd.DataFrame:
        return self.df.copy()
        
    def get_summary(self) -> dict:
        cleaned_shape = self.df.shape
        return {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_rows': cleaned_shape[0],
            'cleaned_columns': cleaned_shape[1],
            'rows_removed': self.original_shape[0] - cleaned_shape[0],
            'missing_values': self.df.isnull().sum().sum()
        }

def clean_dataset(df: pd.DataFrame, 
                  remove_dups: bool = True,
                  handle_nulls: str = 'drop',
                  normalize_cols: Optional[List[str]] = None) -> pd.DataFrame:
    
    cleaner = DataCleaner(df)
    
    if remove_dups:
        cleaner.remove_duplicates()
        
    cleaner.handle_missing_values(strategy=handle_nulls)
    
    if normalize_cols:
        for col in normalize_cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                cleaner.normalize_column(col)
    
    return cleaner.get_cleaned_data()