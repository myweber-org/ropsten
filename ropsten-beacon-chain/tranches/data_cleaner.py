
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
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                
                mask = (self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)
                df_clean = df_clean[mask]
                
        self.df = df_clean.reset_index(drop=True)
        return self
        
    def remove_outliers_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_clean = self.df.copy()
        for col in columns:
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                z_scores = np.abs(stats.zscore(self.df[col].fillna(self.df[col].mean())))
                mask = z_scores < threshold
                df_clean = df_clean[mask]
                
        self.df = df_clean.reset_index(drop=True)
        return self
        
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_normalized = self.df.copy()
        for col in columns:
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    df_normalized[col] = (self.df[col] - min_val) / (max_val - min_val)
                    
        self.df = df_normalized
        return self
        
    def normalize_zscore(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_normalized = self.df.copy()
        for col in columns:
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    df_normalized[col] = (self.df[col] - mean_val) / std_val
                    
        self.df = df_normalized
        return self
        
    def fill_missing(self, strategy='mean', columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_filled = self.df.copy()
        for col in columns:
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                if strategy == 'mean':
                    fill_value = self.df[col].mean()
                elif strategy == 'median':
                    fill_value = self.df[col].median()
                elif strategy == 'mode':
                    fill_value = self.df[col].mode()[0] if not self.df[col].mode().empty else 0
                else:
                    fill_value = 0
                    
                df_filled[col] = self.df[col].fillna(fill_value)
                
        self.df = df_filled
        return self
        
    def get_cleaned_data(self):
        return self.df
        
    def get_removed_count(self):
        return self.original_shape[0] - self.df.shape[0]
        
    def summary(self):
        print(f"Original data shape: {self.original_shape}")
        print(f"Cleaned data shape: {self.df.shape}")
        print(f"Rows removed: {self.get_removed_count()}")
        print(f"Columns: {list(self.df.columns)}")
import numpy as np
import pandas as pd
from scipy import stats

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using Interquartile Range method.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to check for outliers
    threshold (float): Multiplier for IQR (default 1.5)
    
    Returns:
    pd.Series: Boolean series indicating outliers
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    return (data[column] < lower_bound) | (data[column] > upper_bound)

def remove_outliers(data, column, method='iqr', **kwargs):
    """
    Remove outliers from a dataframe column.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name
    method (str): 'iqr' or 'zscore'
    **kwargs: Additional arguments for outlier detection
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed
    """
    if method == 'iqr':
        threshold = kwargs.get('threshold', 1.5)
        outliers = detect_outliers_iqr(data, column, threshold)
    elif method == 'zscore':
        zscore_threshold = kwargs.get('zscore_threshold', 3)
        z_scores = np.abs(stats.zscore(data[column].dropna()))
        outliers = pd.Series(z_scores > zscore_threshold, index=data[column].dropna().index)
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    return data[~outliers]

def normalize_column(data, column, method='minmax'):
    """
    Normalize a column using specified method.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to normalize
    method (str): 'minmax' or 'zscore'
    
    Returns:
    pd.DataFrame: Dataframe with normalized column
    """
    data_copy = data.copy()
    
    if method == 'minmax':
        min_val = data_copy[column].min()
        max_val = data_copy[column].max()
        data_copy[column] = (data_copy[column] - min_val) / (max_val - min_val)
    elif method == 'zscore':
        mean_val = data_copy[column].mean()
        std_val = data_copy[column].std()
        data_copy[column] = (data_copy[column] - mean_val) / std_val
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    return data_copy

def clean_dataset(data, numeric_columns, outlier_method='iqr', normalize_method=None):
    """
    Comprehensive data cleaning pipeline.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    numeric_columns (list): List of numeric column names to clean
    outlier_method (str): Method for outlier detection
    normalize_method (str): Normalization method or None
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column in cleaned_data.columns:
            cleaned_data = remove_outliers(cleaned_data, column, method=outlier_method)
            
            if normalize_method:
                cleaned_data = normalize_column(cleaned_data, column, method=normalize_method)
    
    return cleaned_data

def get_data_summary(data):
    """
    Generate statistical summary of dataframe.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    summary = {
        'shape': data.shape,
        'columns': list(data.columns),
        'dtypes': data.dtypes.to_dict(),
        'missing_values': data.isnull().sum().to_dict(),
        'numeric_stats': data.describe().to_dict() if data.select_dtypes(include=[np.number]).shape[1] > 0 else {}
    }
    
    return summary