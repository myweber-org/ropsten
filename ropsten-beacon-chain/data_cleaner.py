
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to process
    factor (float): Multiplier for IQR (default 1.5)
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data.copy()

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to process
    threshold (float): Z-score threshold (default 3)
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    
    if len(z_scores) != len(data):
        valid_indices = data[column].dropna().index
        mask = pd.Series(True, index=data.index)
        mask.loc[valid_indices] = z_scores < threshold
    else:
        mask = z_scores < threshold
    
    filtered_data = data[mask]
    return filtered_data.copy()

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to normalize
    
    Returns:
    pd.Series: Normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    col_data = data[column].copy()
    min_val = col_data.min()
    max_val = col_data.max()
    
    if max_val == min_val:
        return pd.Series(0.5, index=col_data.index)
    
    normalized = (col_data - min_val) / (max_val - min_val)
    return normalized

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to normalize
    
    Returns:
    pd.Series: Standardized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    col_data = data[column].copy()
    mean_val = col_data.mean()
    std_val = col_data.std()
    
    if std_val == 0:
        return pd.Series(0, index=col_data.index)
    
    standardized = (col_data - mean_val) / std_val
    return standardized

def clean_dataset(data, numeric_columns=None, outlier_method='iqr', normalize_method=None):
    """
    Comprehensive data cleaning pipeline.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    numeric_columns (list): List of numeric columns to process
    outlier_method (str): 'iqr', 'zscore', or None
    normalize_method (str): 'minmax', 'zscore', or None
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    # Remove outliers
    if outlier_method:
        for col in numeric_columns:
            if col in cleaned_data.columns:
                if outlier_method == 'iqr':
                    cleaned_data = remove_outliers_iqr(cleaned_data, col)
                elif outlier_method == 'zscore':
                    cleaned_data = remove_outliers_zscore(cleaned_data, col)
    
    # Normalize data
    if normalize_method:
        for col in numeric_columns:
            if col in cleaned_data.columns:
                if normalize_method == 'minmax':
                    cleaned_data[col] = normalize_minmax(cleaned_data, col)
                elif normalize_method == 'zscore':
                    cleaned_data[col] = normalize_zscore(cleaned_data, col)
    
    return cleaned_data

def validate_data(data, required_columns=None, check_missing=True, check_duplicates=True):
    """
    Validate dataset for common issues.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    required_columns (list): List of required columns
    check_missing (bool): Check for missing values
    check_duplicates (bool): Check for duplicate rows
    
    Returns:
    dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'issues': [],
        'missing_counts': {},
        'duplicate_count': 0
    }
    
    # Check required columns
    if required_columns:
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Missing required columns: {missing_cols}")
    
    # Check missing values
    if check_missing:
        missing_counts = data.isnull().sum()
        missing_counts = missing_counts[missing_counts > 0].to_dict()
        if missing_counts:
            validation_results['missing_counts'] = missing_counts
            validation_results['issues'].append(f"Missing values found in columns: {list(missing_counts.keys())}")
    
    # Check duplicates
    if check_duplicates:
        duplicate_count = data.duplicated().sum()
        if duplicate_count > 0:
            validation_results['duplicate_count'] = duplicate_count
            validation_results['issues'].append(f"Found {duplicate_count} duplicate rows")
    
    return validation_resultsimport pandas as pd

def clean_dataset(df, id_column='id'):
    """
    Remove duplicate rows based on ID column and standardize column names.
    """
    # Standardize column names: strip whitespace and convert to lowercase
    df.columns = df.columns.str.strip().str.lower()
    
    # Remove duplicate rows based on the specified ID column
    if id_column in df.columns:
        df = df.drop_duplicates(subset=[id_column], keep='first')
    else:
        print(f"Warning: Column '{id_column}' not found. Skipping duplicate removal.")
    
    # Reset index after cleaning
    df = df.reset_index(drop=True)
    
    return df

def validate_data(df, required_columns):
    """
    Validate that required columns exist in the DataFrame.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'ID': [1, 2, 2, 3, 4],
        'Name': ['Alice', 'Bob', 'Bob', 'Charlie', 'David'],
        'Age': [25, 30, 30, 35, 28]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_dataset(df, id_column='id')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    try:
        validate_data(cleaned_df, ['id', 'name', 'age'])
        print("\nData validation passed.")
    except ValueError as e:
        print(f"\nData validation failed: {e}")import pandas as pd
import re

def clean_dataframe(df, columns_to_clean=None):
    """
    Clean a pandas DataFrame by removing duplicate rows and normalizing
    specified string columns (strip whitespace, convert to lowercase).
    If columns_to_clean is None, applies to all object dtype columns.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates().reset_index(drop=True)
    
    # Determine which columns to clean
    if columns_to_clean is None:
        columns_to_clean = df_cleaned.select_dtypes(include=['object']).columns
    
    # Normalize string columns
    for col in columns_to_clean:
        if col in df_cleaned.columns and df_cleaned[col].dtype == 'object':
            df_cleaned[col] = df_cleaned[col].astype(str).str.strip().str.lower()
    
    return df_cleaned

def remove_special_characters(text, regex_pattern=r'[^a-zA-Z0-9\s]'):
    """
    Remove special characters from a string using regex.
    Default pattern keeps only alphanumeric and whitespace.
    """
    if pd.isna(text):
        return text
    return re.sub(regex_pattern, '', str(text))

def validate_email(email):
    """
    Basic email validation using regex.
    Returns True if email matches pattern, False otherwise.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if pd.isna(email):
        return False
    return bool(re.match(pattern, str(email).strip()))import pandas as pd
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
            if col in self.df.columns and self.df[col].isnull().any():
                if strategy == 'mean':
                    fill_value = self.df[col].mean()
                elif strategy == 'median':
                    fill_value = self.df[col].median()
                elif strategy == 'mode':
                    fill_value = self.df[col].mode()[0]
                elif strategy == 'drop':
                    self.df = self.df.dropna(subset=[col])
                else:
                    fill_value = 0
                
                if strategy != 'drop':
                    self.df[col] = self.df[col].fillna(fill_value)
        
        return self
    
    def convert_dtypes(self, type_mapping: Dict[str, str]) -> 'DataCleaner':
        for col, dtype in type_mapping.items():
            if col in self.df.columns:
                try:
                    if dtype == 'datetime':
                        self.df[col] = pd.to_datetime(self.df[col])
                    else:
                        self.df[col] = self.df[col].astype(dtype)
                except Exception as e:
                    print(f"Warning: Could not convert {col} to {dtype}: {e}")
        
        return self
    
    def remove_duplicates(self, subset: Optional[List[str]] = None, keep: str = 'first') -> 'DataCleaner':
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
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
            'missing_values_before': self.df.isnull().sum().sum(),
            'missing_values_after': 0
        }
        return report

def clean_csv_file(input_path: str, output_path: str, cleaning_steps: Dict) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    cleaner = DataCleaner(df)
    
    if cleaning_steps.get('handle_missing'):
        cleaner.handle_missing_values(**cleaning_steps['handle_missing'])
    
    if cleaning_steps.get('convert_dtypes'):
        cleaner.convert_dtypes(cleaning_steps['convert_dtypes'])
    
    if cleaning_steps.get('remove_duplicates'):
        cleaner.remove_duplicates(**cleaning_steps['remove_duplicates'])
    
    cleaned_df = cleaner.get_cleaned_data()
    cleaned_df.to_csv(output_path, index=False)
    
    report = cleaner.get_cleaning_report()
    print(f"Cleaning complete. Report: {report}")
    
    return cleaned_df