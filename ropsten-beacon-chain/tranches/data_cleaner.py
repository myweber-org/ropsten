
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to remove duplicate rows.
    fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates()
    
    if fill_missing == 'drop':
        df_clean = df_clean.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
            elif fill_missing == 'median':
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
    elif fill_missing == 'mode':
        for col in df_clean.columns:
            df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else None, inplace=True)
    
    return df_clean

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    tuple: (bool, str) indicating validation result and message.
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Data validation passed"

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, None],
        'B': [5, None, 7, 8, 9],
        'C': ['x', 'y', 'x', 'z', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    is_valid, message = validate_data(cleaned_df, required_columns=['A', 'B'])
    print(f"\nValidation: {is_valid} - {message}")import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    data (pandas.DataFrame): The input dataframe.
    column (str): The column name to clean.
    
    Returns:
    pandas.DataFrame: Dataframe with outliers removed.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    return filtered_data.reset_index(drop=True)

def calculate_basic_stats(data, column):
    """
    Calculate basic statistics for a column after outlier removal.
    
    Parameters:
    data (pandas.DataFrame): The input dataframe.
    column (str): The column name to analyze.
    
    Returns:
    dict: Dictionary containing statistics.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    stats = {
        'mean': np.mean(data[column]),
        'median': np.median(data[column]),
        'std': np.std(data[column]),
        'min': np.min(data[column]),
        'max': np.max(data[column]),
        'count': len(data[column])
    }
    
    return stats
import pandas as pd
import numpy as np
from typing import List, Optional

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_duplicates(self, subset: Optional[List[str]] = None) -> 'DataCleaner':
        self.df = self.df.drop_duplicates(subset=subset, keep='first')
        return self
        
    def normalize_column(self, column_name: str) -> 'DataCleaner':
        if column_name in self.df.columns:
            col_data = self.df[column_name]
            if pd.api.types.is_numeric_dtype(col_data):
                mean_val = col_data.mean()
                std_val = col_data.std()
                if std_val > 0:
                    self.df[column_name] = (col_data - mean_val) / std_val
        return self
        
    def fill_missing_values(self, strategy: str = 'mean') -> 'DataCleaner':
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if self.df[col].isnull().any():
                if strategy == 'mean':
                    fill_value = self.df[col].mean()
                elif strategy == 'median':
                    fill_value = self.df[col].median()
                elif strategy == 'mode':
                    fill_value = self.df[col].mode()[0]
                else:
                    fill_value = 0
                    
                self.df[col] = self.df[col].fillna(fill_value)
                
        return self
        
    def remove_outliers(self, column_name: str, n_std: float = 3) -> 'DataCleaner':
        if column_name in self.df.columns and pd.api.types.is_numeric_dtype(self.df[column_name]):
            mean_val = self.df[column_name].mean()
            std_val = self.df[column_name].std()
            
            lower_bound = mean_val - n_std * std_val
            upper_bound = mean_val + n_std * std_val
            
            self.df = self.df[
                (self.df[column_name] >= lower_bound) & 
                (self.df[column_name] <= upper_bound)
            ]
            
        return self
        
    def get_cleaned_data(self) -> pd.DataFrame:
        return self.df
        
    def get_cleaning_report(self) -> dict:
        cleaned_shape = self.df.shape
        return {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_rows': cleaned_shape[0],
            'cleaned_columns': cleaned_shape[1],
            'rows_removed': self.original_shape[0] - cleaned_shape[0],
            'columns_removed': self.original_shape[1] - cleaned_shape[1]
        }

def create_sample_data() -> pd.DataFrame:
    data = {
        'id': [1, 2, 3, 4, 5, 5, 6],
        'value': [10.5, 20.3, 15.7, np.nan, 1000.0, 10.5, 12.8],
        'category': ['A', 'B', 'A', 'B', 'C', 'A', 'B']
    }
    return pd.DataFrame(data)

if __name__ == "__main__":
    sample_df = create_sample_data()
    print("Original data:")
    print(sample_df)
    print("\n" + "="*50 + "\n")
    
    cleaner = DataCleaner(sample_df)
    cleaned_df = (cleaner
                 .remove_duplicates(['id'])
                 .fill_missing_values('mean')
                 .remove_outliers('value', 2)
                 .normalize_column('value')
                 .get_cleaned_data())
    
    print("Cleaned data:")
    print(cleaned_df)
    print("\nCleaning report:")
    print(cleaner.get_cleaning_report())
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
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        removed_count = len(self.df) - len(df_clean)
        self.df = df_clean
        return removed_count
    
    def remove_outliers_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_clean = self.df.copy()
        for col in columns:
            if col in df_clean.columns:
                z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
                df_clean = df_clean[(z_scores < threshold) | df_clean[col].isna()]
        
        removed_count = len(self.df) - len(df_clean)
        self.df = df_clean
        return removed_count
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_normalized = self.df.copy()
        for col in columns:
            if col in df_normalized.columns:
                min_val = df_normalized[col].min()
                max_val = df_normalized[col].max()
                if max_val > min_val:
                    df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
        
        self.df = df_normalized
        return self.df
    
    def normalize_zscore(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_normalized = self.df.copy()
        for col in columns:
            if col in df_normalized.columns:
                mean_val = df_normalized[col].mean()
                std_val = df_normalized[col].std()
                if std_val > 0:
                    df_normalized[col] = (df_normalized[col] - mean_val) / std_val
        
        self.df = df_normalized
        return self.df
    
    def fill_missing_mean(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_filled = self.df.copy()
        for col in columns:
            if col in df_filled.columns:
                mean_val = df_filled[col].mean()
                df_filled[col] = df_filled[col].fillna(mean_val)
        
        self.df = df_filled
        return self.df
    
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_filled = self.df.copy()
        for col in columns:
            if col in df_filled.columns:
                median_val = df_filled[col].median()
                df_filled[col] = df_filled[col].fillna(median_val)
        
        self.df = df_filled
        return self.df
    
    def get_cleaned_data(self):
        return self.df
    
    def get_removal_stats(self):
        cleaned_count = len(self.df)
        original_count = self.original_shape[0]
        removed_count = original_count - cleaned_count
        removal_percentage = (removed_count / original_count * 100) if original_count > 0 else 0
        
        return {
            'original_rows': original_count,
            'cleaned_rows': cleaned_count,
            'removed_rows': removed_count,
            'removal_percentage': removal_percentage
        }import re
import pandas as pd
from typing import Union, List, Optional

def normalize_string(text: str, case: str = 'lower', strip: bool = True) -> str:
    """
    Normalize a string by adjusting case and stripping whitespace.
    """
    if not isinstance(text, str):
        return text

    if strip:
        text = text.strip()

    if case == 'lower':
        text = text.lower()
    elif case == 'upper':
        text = text.upper()
    elif case == 'title':
        text = text.title()

    return text

def remove_special_chars(text: str, keep_chars: str = '') -> str:
    """
    Remove special characters from a string, optionally keeping specified characters.
    """
    if not isinstance(text, str):
        return text

    pattern = f'[^A-Za-z0-9\\s{re.escape(keep_chars)}]'
    return re.sub(pattern, '', text)

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column names of a DataFrame by normalizing and removing special characters.
    """
    cleaned_names = []
    for col in df.columns:
        col_str = str(col)
        col_str = normalize_string(col_str, case='lower')
        col_str = remove_special_chars(col_str, keep_chars='_')
        col_str = re.sub(r'\\s+', '_', col_str)
        cleaned_names.append(col_str)
    df.columns = cleaned_names
    return df

def fill_missing_with_mean(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Fill missing values in specified columns with the column mean.
    If columns is None, applies to all numeric columns.
    """
    df_filled = df.copy()
    if columns is None:
        columns = df_filled.select_dtypes(include=['number']).columns.tolist()

    for col in columns:
        if col in df_filled.columns and pd.api.types.is_numeric_dtype(df_filled[col]):
            mean_val = df_filled[col].mean()
            df_filled[col].fillna(mean_val, inplace=True)

    return df_filled

def drop_duplicates_subset(df: pd.DataFrame, subset: Optional[List[str]] = None, keep: str = 'first') -> pd.DataFrame:
    """
    Drop duplicate rows based on a subset of columns.
    """
    return df.drop_duplicates(subset=subset, keep=keep)
import pandas as pd
import re

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    column_mapping (dict): Optional dictionary to rename columns
    drop_duplicates (bool): Whether to remove duplicate rows
    normalize_text (bool): Whether to normalize text in string columns
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    
    cleaned_df = df.copy()
    
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    
    if normalize_text:
        for col in cleaned_df.select_dtypes(include=['object']).columns:
            cleaned_df[col] = cleaned_df[col].apply(_normalize_string)
    
    return cleaned_df

def _normalize_string(text):
    """
    Normalize a string by converting to lowercase, removing extra whitespace,
    and stripping special characters.
    """
    if pd.isna(text):
        return text
    
    text = str(text)
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s-]', '', text)
    
    return text

def validate_email_column(df, email_column):
    """
    Validate email addresses in a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    email_column (str): Name of the column containing email addresses
    
    Returns:
    pd.Series: Boolean series indicating valid emails
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return df[email_column].astype(str).str.match(email_pattern)

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to check for outliers
    multiplier (float): IQR multiplier for outlier detection
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' must be numeric")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]