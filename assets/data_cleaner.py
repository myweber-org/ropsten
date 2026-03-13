
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
        
        self.df = df_clean
        removed_count = self.original_shape[0] - self.df.shape[0]
        return removed_count
    
    def normalize_data(self, columns=None, method='zscore'):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_normalized = self.df.copy()
        for col in columns:
            if col in df_normalized.columns:
                if method == 'zscore':
                    df_normalized[col] = stats.zscore(df_normalized[col])
                elif method == 'minmax':
                    min_val = df_normalized[col].min()
                    max_val = df_normalized[col].max()
                    if max_val != min_val:
                        df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
                elif method == 'robust':
                    median = df_normalized[col].median()
                    iqr = df_normalized[col].quantile(0.75) - df_normalized[col].quantile(0.25)
                    if iqr != 0:
                        df_normalized[col] = (df_normalized[col] - median) / iqr
        
        self.df = df_normalized
        return self.df
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_filled = self.df.copy()
        for col in columns:
            if col in df_filled.columns and df_filled[col].isnull().any():
                if strategy == 'mean':
                    fill_value = df_filled[col].mean()
                elif strategy == 'median':
                    fill_value = df_filled[col].median()
                elif strategy == 'mode':
                    fill_value = df_filled[col].mode()[0]
                elif strategy == 'drop':
                    df_filled = df_filled.dropna(subset=[col])
                    continue
                
                df_filled[col] = df_filled[col].fillna(fill_value)
        
        self.df = df_filled
        return self.df
    
    def get_clean_data(self):
        return self.df.copy()
    
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'current_rows': self.df.shape[0],
            'original_columns': self.original_shape[1],
            'current_columns': self.df.shape[1],
            'missing_values': self.df.isnull().sum().sum(),
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(include=['object']).columns)
        }
        return summary

def example_usage():
    np.random.seed(42)
    data = {
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    df = pd.DataFrame(data)
    
    cleaner = DataCleaner(df)
    print(f"Original shape: {cleaner.original_shape}")
    
    removed = cleaner.remove_outliers_iqr(['feature1', 'feature2'])
    print(f"Removed {removed} outliers")
    
    cleaner.handle_missing_values(strategy='mean')
    cleaner.normalize_data(method='zscore')
    
    clean_df = cleaner.get_clean_data()
    summary = cleaner.get_summary()
    
    print(f"Clean data shape: {clean_df.shape}")
    print(f"Summary: {summary}")
    
    return clean_df

if __name__ == "__main__":
    example_usage()
import pandas as pd
import numpy as np
from typing import List, Optional

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_duplicates(self, subset: Optional[List[str]] = None) -> pd.DataFrame:
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep='first')
        removed = initial_count - len(self.df)
        print(f"Removed {removed} duplicate rows")
        return self.df
    
    def normalize_text_columns(self, columns: List[str]) -> pd.DataFrame:
        for col in columns:
            if col in self.df.columns and self.df[col].dtype == 'object':
                self.df[col] = self.df[col].str.strip().str.lower()
                self.df[col] = self.df[col].replace('', np.nan)
        print(f"Normalized text columns: {columns}")
        return self.df
    
    def fill_missing_values(self, strategy: str = 'mean', columns: Optional[List[str]] = None) -> pd.DataFrame:
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col in self.df.columns:
                if strategy == 'mean' and self.df[col].dtype in [np.float64, np.int64]:
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                elif strategy == 'median' and self.df[col].dtype in [np.float64, np.int64]:
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                elif strategy == 'mode':
                    self.df[col].fillna(self.df[col].mode()[0] if not self.df[col].mode().empty else None, inplace=True)
                elif strategy == 'ffill':
                    self.df[col].fillna(method='ffill', inplace=True)
                elif strategy == 'bfill':
                    self.df[col].fillna(method='bfill', inplace=True)
        
        print(f"Filled missing values using {strategy} strategy for columns: {columns}")
        return self.df
    
    def remove_outliers_iqr(self, columns: List[str], multiplier: float = 1.5) -> pd.DataFrame:
        initial_count = len(self.df)
        
        for col in columns:
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        
        removed = initial_count - len(self.df)
        print(f"Removed {removed} outliers using IQR method from columns: {columns}")
        return self.df
    
    def get_cleaning_report(self) -> dict:
        final_shape = self.df.shape
        return {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'final_rows': final_shape[0],
            'final_columns': final_shape[1],
            'rows_removed': self.original_shape[0] - final_shape[0],
            'columns_removed': self.original_shape[1] - final_shape[1],
            'remaining_missing_values': self.df.isnull().sum().sum()
        }
    
    def save_cleaned_data(self, filepath: str, format: str = 'csv'):
        if format == 'csv':
            self.df.to_csv(filepath, index=False)
        elif format == 'excel':
            self.df.to_excel(filepath, index=False)
        elif format == 'parquet':
            self.df.to_parquet(filepath, index=False)
        print(f"Cleaned data saved to {filepath} in {format} format")

def create_sample_data() -> pd.DataFrame:
    data = {
        'id': [1, 2, 3, 4, 5, 5, 6, 7],
        'name': ['John', 'Jane', 'Bob', 'Alice', 'John', 'John', 'Eve', 'Frank'],
        'age': [25, 30, 35, 28, 25, 25, 40, 150],
        'salary': [50000, 60000, 70000, None, 50000, 50000, 80000, 1000000],
        'department': ['IT', 'HR', 'IT', 'Finance', 'IT', 'IT', 'HR', 'Management']
    }
    return pd.DataFrame(data)

if __name__ == "__main__":
    sample_df = create_sample_data()
    cleaner = DataCleaner(sample_df)
    
    cleaner.remove_duplicates(subset=['id', 'name'])
    cleaner.normalize_text_columns(['name', 'department'])
    cleaner.fill_missing_values(strategy='mean', columns=['salary'])
    cleaner.remove_outliers_iqr(columns=['age', 'salary'], multiplier=1.5)
    
    report = cleaner.get_cleaning_report()
    print("\nCleaning Report:")
    for key, value in report.items():
        print(f"{key}: {value}")
    
    cleaner.save_cleaned_data('cleaned_data.csv')
import pandas as pd

def clean_dataset(df, drop_na=True, column_case='lower'):
    """
    Clean a pandas DataFrame by handling missing values and standardizing column names.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_na (bool): If True, drop rows with any null values. Default True.
    column_case (str): Target case for column names ('lower', 'upper', 'title'). Default 'lower'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_na:
        cleaned_df = cleaned_df.dropna()
    
    if column_case == 'lower':
        cleaned_df.columns = cleaned_df.columns.str.lower()
    elif column_case == 'upper':
        cleaned_df.columns = cleaned_df.columns.str.upper()
    elif column_case == 'title':
        cleaned_df.columns = cleaned_df.columns.str.title()
    
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    subset (list): Columns to consider for duplicates. If None, all columns are used.
    keep (str): Which duplicates to keep ('first', 'last', False). Default 'first'.
    
    Returns:
    pd.DataFrame: DataFrame with duplicates removed.
    """
    return df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)

def validate_numeric_columns(df, columns):
    """
    Convert specified columns to numeric, coercing errors to NaN.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns (list): List of column names to convert.
    
    Returns:
    pd.DataFrame: DataFrame with converted columns.
    """
    validated_df = df.copy()
    for col in columns:
        if col in validated_df.columns:
            validated_df[col] = pd.to_numeric(validated_df[col], errors='coerce')
    return validated_df
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if min_val == max_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    normalized = (data[column] - mean_val) / std_val
    return normalized

def handle_missing_values(data, strategy='mean'):
    """
    Handle missing values in numeric columns
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    if strategy == 'mean':
        for col in numeric_cols:
            data[col] = data[col].fillna(data[col].mean())
    elif strategy == 'median':
        for col in numeric_cols:
            data[col] = data[col].fillna(data[col].median())
    elif strategy == 'mode':
        for col in numeric_cols:
            data[col] = data[col].fillna(data[col].mode()[0])
    elif strategy == 'drop':
        data = data.dropna(subset=numeric_cols)
    else:
        raise ValueError("Strategy must be 'mean', 'median', 'mode', or 'drop'")
    
    return data

def clean_dataset(data, outlier_method='iqr', normalize_method='minmax', missing_strategy='mean'):
    """
    Comprehensive data cleaning pipeline
    """
    cleaned_data = data.copy()
    
    numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if outlier_method == 'iqr':
            cleaned_data = remove_outliers_iqr(cleaned_data, col)
        elif outlier_method == 'zscore':
            cleaned_data = remove_outliers_zscore(cleaned_data, col)
    
    cleaned_data = handle_missing_values(cleaned_data, strategy=missing_strategy)
    
    for col in numeric_cols:
        if col in cleaned_data.columns:
            if normalize_method == 'minmax':
                cleaned_data[col] = normalize_minmax(cleaned_data, col)
            elif normalize_method == 'zscore':
                cleaned_data[col] = normalize_zscore(cleaned_data, col)
    
    return cleaned_data