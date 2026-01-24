
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    cleaned_df = cleaned_df.dropna()
    cleaned_df = cleaned_df.reset_index(drop=True)
    return cleaned_df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': np.random.normal(100, 15, 200),
        'B': np.random.exponential(50, 200),
        'C': np.random.uniform(0, 1, 200)
    })
    sample_data.loc[10:15, 'A'] = 500
    sample_data.loc[20:25, 'B'] = 1000
    
    numeric_cols = ['A', 'B', 'C']
    result = clean_dataset(sample_data, numeric_cols)
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {result.shape}")
    print(f"Removed {len(sample_data) - len(result)} outliers")
import numpy as np

def remove_outliers_iqr(data, column):
    Q1 = np.percentile(data[column], 25)
    Q3 = np.percentile(data[column], 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_dataimport pandas as pd
import numpy as np
from scipy import stats

def load_data(filepath):
    """Load dataset from CSV file."""
    return pd.read_csv(filepath)

def remove_outliers_iqr(df, column):
    """Remove outliers using IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_column(df, column):
    """Normalize column values to range [0,1]."""
    min_val = df[column].min()
    max_val = df[column].max()
    df[column] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(input_file, output_file):
    """Main cleaning pipeline."""
    df = load_data(input_file)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        df = remove_outliers_iqr(df, col)
        df = normalize_column(df, col)
    
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")
    return df

if __name__ == "__main__":
    cleaned_df = clean_dataset('raw_data.csv', 'cleaned_data.csv')
import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        subset (list, optional): Column names to consider for duplicates
        keep (str, optional): Which duplicates to keep ('first', 'last', False)
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """
    if subset is None:
        subset = df.columns.tolist()
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(df) - len(cleaned_df)
    print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df

def clean_numeric_columns(df, columns):
    """
    Clean numeric columns by converting to numeric and filling NaN.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of column names to clean
    
    Returns:
        pd.DataFrame: DataFrame with cleaned numeric columns
    """
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
    
    return cleaned_df

def standardize_text(df, columns):
    """
    Standardize text columns by converting to lowercase and stripping whitespace.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of column names to standardize
    
    Returns:
        pd.DataFrame: DataFrame with standardized text columns
    """
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].astype(str).str.lower().str.strip()
    
    return cleaned_df

def clean_dataframe(df, duplicate_subset=None, numeric_cols=None, text_cols=None):
    """
    Comprehensive data cleaning function.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        duplicate_subset (list, optional): Columns for duplicate removal
        numeric_cols (list, optional): Numeric columns to clean
        text_cols (list, optional): Text columns to standardize
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if duplicate_subset is not None:
        cleaned_df = remove_duplicates(cleaned_df, subset=duplicate_subset)
    
    if numeric_cols is not None:
        cleaned_df = clean_numeric_columns(cleaned_df, numeric_cols)
    
    if text_cols is not None:
        cleaned_df = standardize_text(cleaned_df, text_cols)
    
    return cleaned_dfimport pandas as pd
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
                    raise ValueError(f"Unknown strategy: {strategy}")
                    
                if strategy != 'drop':
                    self.df[col] = self.df[col].fillna(fill_value)
                    
        return self
    
    def convert_types(self, type_map: Dict[str, str]) -> 'DataCleaner':
        for col, dtype in type_map.items():
            if col in self.df.columns:
                try:
                    self.df[col] = self.df[col].astype(dtype)
                except ValueError as e:
                    print(f"Warning: Could not convert {col} to {dtype}: {e}")
        return self
    
    def remove_outliers(self, columns: List[str], method: str = 'iqr', threshold: float = 1.5) -> 'DataCleaner':
        for col in columns:
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                if method == 'iqr':
                    Q1 = self.df[col].quantile(0.25)
                    Q3 = self.df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
                    
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
    df = pd.read_csv(input_path)
    cleaner = DataCleaner(df)
    
    if 'missing_strategy' in kwargs:
        cleaner.handle_missing_values(strategy=kwargs['missing_strategy'])
    
    if 'type_map' in kwargs:
        cleaner.convert_types(kwargs['type_map'])
    
    if 'outlier_columns' in kwargs:
        cleaner.remove_outliers(columns=kwargs['outlier_columns'])
    
    cleaned_df = cleaner.get_cleaned_data()
    cleaned_df.to_csv(output_path, index=False)
    
    return cleaner.get_summary()