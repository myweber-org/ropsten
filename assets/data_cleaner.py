import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        subset (list, optional): Column labels to consider for duplicates.
        keep (str, optional): Which duplicates to keep.

    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    if df.empty:
        return df
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    return cleaned_df

def clean_numeric_column(df, column_name, fill_method='mean'):
    """
    Clean a numeric column by filling missing values.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column_name (str): Name of the column to clean.
        fill_method (str): Method to fill missing values.

    Returns:
        pd.DataFrame: DataFrame with cleaned column.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    if fill_method == 'mean':
        fill_value = df[column_name].mean()
    elif fill_method == 'median':
        fill_value = df[column_name].median()
    elif fill_method == 'mode':
        fill_value = df[column_name].mode()[0]
    else:
        fill_value = 0
    
    df[column_name] = df[column_name].fillna(fill_value)
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.

    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.

    Returns:
        bool: True if validation passes, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    return True
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
            if col in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[col]):
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        removed_count = self.original_shape[0] - df_clean.shape[0]
        self.df = df_clean
        return removed_count
    
    def remove_outliers_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_clean = self.df.copy()
        for col in columns:
            if col in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[col]):
                z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
                mask = z_scores < threshold
                valid_indices = df_clean[col].dropna().index[mask]
                df_clean = df_clean.loc[df_clean.index.isin(valid_indices) | df_clean[col].isna()]
        
        removed_count = self.original_shape[0] - df_clean.shape[0]
        self.df = df_clean
        return removed_count
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_normalized = self.df.copy()
        for col in columns:
            if col in df_normalized.columns and pd.api.types.is_numeric_dtype(df_normalized[col]):
                col_min = df_normalized[col].min()
                col_max = df_normalized[col].max()
                if col_max > col_min:
                    df_normalized[col] = (df_normalized[col] - col_min) / (col_max - col_min)
        
        self.df = df_normalized
        return self
    
    def normalize_zscore(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_normalized = self.df.copy()
        for col in columns:
            if col in df_normalized.columns and pd.api.types.is_numeric_dtype(df_normalized[col]):
                col_mean = df_normalized[col].mean()
                col_std = df_normalized[col].std()
                if col_std > 0:
                    df_normalized[col] = (df_normalized[col] - col_mean) / col_std
        
        self.df = df_normalized
        return self
    
    def fill_missing(self, strategy='mean', columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_filled = self.df.copy()
        for col in columns:
            if col in df_filled.columns and pd.api.types.is_numeric_dtype(df_filled[col]):
                if strategy == 'mean':
                    fill_value = df_filled[col].mean()
                elif strategy == 'median':
                    fill_value = df_filled[col].median()
                elif strategy == 'mode':
                    fill_value = df_filled[col].mode()[0] if not df_filled[col].mode().empty else 0
                elif strategy == 'zero':
                    fill_value = 0
                else:
                    fill_value = df_filled[col].mean()
                
                df_filled[col] = df_filled[col].fillna(fill_value)
        
        self.df = df_filled
        return self
    
    def get_dataframe(self):
        return self.df.copy()
    
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'current_rows': self.df.shape[0],
            'columns': self.df.shape[1],
            'removed_rows': self.original_shape[0] - self.df.shape[0],
            'missing_values': self.df.isna().sum().sum(),
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(include=['object', 'category']).columns)
        }
        return summary

def example_usage():
    np.random.seed(42)
    data = {
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'feature3': np.random.randint(1, 100, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    df = pd.DataFrame(data)
    df.iloc[10:20, 0] = np.nan
    df.iloc[50:60, 1] = np.nan
    
    cleaner = DataCleaner(df)
    print("Initial summary:", cleaner.get_summary())
    
    removed = cleaner.remove_outliers_iqr(['feature1', 'feature2'])
    print(f"Removed {removed} outliers using IQR method")
    
    cleaner.fill_missing(strategy='mean')
    cleaner.normalize_minmax(['feature1', 'feature2', 'feature3'])
    
    print("Final summary:", cleaner.get_summary())
    print("First 5 rows of cleaned data:")
    print(cleaner.get_dataframe().head())
    
    return cleaner.get_dataframe()

if __name__ == "__main__":
    cleaned_df = example_usage()