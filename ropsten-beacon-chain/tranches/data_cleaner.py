import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    def detect_outliers_iqr(self, column, threshold=1.5):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
        return outliers.index.tolist()
    
    def remove_outliers(self, columns=None, threshold=1.5):
        if columns is None:
            columns = self.numeric_columns
        
        outlier_indices = []
        for col in columns:
            if col in self.numeric_columns:
                outlier_indices.extend(self.detect_outliers_iqr(col, threshold))
        
        unique_outliers = list(set(outlier_indices))
        self.df = self.df.drop(unique_outliers)
        return self.df
    
    def impute_missing_mean(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        for col in columns:
            if col in self.numeric_columns and self.df[col].isnull().any():
                mean_val = self.df[col].mean()
                self.df[col].fillna(mean_val, inplace=True)
        return self.df
    
    def impute_missing_median(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        for col in columns:
            if col in self.numeric_columns and self.df[col].isnull().any():
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
        return self.df
    
    def standardize_data(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        for col in columns:
            if col in self.numeric_columns:
                mean = self.df[col].mean()
                std = self.df[col].std()
                if std > 0:
                    self.df[col] = (self.df[col] - mean) / std
        return self.df
    
    def normalize_data(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        for col in columns:
            if col in self.numeric_columns:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
        return self.df
    
    def get_clean_data(self):
        return self.df.copy()
    
    def summary(self):
        summary_dict = {
            'original_shape': self.df.shape,
            'missing_values': self.df.isnull().sum().sum(),
            'numeric_columns': len(self.numeric_columns),
            'categorical_columns': len(self.df.select_dtypes(exclude=[np.number]).columns)
        }
        return summary_dict

def example_usage():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'feature_c': np.random.randint(1, 100, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(data)
    df.loc[10:15, 'feature_a'] = np.nan
    df.loc[5, 'feature_b'] = 1000
    
    cleaner = DataCleaner(df)
    print("Initial summary:", cleaner.summary())
    
    cleaner.impute_missing_mean(['feature_a'])
    cleaner.remove_outliers(['feature_b'])
    cleaner.standardize_data(['feature_a', 'feature_b'])
    
    clean_df = cleaner.get_clean_data()
    print("Cleaned shape:", clean_df.shape)
    print("Missing values after cleaning:", clean_df.isnull().sum().sum())
    
    return clean_df

if __name__ == "__main__":
    result = example_usage()
    print("Data cleaning completed successfully.")import pandas as pd
import numpy as np

def load_and_clean_csv(filepath, drop_na=True, fill_value=0):
    """
    Load a CSV file and perform basic cleaning operations.
    
    Args:
        filepath (str): Path to the CSV file.
        drop_na (bool): Whether to drop rows with missing values.
        fill_value: Value to fill missing data if drop_na is False.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at {filepath}")
    
    if drop_na:
        df = df.dropna()
    else:
        df = df.fillna(fill_value)
    
    return df

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        subset (list): Columns to consider for duplicates.
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    return df.drop_duplicates(subset=subset, keep='first')

def normalize_column(df, column_name):
    """
    Normalize a numeric column to range [0, 1].
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column_name (str): Name of column to normalize.
    
    Returns:
        pd.DataFrame: DataFrame with normalized column.
    """
    if column_name not in df.columns:
        raise KeyError(f"Column '{column_name}' not found in DataFrame")
    
    col = df[column_name]
    if not np.issubdtype(col.dtype, np.number):
        raise TypeError(f"Column '{column_name}' must be numeric")
    
    min_val = col.min()
    max_val = col.max()
    
    if max_val == min_val:
        df[column_name] = 0.5
    else:
        df[column_name] = (col - min_val) / (max_val - min_val)
    
    return df

def filter_by_quantile(df, column_name, lower=0.05, upper=0.95):
    """
    Filter DataFrame rows based on column quantile thresholds.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column_name (str): Column name for filtering.
        lower (float): Lower quantile threshold.
        upper (float): Upper quantile threshold.
    
    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    if column_name not in df.columns:
        raise KeyError(f"Column '{column_name}' not found in DataFrame")
    
    col = df[column_name]
    if not np.issubdtype(col.dtype, np.number):
        raise TypeError(f"Column '{column_name}' must be numeric")
    
    lower_bound = col.quantile(lower)
    upper_bound = col.quantile(upper)
    
    return df[(col >= lower_bound) & (col <= upper_bound)]
import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape

    def handle_missing_values(self, strategy='mean', columns=None):
        if columns is None:
            columns = self.df.columns

        for col in columns:
            if self.df[col].isnull().any():
                if strategy == 'mean':
                    fill_value = self.df[col].mean()
                elif strategy == 'median':
                    fill_value = self.df[col].median()
                elif strategy == 'mode':
                    fill_value = self.df[col].mode()[0]
                elif strategy == 'drop':
                    self.df = self.df.dropna(subset=[col])
                    continue
                else:
                    raise ValueError("Strategy must be 'mean', 'median', 'mode', or 'drop'")
                
                self.df[col] = self.df[col].fillna(fill_value)
        
        return self

    def remove_outliers_iqr(self, columns=None, factor=1.5):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns

        for col in columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        
        return self

    def standardize_data(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns

        for col in columns:
            mean = self.df[col].mean()
            std = self.df[col].std()
            self.df[col] = (self.df[col] - mean) / std
        
        return self

    def get_cleaned_data(self):
        return self.df

    def get_cleaning_report(self):
        removed_rows = self.original_shape[0] - self.df.shape[0]
        removed_cols = self.original_shape[1] - self.df.shape[1]
        
        report = {
            'original_shape': self.original_shape,
            'cleaned_shape': self.df.shape,
            'rows_removed': removed_rows,
            'columns_removed': removed_cols,
            'missing_values_remaining': self.df.isnull().sum().sum()
        }
        
        return report

def load_and_clean_data(filepath, cleaning_steps=None):
    df = pd.read_csv(filepath)
    cleaner = DataCleaner(df)
    
    if cleaning_steps:
        for step in cleaning_steps:
            if step['method'] == 'handle_missing':
                cleaner.handle_missing_values(**step.get('params', {}))
            elif step['method'] == 'remove_outliers':
                cleaner.remove_outliers_iqr(**step.get('params', {}))
            elif step['method'] == 'standardize':
                cleaner.standardize_data(**step.get('params', {}))
    
    return cleaner.get_cleaned_data(), cleaner.get_cleaning_report()