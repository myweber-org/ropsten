
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
        
        clean_df = self.df.copy()
        for col in columns:
            if col in clean_df.columns:
                Q1 = clean_df[col].quantile(0.25)
                Q3 = clean_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                clean_df = clean_df[(clean_df[col] >= lower_bound) & (clean_df[col] <= upper_bound)]
        
        self.df = clean_df
        return self
    
    def remove_outliers_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        clean_df = self.df.copy()
        for col in columns:
            if col in clean_df.columns:
                z_scores = np.abs(stats.zscore(clean_df[col].dropna()))
                clean_df = clean_df[(z_scores < threshold) | clean_df[col].isna()]
        
        self.df = clean_df
        return self
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        normalized_df = self.df.copy()
        for col in columns:
            if col in normalized_df.columns:
                col_min = normalized_df[col].min()
                col_max = normalized_df[col].max()
                if col_max != col_min:
                    normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)
        
        self.df = normalized_df
        return self
    
    def normalize_zscore(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        normalized_df = self.df.copy()
        for col in columns:
            if col in normalized_df.columns:
                col_mean = normalized_df[col].mean()
                col_std = normalized_df[col].std()
                if col_std > 0:
                    normalized_df[col] = (normalized_df[col] - col_mean) / col_std
        
        self.df = normalized_df
        return self
    
    def fill_missing_mean(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        filled_df = self.df.copy()
        for col in columns:
            if col in filled_df.columns:
                filled_df[col] = filled_df[col].fillna(filled_df[col].mean())
        
        self.df = filled_df
        return self
    
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        filled_df = self.df.copy()
        for col in columns:
            if col in filled_df.columns:
                filled_df[col] = filled_df[col].fillna(filled_df[col].median())
        
        self.df = filled_df
        return self
    
    def get_cleaned_data(self):
        return self.df
    
    def get_removed_count(self):
        return self.original_shape[0] - self.df.shape[0]
    
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'cleaned_rows': self.df.shape[0],
            'rows_removed': self.get_removed_count(),
            'removal_percentage': (self.get_removed_count() / self.original_shape[0]) * 100,
            'columns': list(self.df.columns)
        }
        return summary

def clean_dataset(df, method='iqr', normalization=None, fill_na='mean'):
    cleaner = DataCleaner(df)
    
    if method == 'iqr':
        cleaner.remove_outliers_iqr()
    elif method == 'zscore':
        cleaner.remove_outliers_zscore()
    
    if fill_na == 'mean':
        cleaner.fill_missing_mean()
    elif fill_na == 'median':
        cleaner.fill_missing_median()
    
    if normalization == 'minmax':
        cleaner.normalize_minmax()
    elif normalization == 'zscore':
        cleaner.normalize_zscore()
    
    return cleaner.get_cleaned_data(), cleaner.get_summary()
import numpy as np
import pandas as pd

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
    
    return filtered_df.copy()

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
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

def process_dataset(file_path, column_to_clean):
    """
    Main function to load dataset, clean outliers, and return processed data.
    
    Parameters:
    file_path (str): Path to CSV file
    column_to_clean (str): Column name to remove outliers from
    
    Returns:
    tuple: (cleaned_df, original_stats, cleaned_stats)
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    
    original_stats = calculate_summary_statistics(df, column_to_clean)
    cleaned_df = remove_outliers_iqr(df, column_to_clean)
    cleaned_stats = calculate_summary_statistics(cleaned_df, column_to_clean)
    
    return cleaned_df, original_stats, cleaned_stats

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'values': np.concatenate([
            np.random.normal(100, 10, 90),
            np.random.normal(300, 10, 10)
        ])
    })
    
    cleaned_data, original, cleaned = process_dataset(sample_data, 'values')
    
    print(f"Original data shape: {sample_data.shape}")
    print(f"Cleaned data shape: {cleaned_data.shape}")
    print(f"Outliers removed: {len(sample_data) - len(cleaned_data)}")
    print(f"Original mean: {original['mean']:.2f}")
    print(f"Cleaned mean: {cleaned['mean']:.2f}")