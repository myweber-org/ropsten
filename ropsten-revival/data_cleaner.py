
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
            if col in self.df.columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                mask = (self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)
                df_clean = df_clean[mask]
        
        removed_count = len(self.df) - len(df_clean)
        self.df = df_clean.reset_index(drop=True)
        return removed_count
    
    def normalize_data(self, columns=None, method='zscore'):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_normalized = self.df.copy()
        
        for col in columns:
            if col in self.df.columns:
                if method == 'zscore':
                    df_normalized[col] = stats.zscore(self.df[col])
                elif method == 'minmax':
                    min_val = self.df[col].min()
                    max_val = self.df[col].max()
                    if max_val != min_val:
                        df_normalized[col] = (self.df[col] - min_val) / (max_val - min_val)
                elif method == 'robust':
                    median = self.df[col].median()
                    iqr = self.df[col].quantile(0.75) - self.df[col].quantile(0.25)
                    if iqr != 0:
                        df_normalized[col] = (self.df[col] - median) / iqr
        
        self.df = df_normalized
        return self.df
    
    def handle_missing_values(self, strategy='mean', fill_value=None):
        df_filled = self.df.copy()
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(exclude=[np.number]).columns
        
        for col in numeric_cols:
            if self.df[col].isnull().any():
                if strategy == 'mean':
                    fill_val = self.df[col].mean() if fill_value is None else fill_value
                elif strategy == 'median':
                    fill_val = self.df[col].median() if fill_value is None else fill_value
                elif strategy == 'mode':
                    fill_val = self.df[col].mode()[0] if fill_value is None else fill_value
                else:
                    fill_val = fill_value
                
                df_filled[col] = self.df[col].fillna(fill_val)
        
        for col in categorical_cols:
            if self.df[col].isnull().any():
                fill_val = self.df[col].mode()[0] if fill_value is None else fill_value
                df_filled[col] = self.df[col].fillna(fill_val)
        
        self.df = df_filled
        return self.df
    
    def get_cleaned_data(self):
        return self.df
    
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_rows': self.df.shape[0],
            'cleaned_columns': self.df.shape[1],
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'missing_values': self.df.isnull().sum().sum()
        }
        return summary

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'feature_c': np.random.randint(1, 100, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(data)
    
    df.loc[10:15, 'feature_a'] = np.nan
    df.loc[20:25, 'feature_b'] = np.nan
    
    outliers = np.random.randint(0, 100, 5)
    df.loc[outliers, 'feature_a'] = df['feature_a'].mean() + 5 * df['feature_a'].std()
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    cleaner = DataCleaner(sample_df)
    
    print("Original data shape:", cleaner.original_shape)
    print("Missing values before:", sample_df.isnull().sum().sum())
    
    cleaner.handle_missing_values(strategy='mean')
    removed = cleaner.remove_outliers_iqr(threshold=1.5)
    normalized_df = cleaner.normalize_data(method='zscore')
    
    summary = cleaner.get_summary()
    print("\nCleaning Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print("\nFirst 5 rows of cleaned data:")
    print(cleaner.get_cleaned_data().head())