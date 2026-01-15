
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
                
                mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
                df_clean = df_clean[mask]
                
        self.df = df_clean
        removed_count = self.original_shape[0] - self.df.shape[0]
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
    
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_filled = self.df.copy()
        for col in columns:
            if col in df_filled.columns and df_filled[col].isnull().any():
                median_val = df_filled[col].median()
                df_filled[col] = df_filled[col].fillna(median_val)
                
        self.df = df_filled
        filled_count = self.df.isnull().sum().sum()
        return filled_count
    
    def get_clean_data(self):
        return self.df
    
    def get_cleaning_stats(self):
        stats_dict = {
            'original_rows': self.original_shape[0],
            'current_rows': self.df.shape[0],
            'original_columns': self.original_shape[1],
            'current_columns': self.df.shape[1],
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'missing_values': self.df.isnull().sum().sum()
        }
        return stats_dict

def process_dataset(filepath):
    try:
        df = pd.read_csv(filepath)
        cleaner = DataCleaner(df)
        
        print(f"Original dataset shape: {df.shape}")
        
        outliers_removed = cleaner.remove_outliers_iqr()
        print(f"Removed {outliers_removed} outliers using IQR method")
        
        missing_filled = cleaner.fill_missing_median()
        print(f"Filled {missing_filled} missing values with median")
        
        cleaner.normalize_minmax()
        print("Applied min-max normalization to numerical columns")
        
        stats = cleaner.get_cleaning_stats()
        print(f"Final dataset shape: {cleaner.df.shape}")
        print(f"Missing values remaining: {stats['missing_values']}")
        
        return cleaner.get_clean_data()
        
    except Exception as e:
        print(f"Error processing dataset: {e}")
        return None

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.randint(1, 100, 1000)
    })
    
    sample_data.iloc[10:20, 0] = np.nan
    sample_data.iloc[100:110, 1] = 1000
    
    cleaner = DataCleaner(sample_data)
    cleaned_data = process_dataset('sample_data.csv')
    
    if cleaned_data is not None:
        print("Data cleaning completed successfully")
        print(f"Cleaned data shape: {cleaned_data.shape}")
        print(f"Cleaned data summary:\n{cleaned_data.describe()}")