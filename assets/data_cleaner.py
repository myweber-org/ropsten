
import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    def remove_outliers_zscore(self, threshold=3):
        df_clean = self.df.copy()
        for col in self.numeric_columns:
            z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
            df_clean = df_clean[(z_scores < threshold) | df_clean[col].isna()]
        return df_clean
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        df_normalized = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                col_min = df_normalized[col].min()
                col_max = df_normalized[col].max()
                if col_max != col_min:
                    df_normalized[col] = (df_normalized[col] - col_min) / (col_max - col_min)
        return df_normalized
    
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        df_filled = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                median_val = df_filled[col].median()
                df_filled[col] = df_filled[col].fillna(median_val)
        return df_filled
    
    def get_summary(self):
        summary = {
            'original_shape': self.df.shape,
            'missing_values': self.df.isnull().sum().to_dict(),
            'numeric_columns': self.numeric_columns,
            'data_types': self.df.dtypes.to_dict()
        }
        return summary

def example_usage():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    data['feature_a'][[10, 25, 50]] = 500
    data['feature_b'][[30, 75]] = np.nan
    
    df = pd.DataFrame(data)
    cleaner = DataCleaner(df)
    
    print("Original data summary:")
    print(cleaner.get_summary())
    
    df_no_outliers = cleaner.remove_outliers_zscore()
    print(f"\nAfter outlier removal: {df_no_outliers.shape}")
    
    df_filled = cleaner.fill_missing_median()
    df_normalized = cleaner.normalize_minmax()
    
    return df, df_no_outliers, df_normalized

if __name__ == "__main__":
    example_usage()