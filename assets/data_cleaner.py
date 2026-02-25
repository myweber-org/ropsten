import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    def impute_missing_values(self, strategy='median', fill_value=None):
        df_clean = self.df.copy()
        
        for col in self.numeric_columns:
            if df_clean[col].isnull().any():
                if strategy == 'mean':
                    df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                elif strategy == 'median':
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                elif strategy == 'mode':
                    df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
                elif strategy == 'constant' and fill_value is not None:
                    df_clean[col].fillna(fill_value, inplace=True)
        
        for col in self.categorical_columns:
            if df_clean[col].isnull().any():
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        
        return df_clean
    
    def remove_outliers_iqr(self, columns=None, multiplier=1.5):
        if columns is None:
            columns = self.numeric_columns
        
        df_clean = self.df.copy()
        
        for col in columns:
            if col in self.numeric_columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                
                mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
                df_clean = df_clean[mask]
        
        return df_clean.reset_index(drop=True)
    
    def remove_outliers_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.numeric_columns
        
        df_clean = self.df.copy()
        
        for col in columns:
            if col in self.numeric_columns:
                z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
                mask = z_scores < threshold
                valid_indices = df_clean[col].dropna().index[mask]
                df_clean = df_clean.loc[valid_indices.union(df_clean[col].index[df_clean[col].isna()])]
        
        return df_clean.reset_index(drop=True)
    
    def standardize_data(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        df_clean = self.df.copy()
        
        for col in columns:
            if col in self.numeric_columns:
                mean = df_clean[col].mean()
                std = df_clean[col].std()
                if std > 0:
                    df_clean[col] = (df_clean[col] - mean) / std
        
        return df_clean
    
    def normalize_data(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        df_clean = self.df.copy()
        
        for col in columns:
            if col in self.numeric_columns:
                min_val = df_clean[col].min()
                max_val = df_clean[col].max()
                if max_val > min_val:
                    df_clean[col] = (df_clean[col] - min_val) / (max_val - min_val)
        
        return df_clean
    
    def get_summary(self):
        summary = {
            'original_shape': self.df.shape,
            'numeric_columns': self.numeric_columns,
            'categorical_columns': self.categorical_columns,
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.to_dict()
        }
        return summary

def create_sample_data():
    np.random.seed(42)
    data = {
        'age': np.random.normal(35, 10, 100),
        'salary': np.random.normal(50000, 15000, 100),
        'department': np.random.choice(['Sales', 'Engineering', 'Marketing', 'HR'], 100),
        'experience': np.random.randint(1, 20, 100)
    }
    
    df = pd.DataFrame(data)
    
    df.loc[np.random.choice(100, 10), 'age'] = np.nan
    df.loc[np.random.choice(100, 5), 'salary'] = np.nan
    df.loc[np.random.choice(100, 3), 'department'] = np.nan
    
    df.loc[0, 'salary'] = 200000
    df.loc[1, 'age'] = 120
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    cleaner = DataCleaner(sample_df)
    
    print("Data Summary:")
    summary = cleaner.get_summary()
    print(f"Original shape: {summary['original_shape']}")
    print(f"Missing values: {summary['missing_values']}")
    
    cleaned_df = cleaner.impute_missing_values(strategy='median')
    cleaned_df = cleaner.remove_outliers_iqr(multiplier=1.5)
    
    print(f"\nCleaned shape: {cleaned_df.shape}")
    print("\nFirst 5 rows of cleaned data:")
    print(cleaned_df.head())