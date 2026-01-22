import numpy as np
import pandas as pd

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    def remove_outliers_iqr(self, columns=None, multiplier=1.5):
        if columns is None:
            columns = self.numeric_cols
        
        clean_df = self.df.copy()
        for col in columns:
            if col in self.numeric_cols:
                Q1 = clean_df[col].quantile(0.25)
                Q3 = clean_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                clean_df = clean_df[(clean_df[col] >= lower_bound) & (clean_df[col] <= upper_bound)]
        return clean_df
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.numeric_cols
        
        normalized_df = self.df.copy()
        for col in columns:
            if col in self.numeric_cols:
                min_val = normalized_df[col].min()
                max_val = normalized_df[col].max()
                if max_val > min_val:
                    normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
        return normalized_df
    
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.numeric_cols
        
        filled_df = self.df.copy()
        for col in columns:
            if col in self.numeric_cols:
                filled_df[col] = filled_df[col].fillna(filled_df[col].median())
        return filled_df
    
    def get_summary(self):
        summary = {
            'original_rows': len(self.df),
            'original_columns': len(self.df.columns),
            'numeric_columns': len(self.numeric_cols),
            'missing_values': self.df[self.numeric_cols].isnull().sum().sum(),
            'data_types': self.df.dtypes.value_counts().to_dict()
        }
        return summary

def example_usage():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    df = pd.DataFrame(data)
    df.loc[np.random.choice(100, 5), 'feature_a'] = np.nan
    
    cleaner = DataCleaner(df)
    print("Data Summary:", cleaner.get_summary())
    
    cleaned = cleaner.remove_outliers_iqr(['feature_a', 'feature_b'])
    normalized = cleaner.normalize_minmax()
    filled = cleaner.fill_missing_median()
    
    return cleaned, normalized, filled

if __name__ == "__main__":
    cleaned_df, normalized_df, filled_df = example_usage()
    print(f"Cleaned shape: {cleaned_df.shape}")
    print(f"Normalized range: [{normalized_df['feature_a'].min():.3f}, {normalized_df['feature_a'].max():.3f}]")
    print(f"Missing values after fill: {filled_df.isnull().sum().sum()}")