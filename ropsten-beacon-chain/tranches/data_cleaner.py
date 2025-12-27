import numpy as np
import pandas as pd

def remove_outliers_iqr(dataframe, column):
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]
    return filtered_df

def normalize_minmax(dataframe, column):
    min_val = dataframe[column].min()
    max_val = dataframe[column].max()
    if max_val == min_val:
        return dataframe[column].apply(lambda x: 0.0)
    normalized = (dataframe[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(dataframe, column):
    mean_val = dataframe[column].mean()
    std_val = dataframe[column].std()
    if std_val == 0:
        return dataframe[column].apply(lambda x: 0.0)
    standardized = (dataframe[column] - mean_val) / std_val
    return standardized

def handle_missing_mean(dataframe, column):
    mean_value = dataframe[column].mean()
    filled_column = dataframe[column].fillna(mean_value)
    return filled_column

def validate_dataframe(dataframe, required_columns):
    missing_columns = [col for col in required_columns if col not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    return True
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
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        self.df = df_clean
        removed_count = self.original_shape[0] - self.df.shape[0]
        print(f"Removed {removed_count} outliers")
        return self
        
    def normalize_data(self, columns=None, method='zscore'):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_normalized = self.df.copy()
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                if method == 'zscore':
                    df_normalized[col] = stats.zscore(self.df[col])
                elif method == 'minmax':
                    df_normalized[col] = (self.df[col] - self.df[col].min()) / (self.df[col].max() - self.df[col].min())
                elif method == 'robust':
                    median = self.df[col].median()
                    iqr = self.df[col].quantile(0.75) - self.df[col].quantile(0.25)
                    df_normalized[col] = (self.df[col] - median) / iqr
        
        self.df = df_normalized
        print(f"Normalized {len(columns)} columns using {method} method")
        return self
        
    def handle_missing_values(self, columns=None, strategy='mean'):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_filled = self.df.copy()
        for col in columns:
            if col in self.df.columns and self.df[col].isnull().any():
                if strategy == 'mean':
                    fill_value = self.df[col].mean()
                elif strategy == 'median':
                    fill_value = self.df[col].median()
                elif strategy == 'mode':
                    fill_value = self.df[col].mode()[0]
                elif strategy == 'drop':
                    df_filled = df_filled.dropna(subset=[col])
                    continue
                    
                df_filled[col] = self.df[col].fillna(fill_value)
                print(f"Filled missing values in {col} with {strategy}: {fill_value}")
        
        self.df = df_filled
        return self
        
    def get_cleaned_data(self):
        return self.df
        
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'cleaned_rows': self.df.shape[0],
            'original_columns': self.original_shape[1],
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
        'feature_c': np.random.uniform(0, 1, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(data)
    
    df.loc[10:15, 'feature_a'] = np.nan
    df.loc[95, 'feature_b'] = 1000
    df.loc[96, 'feature_a'] = 500
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    print("Original data shape:", sample_df.shape)
    
    cleaner = DataCleaner(sample_df)
    cleaned_df = (cleaner
                 .handle_missing_values(strategy='median')
                 .remove_outliers_iqr(factor=1.5)
                 .normalize_data(method='zscore')
                 .get_cleaned_data())
    
    print("\nCleaned data shape:", cleaned_df.shape)
    print("\nSummary:")
    for key, value in cleaner.get_summary().items():
        print(f"{key}: {value}")