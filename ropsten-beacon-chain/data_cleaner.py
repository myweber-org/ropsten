import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline.
    """
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col not in cleaned_df.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        elif outlier_method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
        
        if normalize_method == 'minmax':
            cleaned_df[f'{col}_normalized'] = normalize_minmax(cleaned_df, col)
        elif normalize_method == 'zscore':
            cleaned_df[f'{col}_standardized'] = normalize_zscore(cleaned_df, col)
    
    return cleaned_df

def validate_data(df, required_columns, numeric_threshold=0.8):
    """
    Validate dataset structure and quality.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) / len(df.columns) < numeric_threshold:
        print(f"Warning: Less than {numeric_threshold*100}% of columns are numeric")
    
    return {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'numeric_columns': len(numeric_cols),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
import pandas as pd
import numpy as np

def clean_dataset(df):
    """
    Clean a pandas DataFrame by removing duplicates,
    standardizing column names, and handling missing values.
    """
    # Remove duplicate rows
    df_clean = df.drop_duplicates()
    
    # Standardize column names: lowercase and replace spaces with underscores
    df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_')
    
    # Fill missing numeric values with column median
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # Fill missing categorical values with mode
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'unknown')
    
    return df_clean

def validate_data(df):
    """
    Validate that the cleaned DataFrame meets basic quality checks.
    """
    checks = {}
    
    # Check for remaining null values
    checks['has_nulls'] = df.isnull().any().any()
    
    # Check for duplicate rows
    checks['has_duplicates'] = df.duplicated().any()
    
    # Check column name format (should be lowercase with underscores)
    column_format_valid = all(col.islower() and ' ' not in col for col in df.columns)
    checks['valid_column_names'] = column_format_valid
    
    return checks

if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'Customer ID': [1, 2, 2, 3, 4],
        'Order Value': [100.0, 200.0, 200.0, None, 400.0],
        'Product Category': ['A', 'B', 'B', None, 'C'],
        'Region': ['North', 'South', 'South', 'East', 'West']
    })
    
    print("Original dataset:")
    print(sample_data)
    print("\nCleaning dataset...")
    
    cleaned_data = clean_dataset(sample_data)
    print("\nCleaned dataset:")
    print(cleaned_data)
    
    validation_results = validate_data(cleaned_data)
    print("\nData validation results:")
    for check, result in validation_results.items():
        print(f"{check}: {result}")
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
        
        removed_count = self.original_shape[0] - clean_df.shape[0]
        self.df = clean_df
        return removed_count
    
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
        return self.df
    
    def standardize_zscore(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        standardized_df = self.df.copy()
        for col in columns:
            if col in standardized_df.columns:
                col_mean = standardized_df[col].mean()
                col_std = standardized_df[col].std()
                if col_std > 0:
                    standardized_df[col] = (standardized_df[col] - col_mean) / col_std
        
        self.df = standardized_df
        return self.df
    
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        filled_df = self.df.copy()
        for col in columns:
            if col in filled_df.columns and filled_df[col].isnull().any():
                median_val = filled_df[col].median()
                filled_df[col] = filled_df[col].fillna(median_val)
        
        self.df = filled_df
        return self.df
    
    def get_cleaned_data(self):
        return self.df.copy()
    
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'current_rows': self.df.shape[0],
            'columns': self.df.shape[1],
            'missing_values': self.df.isnull().sum().sum(),
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns)
        }
        return summary

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.uniform(0, 1, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    df = pd.DataFrame(data)
    df.loc[np.random.choice(df.index, 50), 'feature_a'] = np.nan
    df.loc[np.random.choice(df.index, 20), 'feature_b'] = 1000
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    cleaner = DataCleaner(sample_df)
    
    print("Initial summary:")
    print(cleaner.get_summary())
    
    removed = cleaner.remove_outliers_iqr(['feature_a', 'feature_b'])
    print(f"\nRemoved {removed} outliers")
    
    cleaner.fill_missing_median()
    print("Filled missing values with median")
    
    cleaner.normalize_minmax(['feature_a', 'feature_b', 'feature_c'])
    print("Applied min-max normalization")
    
    print("\nFinal summary:")
    print(cleaner.get_summary())
    
    cleaned_data = cleaner.get_cleaned_data()
    print(f"\nCleaned data shape: {cleaned_data.shape}")