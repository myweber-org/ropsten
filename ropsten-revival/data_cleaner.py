
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns
        
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
                
        return df_clean
    
    def remove_outliers_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.numeric_columns
            
        df_clean = self.df.copy()
        
        for col in columns:
            if col in self.numeric_columns:
                z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
                mask = z_scores < threshold
                df_clean = df_clean[mask]
                
        return df_clean
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
            
        df_normalized = self.df.copy()
        
        for col in columns:
            if col in self.numeric_columns:
                min_val = df_normalized[col].min()
                max_val = df_normalized[col].max()
                if max_val != min_val:
                    df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
                    
        return df_normalized
    
    def normalize_zscore(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
            
        df_normalized = self.df.copy()
        
        for col in columns:
            if col in self.numeric_columns:
                mean_val = df_normalized[col].mean()
                std_val = df_normalized[col].std()
                if std_val > 0:
                    df_normalized[col] = (df_normalized[col] - mean_val) / std_val
                    
        return df_normalized
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if columns is None:
            columns = self.numeric_columns
            
        df_filled = self.df.copy()
        
        for col in columns:
            if col in self.numeric_columns and df_filled[col].isnull().any():
                if strategy == 'mean':
                    fill_value = df_filled[col].mean()
                elif strategy == 'median':
                    fill_value = df_filled[col].median()
                elif strategy == 'mode':
                    fill_value = df_filled[col].mode()[0]
                elif strategy == 'zero':
                    fill_value = 0
                else:
                    continue
                    
                df_filled[col] = df_filled[col].fillna(fill_value)
                
        return df_filled
    
    def get_summary(self):
        summary = {
            'original_shape': self.df.shape,
            'numeric_columns': list(self.numeric_columns),
            'missing_values': self.df[self.numeric_columns].isnull().sum().to_dict(),
            'data_types': self.df.dtypes.to_dict()
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
    
    data['feature_a'][[10, 20, 30]] = [500, -200, 300]
    data['feature_b'][[15, 25, 35]] = [np.nan, np.nan, 1000]
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = create_sample_data()
    cleaner = DataCleaner(df)
    
    print("Original data shape:", cleaner.df.shape)
    print("\nMissing values:")
    print(cleaner.df.isnull().sum())
    
    df_clean = cleaner.remove_outliers_iqr(['feature_a', 'feature_b'])
    print("\nAfter IQR outlier removal:", df_clean.shape)
    
    df_normalized = cleaner.normalize_minmax()
    print("\nAfter min-max normalization:")
    print(df_normalized[['feature_a', 'feature_b', 'feature_c']].describe())
    
    summary = cleaner.get_summary()
    print("\nData summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Args:
        data (list or np.array): Input data array
        column (int): Column index to process (for 2D arrays)
        
    Returns:
        np.array: Data with outliers removed
    """
    if isinstance(data, list):
        data = np.array(data)
    
    # Handle 2D arrays by selecting specific column
    if data.ndim == 2:
        column_data = data[:, column]
    else:
        column_data = data
    
    # Calculate IQR
    q1 = np.percentile(column_data, 25)
    q3 = np.percentile(column_data, 75)
    iqr = q3 - q1
    
    # Define outlier bounds
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Filter data
    if data.ndim == 2:
        mask = (data[:, column] >= lower_bound) & (data[:, column] <= upper_bound)
        return data[mask]
    else:
        mask = (data >= lower_bound) & (data <= upper_bound)
        return data[mask]

def calculate_statistics(data):
    """
    Calculate basic statistics for the data.
    
    Args:
        data (np.array): Input data array
        
    Returns:
        dict: Dictionary containing mean, median, std, min, max
    """
    stats = {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'count': len(data)
    }
    return stats

def normalize_data(data, method='minmax'):
    """
    Normalize data using specified method.
    
    Args:
        data (np.array): Input data array
        method (str): Normalization method ('minmax' or 'zscore')
        
    Returns:
        np.array: Normalized data
    """
    if method == 'minmax':
        data_min = np.min(data)
        data_max = np.max(data)
        if data_max - data_min == 0:
            return np.zeros_like(data)
        return (data - data_min) / (data_max - data_min)
    
    elif method == 'zscore':
        data_mean = np.mean(data)
        data_std = np.std(data)
        if data_std == 0:
            return np.zeros_like(data)
        return (data - data_mean) / data_std
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")

# Example usage
if __name__ == "__main__":
    # Create sample data with outliers
    np.random.seed(42)
    clean_data = np.random.normal(100, 15, 90)
    outlier_data = np.random.normal(200, 30, 10)
    sample_data = np.concatenate([clean_data, outlier_data])
    
    print("Original data statistics:")
    original_stats = calculate_statistics(sample_data)
    for key, value in original_stats.items():
        print(f"{key}: {value:.2f}")
    
    # Remove outliers
    cleaned_data = remove_outliers_iqr(sample_data, column=0)
    
    print("\nCleaned data statistics:")
    cleaned_stats = calculate_statistics(cleaned_data)
    for key, value in cleaned_stats.items():
        print(f"{key}: {value:.2f}")
    
    # Normalize cleaned data
    normalized_data = normalize_data(cleaned_data, method='minmax')
    
    print(f"\nOriginal data points: {len(sample_data)}")
    print(f"Cleaned data points: {len(cleaned_data)}")
    print(f"Outliers removed: {len(sample_data) - len(cleaned_data)}")