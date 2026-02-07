
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column in a dataset using the IQR method.
    
    Parameters:
    data (numpy.ndarray): The dataset.
    column (int): Index of the column to clean.
    
    Returns:
    numpy.ndarray: Dataset with outliers removed from the specified column.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    
    if column >= data.shape[1]:
        raise IndexError("Column index out of bounds")
    
    col_data = data[:, column]
    
    Q1 = np.percentile(col_data, 25)
    Q3 = np.percentile(col_data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    mask = (col_data >= lower_bound) & (col_data <= upper_bound)
    
    return data[mask]
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
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
                df_clean = df_clean[mask]
        
        self.df = df_clean
        return self
    
    def remove_outliers_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_clean = self.df.copy()
        for col in columns:
            if col in df_clean.columns:
                z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
                mask = z_scores < threshold
                df_clean = df_clean[mask]
        
        self.df = df_clean
        return self
    
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
        return self
    
    def normalize_zscore(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_normalized = self.df.copy()
        for col in columns:
            if col in df_normalized.columns:
                mean_val = df_normalized[col].mean()
                std_val = df_normalized[col].std()
                if std_val > 0:
                    df_normalized[col] = (df_normalized[col] - mean_val) / std_val
        
        self.df = df_normalized
        return self
    
    def fill_missing_mean(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_filled = self.df.copy()
        for col in columns:
            if col in df_filled.columns:
                mean_val = df_filled[col].mean()
                df_filled[col] = df_filled[col].fillna(mean_val)
        
        self.df = df_filled
        return self
    
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_filled = self.df.copy()
        for col in columns:
            if col in df_filled.columns:
                median_val = df_filled[col].median()
                df_filled[col] = df_filled[col].fillna(median_val)
        
        self.df = df_filled
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

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'feature3': np.random.uniform(0, 1, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    df = pd.DataFrame(data)
    
    indices = np.random.choice(df.index, size=50, replace=False)
    df.loc[indices, 'feature1'] = np.random.normal(300, 50, 50)
    
    indices = np.random.choice(df.index, size=30, replace=False)
    df.loc[indices, 'feature2'] = np.random.normal(500, 100, 30)
    
    indices = np.random.choice(df.index, size=20, replace=False)
    df.loc[indices, 'feature1'] = np.nan
    df.loc[indices, 'feature2'] = np.nan
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    print("Original data shape:", sample_df.shape)
    print("\nMissing values:")
    print(sample_df.isnull().sum())
    
    cleaner = DataCleaner(sample_df)
    
    cleaner.remove_outliers_iqr(['feature1', 'feature2'])
    cleaner.fill_missing_mean(['feature1', 'feature2'])
    cleaner.normalize_minmax(['feature1', 'feature2', 'feature3'])
    
    cleaned_df = cleaner.get_cleaned_data()
    print("\nCleaned data shape:", cleaned_df.shape)
    
    summary = cleaner.get_summary()
    print("\nCleaning summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def detect_outliers_iqr(self, column):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
        return outliers
    
    def remove_outliers_zscore(self, columns, threshold=3):
        df_clean = self.df.copy()
        for col in columns:
            if self.df[col].dtype in ['float64', 'int64']:
                z_scores = np.abs(stats.zscore(self.df[col].fillna(self.df[col].mean())))
                df_clean = df_clean[z_scores < threshold]
        self.df = df_clean
        return self
    
    def normalize_column(self, column, method='minmax'):
        if method == 'minmax':
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            self.df[f'{column}_normalized'] = (self.df[column] - min_val) / (max_val - min_val)
        elif method == 'zscore':
            mean_val = self.df[column].mean()
            std_val = self.df[column].std()
            self.df[f'{column}_standardized'] = (self.df[column] - mean_val) / std_val
        return self
    
    def fill_missing(self, strategy='mean'):
        for col in self.df.columns:
            if self.df[col].isnull().any():
                if strategy == 'mean' and self.df[col].dtype in ['float64', 'int64']:
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                elif strategy == 'median' and self.df[col].dtype in ['float64', 'int64']:
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                elif strategy == 'mode':
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        return self
    
    def get_cleaned_data(self):
        print(f"Original shape: {self.original_shape}")
        print(f"Cleaned shape: {self.df.shape}")
        print(f"Rows removed: {self.original_shape[0] - self.df.shape[0]}")
        return self.df

def example_usage():
    np.random.seed(42)
    data = {
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    data['feature1'][np.random.choice(1000, 10)] = np.nan
    data['feature2'][np.random.choice(1000, 5)] = 1000
    
    df = pd.DataFrame(data)
    cleaner = DataCleaner(df)
    
    cleaned_df = (cleaner
                 .fill_missing(strategy='mean')
                 .remove_outliers_zscore(['feature1', 'feature2'])
                 .normalize_column('feature1', method='minmax')
                 .normalize_column('feature2', method='zscore')
                 .get_cleaned_data())
    
    return cleaned_df

if __name__ == "__main__":
    result = example_usage()
    print(result.head())
import pandas as pd
import numpy as np
from pathlib import Path

def clean_csv_data(input_path, output_path=None, strategy='mean'):
    """
    Clean CSV data by handling missing values.
    
    Parameters:
    input_path (str): Path to input CSV file
    output_path (str, optional): Path for cleaned output CSV
    strategy (str): Method for handling missing values ('mean', 'median', 'drop')
    
    Returns:
    pandas.DataFrame: Cleaned DataFrame
    """
    
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    df = pd.read_csv(input_path)
    
    print(f"Original data shape: {df.shape}")
    print(f"Missing values per column:\n{df.isnull().sum()}")
    
    if strategy == 'drop':
        df_cleaned = df.dropna()
    elif strategy == 'mean':
        df_cleaned = df.fillna(df.mean(numeric_only=True))
    elif strategy == 'median':
        df_cleaned = df.fillna(df.median(numeric_only=True))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    print(f"Cleaned data shape: {df_cleaned.shape}")
    
    if output_path:
        df_cleaned.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
    
    return df_cleaned

def detect_outliers_iqr(df, column):
    """
    Detect outliers using IQR method.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame
    column (str): Column name to check for outliers
    
    Returns:
    tuple: (lower_bound, upper_bound, outlier_indices)
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if not np.issubdtype(df[column].dtype, np.number):
        raise TypeError(f"Column '{column}' must be numeric")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    return lower_bound, upper_bound, outliers.index.tolist()

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 5, 100],
        'B': [10, np.nan, 30, 40, 50, 60],
        'C': [100, 200, 300, np.nan, 500, 600]
    }
    
    df_sample = pd.DataFrame(sample_data)
    df_sample.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data('sample_data.csv', 'cleaned_sample.csv', strategy='median')
    
    for col in ['A', 'B', 'C']:
        lower, upper, outliers = detect_outliers_iqr(cleaned_df, col)
        print(f"Column {col}: outliers at indices {outliers}")
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result