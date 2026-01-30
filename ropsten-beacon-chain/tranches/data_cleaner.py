
import pandas as pd
import numpy as np
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
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                clean_df = clean_df[(clean_df[col] >= lower_bound) & (clean_df[col] <= upper_bound)]

        self.df = clean_df
        removed_count = self.original_shape[0] - self.df.shape[0]
        print(f"Removed {removed_count} outliers using IQR method")
        return self

    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns

        for col in columns:
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
                else:
                    self.df[col] = 0

        print("Applied Min-Max normalization")
        return self

    def handle_missing_mean(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns

        for col in columns:
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                mean_val = self.df[col].mean()
                self.df[col].fillna(mean_val, inplace=True)

        print("Filled missing values with column mean")
        return self

    def get_cleaned_data(self):
        return self.df

    def summary(self):
        print(f"Original shape: {self.original_shape}")
        print(f"Cleaned shape: {self.df.shape}")
        print(f"Removed rows: {self.original_shape[0] - self.df.shape[0]}")
        print(f"Removed columns: {self.original_shape[1] - self.df.shape[1]}")
        print("\nColumn dtypes:")
        print(self.df.dtypes)
        print("\nMissing values:")
        print(self.df.isnull().sum())

def example_usage():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'feature_c': np.random.randint(1, 100, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(data)
    df.iloc[5, 0] = np.nan
    df.iloc[10, 1] = np.nan
    df.iloc[15, 2] = 500
    
    cleaner = DataCleaner(df)
    cleaned_df = (cleaner
                  .handle_missing_mean()
                  .remove_outliers_iqr(factor=1.5)
                  .normalize_minmax()
                  .get_cleaned_data())
    
    cleaner.summary()
    return cleaned_df

if __name__ == "__main__":
    result = example_usage()
    print("\nFirst 5 rows of cleaned data:")
    print(result.head())import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
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
    
    return filtered_df.reset_index(drop=True)

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a DataFrame column.
    
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
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    If no columns specified, clean all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    cleaned_df = df.copy()
    
    for column in columns:
        if column in cleaned_df.columns:
            if pd.api.types.is_numeric_dtype(cleaned_df[column]):
                original_count = len(cleaned_df)
                cleaned_df = remove_outliers_iqr(cleaned_df, column)
                removed_count = original_count - len(cleaned_df)
                print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_df

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame shape:", df.shape)
    
    # Add some outliers
    df.loc[1000] = [500, 1000, 300]
    df.loc[1001] = [-100, 2000, 400]
    
    cleaned_df = clean_numeric_data(df)
    print("Cleaned DataFrame shape:", cleaned_df.shape)
    
    for col in df.columns:
        stats = calculate_summary_statistics(df, col)
        print(f"\nStatistics for column '{col}':")
        for key, value in stats.items():
            print(f"  {key}: {value:.2f}")
import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        subset (list, optional): Columns to consider for duplicates
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        strategy (str): 'mean', 'median', 'mode', or 'drop'
        columns (list): Specific columns to process
    
    Returns:
        pd.DataFrame: DataFrame with handled missing values
    """
    if columns is None:
        columns = df.columns
    
    df_copy = df.copy()
    
    for col in columns:
        if col in df_copy.columns:
            if strategy == 'mean':
                df_copy[col].fillna(df_copy[col].mean(), inplace=True)
            elif strategy == 'median':
                df_copy[col].fillna(df_copy[col].median(), inplace=True)
            elif strategy == 'mode':
                df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
            elif strategy == 'drop':
                df_copy = df_copy.dropna(subset=[col])
    
    return df_copy

def normalize_column(df, column, method='minmax'):
    """
    Normalize a column using specified method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to normalize
        method (str): 'minmax' or 'zscore'
    
    Returns:
        pd.DataFrame: DataFrame with normalized column
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_copy = df.copy()
    
    if method == 'minmax':
        min_val = df_copy[column].min()
        max_val = df_copy[column].max()
        if max_val != min_val:
            df_copy[column] = (df_copy[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df_copy[column].mean()
        std_val = df_copy[column].std()
        if std_val > 0:
            df_copy[column] = (df_copy[column] - mean_val) / std_val
    
    return df_copy

def filter_outliers(df, column, method='iqr', threshold=1.5):
    """
    Filter outliers from a DataFrame column.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to filter
        method (str): 'iqr' or 'zscore'
        threshold (float): Threshold for outlier detection
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_copy = df.copy()
    
    if method == 'iqr':
        Q1 = df_copy[column].quantile(0.25)
        Q3 = df_copy[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        df_copy = df_copy[(df_copy[column] >= lower_bound) & (df_copy[column] <= upper_bound)]
    
    elif method == 'zscore':
        z_scores = np.abs((df_copy[column] - df_copy[column].mean()) / df_copy[column].std())
        df_copy = df_copy[z_scores < threshold]
    
    return df_copy

def clean_dataframe(df, config):
    """
    Apply multiple cleaning operations based on configuration.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        config (dict): Configuration dictionary with cleaning steps
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if config.get('remove_duplicates'):
        subset = config.get('duplicate_subset')
        df_clean = remove_duplicates(df_clean, subset)
    
    if config.get('handle_missing'):
        strategy = config.get('missing_strategy', 'mean')
        columns = config.get('missing_columns')
        df_clean = handle_missing_values(df_clean, strategy, columns)
    
    if config.get('normalize'):
        normalize_config = config.get('normalize_config', {})
        for col, method in normalize_config.items():
            if col in df_clean.columns:
                df_clean = normalize_column(df_clean, col, method)
    
    if config.get('filter_outliers'):
        outlier_config = config.get('outlier_config', {})
        for col, params in outlier_config.items():
            if col in df_clean.columns:
                method = params.get('method', 'iqr')
                threshold = params.get('threshold', 1.5)
                df_clean = filter_outliers(df_clean, col, method, threshold)
    
    return df_clean