import pandas as pd
import re

def clean_dataframe(df, text_columns=None):
    """
    Remove duplicate rows and standardize text in specified columns.
    """
    # Remove duplicates
    df_clean = df.drop_duplicates().reset_index(drop=True)
    
    if text_columns:
        for col in text_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].apply(_standardize_text)
    
    return df_clean

def _standardize_text(text):
    """
    Standardize text: lowercase, remove extra spaces, and strip.
    """
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
    return text

def validate_email(email):
    """
    Validate email format using regex.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if isinstance(email, str):
        return bool(re.match(pattern, email))
    return False
import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        self.categorical_columns = self.df.select_dtypes(exclude=[np.number]).columns

    def handle_missing_values(self, strategy='mean', fill_value=None):
        if strategy == 'mean':
            for col in self.numeric_columns:
                self.df[col].fillna(self.df[col].mean(), inplace=True)
        elif strategy == 'median':
            for col in self.numeric_columns:
                self.df[col].fillna(self.df[col].median(), inplace=True)
        elif strategy == 'mode':
            for col in self.numeric_columns:
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        elif strategy == 'constant':
            if fill_value is not None:
                self.df.fillna(fill_value, inplace=True)
            else:
                raise ValueError("fill_value must be provided for constant strategy")
        else:
            raise ValueError("Invalid strategy. Choose from 'mean', 'median', 'mode', or 'constant'")
        
        for col in self.categorical_columns:
            self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        
        return self.df

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
                
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        self.df = df_clean
        return self.df

    def standardize_data(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        df_standardized = self.df.copy()
        
        for col in columns:
            if col in self.numeric_columns:
                mean = df_standardized[col].mean()
                std = df_standardized[col].std()
                if std > 0:
                    df_standardized[col] = (df_standardized[col] - mean) / std
        
        self.df = df_standardized
        return self.df

    def normalize_data(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        df_normalized = self.df.copy()
        
        for col in columns:
            if col in self.numeric_columns:
                min_val = df_normalized[col].min()
                max_val = df_normalized[col].max()
                if max_val > min_val:
                    df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
        
        self.df = df_normalized
        return self.df

    def get_cleaned_data(self):
        return self.df.copy()

def example_usage():
    data = {
        'age': [25, 30, np.nan, 35, 40, 200, 45],
        'salary': [50000, 60000, 70000, np.nan, 90000, 1000000, 55000],
        'department': ['IT', 'HR', 'IT', 'HR', 'IT', 'IT', np.nan]
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaner = DataCleaner(df)
    
    cleaned_df = cleaner.handle_missing_values(strategy='mean')
    print("After handling missing values:")
    print(cleaned_df)
    print("\n")
    
    cleaned_df = cleaner.remove_outliers_iqr(multiplier=1.5)
    print("After removing outliers:")
    print(cleaned_df)
    print("\n")
    
    cleaned_df = cleaner.standardize_data()
    print("After standardization:")
    print(cleaned_df)
    
    return cleaner.get_cleaned_data()

if __name__ == "__main__":
    result = example_usage()
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
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
    
    return filtered_df.copy()

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
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
        'count': df[column].count()
    }
    
    return stats

def clean_dataset(df, columns_to_clean=None):
    """
    Clean multiple columns in a DataFrame by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns_to_clean (list): List of column names to clean. If None, clean all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns_to_clean is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns_to_clean = numeric_cols
    
    cleaned_df = df.copy()
    
    for column in columns_to_clean:
        if column in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[column]):
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_df
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
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
    Calculate summary statistics for a column after outlier removal.
    
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
        'count': len(df[column])
    }
    
    return stats

def clean_dataset(df, columns_to_clean=None):
    """
    Clean multiple columns in a DataFrame by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns_to_clean (list): List of column names to clean. If None, clean all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    dict: Dictionary of summary statistics for each cleaned column
    """
    if columns_to_clean is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns_to_clean = list(numeric_cols)
    
    cleaned_df = df.copy()
    summary_stats = {}
    
    for column in columns_to_clean:
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            
            stats = calculate_summary_statistics(cleaned_df, column)
            stats['outliers_removed'] = removed_count
            summary_stats[column] = stats
    
    return cleaned_df, summary_stats

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[::100, 'A'] = 500
    
    print("Original DataFrame shape:", df.shape)
    print("\nOriginal summary statistics:")
    print(df.describe())
    
    cleaned_df, stats = clean_dataset(df, ['A', 'B'])
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("\nCleaned summary statistics:")
    print(cleaned_df.describe())
    
    print("\nDetailed statistics:")
    for col, col_stats in stats.items():
        print(f"\n{col}:")
        for stat_name, stat_value in col_stats.items():
            print(f"  {stat_name}: {stat_value:.2f}")
import numpy as np
import pandas as pd

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers in a specified column using the Interquartile Range method.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to analyze
    threshold (float): Multiplier for IQR (default 1.5)
    
    Returns:
    pd.DataFrame: Dataframe with outlier flags
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    data = data.copy()
    data[f'{column}_outlier'] = (data[column] < lower_bound) | (data[column] > upper_bound)
    data[f'{column}_lower_bound'] = lower_bound
    data[f'{column}_upper_bound'] = upper_bound
    
    return data

def remove_outliers(data, column, threshold=1.5):
    """
    Remove outliers from a specified column using IQR method.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to clean
    threshold (float): Multiplier for IQR (default 1.5)
    
    Returns:
    pd.DataFrame: Cleaned dataframe without outliers
    """
    data_with_flags = detect_outliers_iqr(data, column, threshold)
    cleaned_data = data_with_flags[~data_with_flags[f'{column}_outlier']].copy()
    cleaned_data = cleaned_data.drop(columns=[f'{column}_outlier', 
                                              f'{column}_lower_bound', 
                                              f'{column}_upper_bound'])
    return cleaned_data

def summarize_outliers(data, column, threshold=1.5):
    """
    Generate summary statistics for outliers in a column.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to analyze
    threshold (float): Multiplier for IQR (default 1.5)
    
    Returns:
    dict: Summary statistics
    """
    data_with_flags = detect_outliers_iqr(data, column, threshold)
    
    outliers = data_with_flags[data_with_flags[f'{column}_outlier']]
    non_outliers = data_with_flags[~data_with_flags[f'{column}_outlier']]
    
    summary = {
        'total_records': len(data),
        'outlier_count': len(outliers),
        'outlier_percentage': (len(outliers) / len(data)) * 100,
        'min_outlier_value': outliers[column].min() if len(outliers) > 0 else np.nan,
        'max_outlier_value': outliers[column].max() if len(outliers) > 0 else np.nan,
        'mean_with_outliers': data[column].mean(),
        'mean_without_outliers': non_outliers[column].mean(),
        'std_with_outliers': data[column].std(),
        'std_without_outliers': non_outliers[column].std()
    }
    
    return summary

def process_multiple_columns(data, columns, threshold=1.5, remove=False):
    """
    Process multiple columns for outlier detection or removal.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    columns (list): List of column names to process
    threshold (float): Multiplier for IQR
    remove (bool): If True, remove outliers; if False, just detect
    
    Returns:
    pd.DataFrame: Processed dataframe
    """
    if not isinstance(columns, list):
        columns = [columns]
    
    result_data = data.copy()
    
    for column in columns:
        if remove:
            result_data = remove_outliers(result_data, column, threshold)
        else:
            result_data = detect_outliers_iqr(result_data, column, threshold)
    
    return result_data
import pandas as pd
import numpy as np

def clean_csv_data(file_path, fill_method='mean', drop_threshold=0.5):
    """
    Load and clean CSV data by handling missing values.
    
    Parameters:
    file_path (str): Path to the CSV file
    fill_method (str): Method for filling missing values ('mean', 'median', 'mode', 'zero')
    drop_threshold (float): Drop columns with missing ratio above this threshold
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    
    original_shape = df.shape
    
    missing_ratio = df.isnull().sum() / len(df)
    columns_to_drop = missing_ratio[missing_ratio > drop_threshold].index
    df = df.drop(columns=columns_to_drop)
    
    for column in df.columns:
        if df[column].dtype in ['float64', 'int64']:
            if fill_method == 'mean':
                fill_value = df[column].mean()
            elif fill_method == 'median':
                fill_value = df[column].median()
            elif fill_method == 'zero':
                fill_value = 0
            else:
                fill_value = df[column].mode()[0] if not df[column].mode().empty else 0
        else:
            fill_value = df[column].mode()[0] if not df[column].mode().empty else 'Unknown'
        
        df[column] = df[column].fillna(fill_value)
    
    df = df.drop_duplicates()
    
    print(f"Original shape: {original_shape}")
    print(f"Cleaned shape: {df.shape}")
    print(f"Dropped columns: {list(columns_to_drop)}")
    
    return df

def detect_outliers_iqr(df, column):
    """
    Detect outliers using IQR method for a specific column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to check for outliers
    
    Returns:
    pd.Series: Boolean series indicating outliers
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' is not numeric")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    
    return outliers

def normalize_column(df, column, method='minmax'):
    """
    Normalize a column using specified method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to normalize
    method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
    pd.DataFrame: DataFrame with normalized column
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' is not numeric")
    
    df_copy = df.copy()
    
    if method == 'minmax':
        min_val = df_copy[column].min()
        max_val = df_copy[column].max()
        
        if max_val != min_val:
            df_copy[column] = (df_copy[column] - min_val) / (max_val - min_val)
        else:
            df_copy[column] = 0
    
    elif method == 'zscore':
        mean_val = df_copy[column].mean()
        std_val = df_copy[column].std()
        
        if std_val != 0:
            df_copy[column] = (df_copy[column] - mean_val) / std_val
        else:
            df_copy[column] = 0
    
    return df_copy

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, np.nan, 5],
        'C': [1, 1, 1, 1, 1],
        'D': [10, 20, 30, 40, 50]
    }
    
    df_sample = pd.DataFrame(sample_data)
    df_sample.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data('sample_data.csv', fill_method='mean', drop_threshold=0.3)
    
    print("\nSample data cleaned successfully")
    print(cleaned_df.head())
    
    import os
    if os.path.exists('sample_data.csv'):
        os.remove('sample_data.csv')