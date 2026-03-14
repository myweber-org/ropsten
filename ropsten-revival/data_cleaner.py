
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
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
    
    return filtered_df

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
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    cleaned_df = df.copy()
    
    for column in columns:
        if column in cleaned_df.columns:
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, column)
            except Exception as e:
                print(f"Warning: Could not clean column '{column}': {e}")
    
    return cleaned_df

def calculate_statistics(df):
    """
    Calculate basic statistics for numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    pd.DataFrame: Statistics summary
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        return pd.DataFrame()
    
    stats = {
        'mean': numeric_df.mean(),
        'median': numeric_df.median(),
        'std': numeric_df.std(),
        'min': numeric_df.min(),
        'max': numeric_df.max(),
        'count': numeric_df.count()
    }
    
    return pd.DataFrame(stats)
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column in a dataset using the IQR method.
    
    Parameters:
    data (numpy.ndarray): The dataset.
    column (int): The index of the column to clean.
    
    Returns:
    numpy.ndarray: The dataset with outliers removed from the specified column.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    
    if column >= data.shape[1] or column < 0:
        raise IndexError("Column index out of bounds")
    
    col_data = data[:, column]
    Q1 = np.percentile(col_data, 25)
    Q3 = np.percentile(col_data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    mask = (col_data >= lower_bound) & (col_data <= upper_bound)
    cleaned_data = data[mask]
    
    return cleaned_data

def example_usage():
    """
    Example usage of the remove_outliers_iqr function.
    """
    np.random.seed(42)
    sample_data = np.random.randn(100, 3)
    sample_data[0, 0] = 10.0
    sample_data[1, 0] = -10.0
    
    print("Original data shape:", sample_data.shape)
    cleaned = remove_outliers_iqr(sample_data, 0)
    print("Cleaned data shape:", cleaned.shape)

if __name__ == "__main__":
    example_usage()import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns
        self.categorical_columns = df.select_dtypes(exclude=[np.number]).columns

    def handle_missing_values(self, strategy='mean', fill_value=None):
        if strategy == 'mean' and self.numeric_columns.any():
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(
                self.df[self.numeric_columns].mean()
            )
        elif strategy == 'median' and self.numeric_columns.any():
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(
                self.df[self.numeric_columns].median()
            )
        elif strategy == 'mode':
            for col in self.df.columns:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
        elif strategy == 'constant' and fill_value is not None:
            self.df = self.df.fillna(fill_value)
        return self

    def remove_outliers(self, method='zscore', threshold=3):
        if method == 'zscore' and self.numeric_columns.any():
            z_scores = np.abs(stats.zscore(self.df[self.numeric_columns]))
            outlier_mask = (z_scores < threshold).all(axis=1)
            self.df = self.df[outlier_mask]
        elif method == 'iqr' and self.numeric_columns.any():
            for col in self.numeric_columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        return self

    def normalize_data(self, method='minmax'):
        if method == 'minmax' and self.numeric_columns.any():
            for col in self.numeric_columns:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
        elif method == 'standard' and self.numeric_columns.any():
            for col in self.numeric_columns:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    self.df[col] = (self.df[col] - mean_val) / std_val
        return self

    def get_cleaned_data(self):
        return self.df

def clean_dataset(df, missing_strategy='mean', outlier_method='zscore', normalize=False):
    cleaner = DataCleaner(df)
    cleaner.handle_missing_values(strategy=missing_strategy)
    cleaner.remove_outliers(method=outlier_method)
    if normalize:
        cleaner.normalize_data()
    return cleaner.get_cleaned_data()
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas DataFrame column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed from the specified column.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas DataFrame column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def calculate_summary_stats(data, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
    column (str): The column name to analyze.
    
    Returns:
    dict: Dictionary containing mean, median, and standard deviation.
    """
    stats = {
        'mean': data[column].mean(),
        'median': data[column].median(),
        'std': data[column].std()
    }
    return stats
import pandas as pd
import numpy as np

def clean_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in a DataFrame using specified strategy.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
    columns (list): List of columns to apply cleaning to, None for all columns
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if df.empty:
        return df
    
    if columns is None:
        columns = df.columns
    
    df_clean = df.copy()
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        if df_clean[col].isnull().sum() == 0:
            continue
            
        if strategy == 'mean':
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
        elif strategy == 'median':
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
        elif strategy == 'mode':
            mode_val = df_clean[col].mode()
            if not mode_val.empty:
                df_clean[col].fillna(mode_val[0], inplace=True)
        elif strategy == 'drop':
            df_clean = df_clean.dropna(subset=[col])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    return df_clean

def remove_outliers(df, columns=None, threshold=3):
    """
    Remove outliers using z-score method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of columns to check for outliers
    threshold (float): Z-score threshold for outlier detection
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if df.empty or columns is None:
        return df
    
    df_clean = df.copy()
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        if not pd.api.types.is_numeric_dtype(df_clean[col]):
            continue
            
        z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
        df_clean = df_clean[z_scores < threshold]
    
    return df_clean

def normalize_data(df, columns=None):
    """
    Normalize numeric columns to range [0, 1].
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of columns to normalize
    
    Returns:
    pd.DataFrame: Normalized DataFrame
    """
    if df.empty or columns is None:
        return df
    
    df_norm = df.copy()
    
    for col in columns:
        if col not in df_norm.columns:
            continue
            
        if not pd.api.types.is_numeric_dtype(df_norm[col]):
            continue
            
        col_min = df_norm[col].min()
        col_max = df_norm[col].max()
        
        if col_max > col_min:
            df_norm[col] = (df_norm[col] - col_min) / (col_max - col_min)
    
    return df_norm

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
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
    
    return filtered_df

def calculate_summary_stats(df, column):
    """
    Calculate summary statistics for a column.
    
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

def process_numerical_data(df, columns):
    """
    Process multiple numerical columns by removing outliers and calculating statistics.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to process
    
    Returns:
    tuple: (cleaned_df, stats_dict)
    """
    cleaned_df = df.copy()
    all_stats = {}
    
    for col in columns:
        if col in cleaned_df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            removed_count = original_count - len(cleaned_df)
            
            stats = calculate_summary_stats(cleaned_df, col)
            stats['outliers_removed'] = removed_count
            all_stats[col] = stats
    
    return cleaned_df, all_stats

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'temperature': [22, 23, 24, 25, 26, 27, 100, 28, 29, 30, -10],
        'humidity': [45, 46, 47, 48, 49, 50, 200, 51, 52, 53, -5],
        'pressure': [1013, 1014, 1015, 1016, 1017, 1018, 2000, 1019, 1020, 1021, 500]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df, stats = process_numerical_data(df, ['temperature', 'humidity', 'pressure'])
    
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\n" + "="*50 + "\n")
    
    print("Summary Statistics:")
    for col, col_stats in stats.items():
        print(f"\n{col}:")
        for stat_name, stat_value in col_stats.items():
            print(f"  {stat_name}: {stat_value:.2f}")import pandas as pd
import numpy as np

def clean_missing_data(file_path, strategy='mean', columns=None):
    """
    Clean missing values in a CSV file using specified strategy.
    
    Args:
        file_path (str): Path to the CSV file
        strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
        columns (list): Specific columns to clean, if None cleans all columns
    
    Returns:
        pandas.DataFrame: Cleaned dataframe
    """
    try:
        df = pd.read_csv(file_path)
        
        if columns is None:
            columns = df.columns
        
        for column in columns:
            if column in df.columns:
                if strategy == 'mean':
                    df[column].fillna(df[column].mean(), inplace=True)
                elif strategy == 'median':
                    df[column].fillna(df[column].median(), inplace=True)
                elif strategy == 'mode':
                    df[column].fillna(df[column].mode()[0], inplace=True)
                elif strategy == 'drop':
                    df.dropna(subset=[column], inplace=True)
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def save_cleaned_data(df, output_path):
    """
    Save cleaned dataframe to CSV file.
    
    Args:
        df (pandas.DataFrame): Cleaned dataframe
        output_path (str): Path to save the cleaned data
    """
    if df is not None:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = clean_missing_data(input_file, strategy='median')
    
    if cleaned_df is not None:
        save_cleaned_data(cleaned_df, output_file)
        print(f"Original shape: {pd.read_csv(input_file).shape}")
        print(f"Cleaned shape: {cleaned_df.shape}")
        print("Missing values after cleaning:")
        print(cleaned_df.isnull().sum())
import pandas as pd
import numpy as np

def clean_dataset(df, missing_strategy='mean', outlier_threshold=3):
    """
    Clean dataset by handling missing values and removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'drop')
    outlier_threshold (float): Z-score threshold for outlier detection
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if missing_strategy == 'mean':
        cleaned_df = cleaned_df.fillna(cleaned_df.mean())
    elif missing_strategy == 'median':
        cleaned_df = cleaned_df.fillna(cleaned_df.median())
    elif missing_strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    
    # Remove outliers using Z-score method
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    z_scores = np.abs((cleaned_df[numeric_cols] - cleaned_df[numeric_cols].mean()) / cleaned_df[numeric_cols].std())
    outlier_mask = (z_scores < outlier_threshold).all(axis=1)
    cleaned_df = cleaned_df[outlier_mask]
    
    return cleaned_df.reset_index(drop=True)

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate dataframe structure and content.
    
    Parameters:
    df (pd.DataFrame): Dataframe to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "Dataframe is empty"
    
    if len(df) < min_rows:
        return False, f"Dataframe has less than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Data validation passed"

def normalize_data(df, method='minmax'):
    """
    Normalize numeric columns in dataframe.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
    pd.DataFrame: Normalized dataframe
    """
    normalized_df = df.copy()
    numeric_cols = normalized_df.select_dtypes(include=[np.number]).columns
    
    if method == 'minmax':
        for col in numeric_cols:
            col_min = normalized_df[col].min()
            col_max = normalized_df[col].max()
            if col_max != col_min:
                normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)
    
    elif method == 'zscore':
        for col in numeric_cols:
            col_mean = normalized_df[col].mean()
            col_std = normalized_df[col].std()
            if col_std != 0:
                normalized_df[col] = (normalized_df[col] - col_mean) / col_std
    
    return normalized_df

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, np.nan, 4, 100],
        'B': [5, 6, 7, np.nan, 8],
        'C': [9, 10, 11, 12, 13]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, missing_strategy='mean', outlier_threshold=2)
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid, message = validate_data(cleaned, required_columns=['A', 'B', 'C'])
    print(f"\nValidation: {is_valid} - {message}")
    
    normalized = normalize_data(cleaned, method='minmax')
    print("\nNormalized DataFrame:")
    print(normalized)