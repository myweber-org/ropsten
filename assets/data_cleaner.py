import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def remove_outliers_zscore(df, column, threshold=3):
    z_scores = np.abs(stats.zscore(df[column]))
    return df[z_scores < threshold]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def normalize_zscore(df, column):
    mean_val = df[column].mean()
    std_val = df[column].std()
    df[column + '_standardized'] = (df[column] - mean_val) / std_val
    return df

def handle_missing_values(df, strategy='mean'):
    if strategy == 'mean':
        return df.fillna(df.mean())
    elif strategy == 'median':
        return df.fillna(df.median())
    elif strategy == 'mode':
        return df.fillna(df.mode().iloc[0])
    elif strategy == 'drop':
        return df.dropna()
    else:
        raise ValueError("Invalid strategy. Choose from 'mean', 'median', 'mode', or 'drop'")

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method=None, missing_strategy='mean'):
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            if outlier_method == 'iqr':
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
            elif outlier_method == 'zscore':
                cleaned_df = remove_outliers_zscore(cleaned_df, col)
            
            if normalize_method == 'minmax':
                cleaned_df = normalize_minmax(cleaned_df, col)
            elif normalize_method == 'zscore':
                cleaned_df = normalize_zscore(cleaned_df, col)
    
    cleaned_df = handle_missing_values(cleaned_df, strategy=missing_strategy)
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 3, 4, 5, 100, 6, 7, 8, 9],
        'B': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'C': [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, ['A', 'B', 'C'], outlier_method='iqr', normalize_method='minmax')
    print("\nCleaned DataFrame:")
    print(cleaned)
import numpy as np
import pandas as pd

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

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing statistical measures
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

def normalize_column(df, column, method='minmax'):
    """
    Normalize a DataFrame column using specified method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to normalize
    method (str): Normalization method ('minmax' or 'zscore')
    
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
            df_copy[f'{column}_normalized'] = (df_copy[column] - min_val) / (max_val - min_val)
        else:
            df_copy[f'{column}_normalized'] = 0.5
    
    elif method == 'zscore':
        mean_val = df_copy[column].mean()
        std_val = df_copy[column].std()
        if std_val > 0:
            df_copy[f'{column}_normalized'] = (df_copy[column] - mean_val) / std_val
        else:
            df_copy[f'{column}_normalized'] = 0
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    return df_copy

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'values': [10, 12, 12, 13, 12, 11, 14, 13, 15, 100, 12, 13, 12, 11, 10]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    # Remove outliers
    cleaned_df = remove_outliers_iqr(df, 'values')
    print("DataFrame after removing outliers:")
    print(cleaned_df)
    print()
    
    # Calculate statistics
    stats = calculate_basic_stats(df, 'values')
    print("Statistics for 'values' column:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}" if isinstance(value, (int, float)) else f"{key}: {value}")
    print()
    
    # Normalize column
    normalized_df = normalize_column(df, 'values', method='minmax')
    print("DataFrame with normalized column:")
    print(normalized_df)
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    cleaned_df = cleaned_df.dropna()
    cleaned_df = cleaned_df.reset_index(drop=True)
    return cleaned_df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    })
    cleaned_data = clean_dataset(sample_data, ['A', 'B', 'C'])
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {cleaned_data.shape}")
import pandas as pd
import numpy as np
from pathlib import Path

def clean_data(input_file, output_file=None):
    """
    Clean and preprocess CSV data by handling missing values,
    removing duplicates, and standardizing column names.
    """
    try:
        df = pd.read_csv(input_file)
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Remove duplicate rows
        initial_count = len(df)
        df.drop_duplicates(inplace=True)
        duplicates_removed = initial_count - len(df)
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col].fillna(df[col].median(), inplace=True)
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col].fillna('unknown', inplace=True)
        
        # Remove rows where critical columns are still null
        critical_cols = ['id', 'timestamp', 'value']
        existing_critical = [col for col in critical_cols if col in df.columns]
        if existing_critical:
            df.dropna(subset=existing_critical, inplace=True)
        
        # Generate output filename if not provided
        if output_file is None:
            input_path = Path(input_file)
            output_file = input_path.parent / f"cleaned_{input_path.name}"
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        
        # Print cleaning summary
        print(f"Data cleaning completed:")
        print(f"  - Removed {duplicates_removed} duplicate rows")
        print(f"  - Processed {len(numeric_cols)} numeric columns")
        print(f"  - Processed {len(categorical_cols)} categorical columns")
        print(f"  - Final dataset: {len(df)} rows, {len(df.columns)} columns")
        print(f"  - Cleaned data saved to: {output_file}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: Input file '{input_file}' is empty.")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage
    input_csv = "raw_data.csv"
    cleaned_df = clean_data(input_csv)
    
    if cleaned_df is not None:
        print("\nFirst 5 rows of cleaned data:")
        print(cleaned_df.head())
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
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                mask = (self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)
                df_clean = df_clean[mask]
                
        self.df = df_clean.reset_index(drop=True)
        return self
        
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
                    
        return self
        
    def standardize_zscore(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    self.df[col] = (self.df[col] - mean_val) / std_val
                    
        return self
        
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                median_val = self.df[col].median()
                self.df[col] = self.df[col].fillna(median_val)
                
        return self
        
    def get_cleaned_data(self):
        return self.df
        
    def get_removed_count(self):
        return self.original_shape[0] - self.df.shape[0]
        
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'cleaned_rows': self.df.shape[0],
            'removed_rows': self.get_removed_count(),
            'columns': list(self.df.columns),
            'dtypes': {col: str(dtype) for col, dtype in self.df.dtypes.items()}
        }
        return summary

def clean_dataset(df, outlier_threshold=1.5, normalize=True, fill_missing=True):
    cleaner = DataCleaner(df)
    
    if fill_missing:
        cleaner.fill_missing_median()
        
    cleaner.remove_outliers_iqr(threshold=outlier_threshold)
    
    if normalize:
        cleaner.normalize_minmax()
        
    return cleaner.get_cleaned_data(), cleaner.get_summary()
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        factor: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        threshold: Z-score threshold (default 3)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    mask = z_scores < threshold
    return data[mask]

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
    Args:
        data: pandas DataFrame or Series
        column: column name to normalize
    
    Returns:
        Normalized data
    """
    if isinstance(data, pd.DataFrame):
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        col_data = data[column]
    else:
        col_data = data
    
    min_val = col_data.min()
    max_val = col_data.max()
    
    if max_val == min_val:
        return col_data
    
    return (col_data - min_val) / (max_val - min_val)

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization.
    
    Args:
        data: pandas DataFrame or Series
        column: column name to normalize
    
    Returns:
        Standardized data
    """
    if isinstance(data, pd.DataFrame):
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        col_data = data[column]
    else:
        col_data = data
    
    mean_val = col_data.mean()
    std_val = col_data.std()
    
    if std_val == 0:
        return col_data
    
    return (col_data - mean_val) / std_val

def clean_dataset(df, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: pandas DataFrame
        numeric_columns: list of numeric columns to process (default: all numeric)
        outlier_method: 'iqr' or 'zscore' (default: 'iqr')
        normalize_method: 'minmax' or 'zscore' (default: 'minmax')
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    # Remove outliers
    for col in numeric_columns:
        if col in cleaned_df.columns:
            if outlier_method == 'iqr':
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
            elif outlier_method == 'zscore':
                cleaned_df = remove_outliers_zscore(cleaned_df, col)
    
    # Normalize data
    for col in numeric_columns:
        if col in cleaned_df.columns:
            if normalize_method == 'minmax':
                cleaned_df[col] = normalize_minmax(cleaned_df, col)
            elif normalize_method == 'zscore':
                cleaned_df[col] = normalize_zscore(cleaned_df, col)
    
    return cleaned_df

def validate_data(df, required_columns=None, allow_nan=False):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: list of required column names
        allow_nan: whether to allow NaN values
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    if not allow_nan and df.isnull().any().any():
        nan_cols = df.columns[df.isnull().any()].tolist()
        return False, f"NaN values found in columns: {nan_cols}"
    
    return True, "Data validation passed"