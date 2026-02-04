
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, data):
        self.data = data
        self.original_shape = data.shape
        
    def remove_outliers_iqr(self, columns=None):
        if columns is None:
            columns = self.data.columns if hasattr(self.data, 'columns') else range(self.data.shape[1])
        
        cleaned_data = self.data.copy()
        for col in columns:
            Q1 = cleaned_data[col].quantile(0.25)
            Q3 = cleaned_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            cleaned_data = cleaned_data[(cleaned_data[col] >= lower_bound) & (cleaned_data[col] <= upper_bound)]
        
        self.removed_count = self.original_shape[0] - cleaned_data.shape[0]
        self.data = cleaned_data
        return self
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.data.columns if hasattr(self.data, 'columns') else range(self.data.shape[1])
        
        normalized_data = self.data.copy()
        for col in columns:
            min_val = normalized_data[col].min()
            max_val = normalized_data[col].max()
            if max_val != min_val:
                normalized_data[col] = (normalized_data[col] - min_val) / (max_val - min_val)
        
        self.data = normalized_data
        return self
    
    def standardize_zscore(self, columns=None):
        if columns is None:
            columns = self.data.columns if hasattr(self.data, 'columns') else range(self.data.shape[1])
        
        standardized_data = self.data.copy()
        for col in columns:
            mean_val = standardized_data[col].mean()
            std_val = standardized_data[col].std()
            if std_val > 0:
                standardized_data[col] = (standardized_data[col] - mean_val) / std_val
        
        self.data = standardized_data
        return self
    
    def get_summary(self):
        summary = {
            'original_samples': self.original_shape[0],
            'current_samples': self.data.shape[0],
            'features': self.data.shape[1],
            'removed_outliers': getattr(self, 'removed_count', 0)
        }
        return summary

def create_sample_data(n_samples=1000, n_features=5):
    np.random.seed(42)
    data = np.random.randn(n_samples, n_features)
    data = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(n_features)])
    return data

if __name__ == "__main__":
    sample_data = create_sample_data()
    cleaner = DataCleaner(sample_data)
    
    print("Original data shape:", cleaner.original_shape)
    
    cleaner.remove_outliers_iqr()
    cleaner.normalize_minmax()
    
    summary = cleaner.get_summary()
    print(f"Cleaned data shape: {summary['current_samples']} samples, {summary['features']} features")
    print(f"Removed outliers: {summary['removed_outliers']}")
    
    print("\nFirst 5 rows of cleaned data:")
    print(cleaner.data.head())
import pandas as pd

def clean_dataset(df, column_name):
    """
    Remove duplicate rows and sort the DataFrame by a specified column.
    
    Args:
        df (pd.DataFrame): The input DataFrame to clean.
        column_name (str): The column name to sort by.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame with duplicates removed and sorted.
    """
    if df.empty:
        return df
    
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    df_cleaned = df.drop_duplicates().reset_index(drop=True)
    df_cleaned = df_cleaned.sort_values(by=column_name).reset_index(drop=True)
    
    return df_cleaned

def filter_by_threshold(df, column_name, threshold):
    """
    Filter rows where the column value is greater than a threshold.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The column to apply the filter on.
        threshold (float): The threshold value.
    
    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    if df.empty:
        return df
    
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    filtered_df = df[df[column_name] > threshold].reset_index(drop=True)
    return filtered_df

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 4, 5],
        'value': [10.5, 20.3, 20.3, 15.7, 8.9, 8.9, 30.1],
        'category': ['A', 'B', 'B', 'A', 'C', 'C', 'B']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    cleaned_df = clean_dataset(df, 'value')
    print("Cleaned DataFrame (duplicates removed, sorted by 'value'):")
    print(cleaned_df)
    print()
    
    filtered_df = filter_by_threshold(cleaned_df, 'value', 15.0)
    print("Filtered DataFrame (value > 15.0):")
    print(filtered_df)import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    original_shape = df.shape
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
        print(f"Removed {original_shape[0] - cleaned_df.shape[0]} duplicate rows.")
    
    if cleaned_df.isnull().sum().any():
        print("Handling missing values...")
        for column in cleaned_df.columns:
            if cleaned_df[column].isnull().sum() > 0:
                if fill_missing == 'mean' and pd.api.types.is_numeric_dtype(cleaned_df[column]):
                    fill_value = cleaned_df[column].mean()
                    cleaned_df[column] = cleaned_df[column].fillna(fill_value)
                    print(f"Filled missing values in '{column}' with mean: {fill_value:.2f}")
                elif fill_missing == 'median' and pd.api.types.is_numeric_dtype(cleaned_df[column]):
                    fill_value = cleaned_df[column].median()
                    cleaned_df[column] = cleaned_df[column].fillna(fill_value)
                    print(f"Filled missing values in '{column}' with median: {fill_value:.2f}")
                elif fill_missing == 'mode':
                    fill_value = cleaned_df[column].mode()[0]
                    cleaned_df[column] = cleaned_df[column].fillna(fill_value)
                    print(f"Filled missing values in '{column}' with mode: {fill_value}")
                else:
                    cleaned_df[column] = cleaned_df[column].fillna('Unknown')
                    print(f"Filled missing values in '{column}' with 'Unknown'")
    
    print(f"Dataset cleaned. Original shape: {original_shape}, Cleaned shape: {cleaned_df.shape}")
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the dataset structure and content.
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    if len(df) < min_rows:
        raise ValueError(f"Dataset must have at least {min_rows} rows, but has {len(df)}")
    
    print("Data validation passed.")
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, np.nan, 6],
        'B': [10, 20, 20, np.nan, 50, 60],
        'C': ['X', 'Y', 'Y', 'Z', np.nan, 'X']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    try:
        validate_data(cleaned, required_columns=['A', 'B', 'C'], min_rows=3)
    except ValueError as e:
        print(f"Validation error: {e}")
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Args:
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
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): List of column names to clean
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{col}'")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and data integrity.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list, optional): List of required column names
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return True
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    return True

def get_data_summary(df):
    """
    Generate a summary of the DataFrame including missing values and basic statistics.
    
    Args:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        dict: Summary statistics
    """
    summary = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.astype(str).to_dict(),
        'numeric_stats': {}
    }
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        summary['numeric_stats'][col] = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'median': df[col].median()
        }
    
    return summary

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.randint(1, 100, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    
    print("Original DataFrame shape:", df.shape)
    print("\nData Summary:")
    summary = get_data_summary(df)
    print(f"Total rows: {summary['total_rows']}")
    print(f"Total columns: {summary['total_columns']}")
    
    cleaned_df = clean_numeric_data(df, ['A', 'B'])
    print("\nCleaned DataFrame shape:", cleaned_df.shape)