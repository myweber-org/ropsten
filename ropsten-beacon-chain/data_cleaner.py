
import pandas as pd
import numpy as np
from scipy import stats

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    original_shape = df.shape
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        z_scores = np.abs(stats.zscore(df[col].dropna()))
        df = df[(z_scores < 3) | df[col].isna()]
    
    for col in numeric_cols:
        if df[col].std() > 0:
            df[col] = (df[col] - df[col].mean()) / df[col].std()
    
    print(f"Original shape: {original_shape}")
    print(f"Cleaned shape: {df.shape}")
    print(f"Removed {original_shape[0] - df.shape[0]} rows")
    
    return df

def save_cleaned_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = load_and_clean_data(input_file)
    save_cleaned_data(cleaned_df, output_file)import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column):
    """
    Remove outliers using the Interquartile Range method.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    """
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    """
    min_val = data[column].min()
    max_val = data[column].max()
    data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    return data

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization.
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    data[column + '_standardized'] = (data[column] - mean_val) / std_val
    return data

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    """
    Main function to clean dataset by removing outliers and normalizing numeric columns.
    """
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if outlier_method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        elif outlier_method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
        
        if normalize_method == 'minmax':
            cleaned_df = normalize_minmax(cleaned_df, col)
        elif normalize_method == 'zscore':
            cleaned_df = normalize_zscore(cleaned_df, col)
    
    return cleaned_dfimport pandas as pd

def clean_dataset(file_path, output_path=None):
    """
    Load a CSV file, remove duplicate rows, sort by the first column,
    and save the cleaned data.
    """
    try:
        df = pd.read_csv(file_path)
        initial_count = len(df)
        df_cleaned = df.drop_duplicates()
        df_cleaned = df_cleaned.sort_values(by=df_cleaned.columns[0])
        final_count = len(df_cleaned)
        duplicates_removed = initial_count - final_count

        if output_path is None:
            output_path = file_path.replace('.csv', '_cleaned.csv')

        df_cleaned.to_csv(output_path, index=False)
        print(f"Cleaning complete. Removed {duplicates_removed} duplicate rows.")
        print(f"Cleaned data saved to: {output_path}")
        return df_cleaned

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    input_file = "sample_data.csv"
    clean_dataset(input_file)
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (str): Method to fill missing values: 'mean', 'median', 'mode', or 'drop'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows.")
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
        print("Removed rows with missing values.")
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            elif fill_missing == 'median':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        print(f"Filled missing numeric values with {fill_missing}.")
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 'Unknown')
        print("Filled missing categorical values with mode.")
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate a DataFrame for required columns and data types.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return False
    
    if df.empty:
        print("DataFrame is empty.")
        return False
    
    print("Dataset validation passed.")
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, None],
        'B': [5, None, 7, 8, 9],
        'C': ['x', 'y', 'y', None, 'z']
    }
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid = validate_dataset(cleaned, required_columns=['A', 'B'])
    print(f"\nDataset valid: {is_valid}")
import pandas as pd
import numpy as np
from scipy import stats

def load_data(filepath):
    return pd.read_csv(filepath)

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_column(df, column):
    df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    return df

def clean_dataset(input_file, output_file):
    df = load_data(input_file)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        df = remove_outliers_iqr(df, col)
    
    for col in numeric_columns:
        df = normalize_column(df, col)
    
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")
    
    return df

if __name__ == "__main__":
    cleaned_df = clean_dataset('raw_data.csv', 'cleaned_data.csv')
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        multiplier: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def normalize_minmax(data, column):
    """
    Normalize data using min-max scaling to [0, 1] range.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(data), index=data.index)
    
    return (data[column] - min_val) / (max_val - min_val)

def standardize_zscore(data, column):
    """
    Standardize data using z-score normalization.
    
    Args:
        data: pandas DataFrame
        column: column name to standardize
    
    Returns:
        Series with standardized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    return (data[column] - mean_val) / std_val

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in specified columns.
    
    Args:
        data: pandas DataFrame
        strategy: imputation strategy ('mean', 'median', 'mode', 'constant')
        columns: list of columns to process (None for all numeric columns)
    
    Returns:
        DataFrame with missing values handled
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    data_filled = data.copy()
    
    for col in columns:
        if col not in data.columns:
            continue
            
        if strategy == 'mean':
            fill_value = data[col].mean()
        elif strategy == 'median':
            fill_value = data[col].median()
        elif strategy == 'mode':
            fill_value = data[col].mode()[0] if not data[col].mode().empty else 0
        elif strategy == 'constant':
            fill_value = 0
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        data_filled[col] = data[col].fillna(fill_value)
    
    return data_filled

def clean_dataset(data, numeric_columns=None, outlier_multiplier=1.5, 
                  normalize_cols=None, standardize_cols=None, 
                  missing_strategy='mean'):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric columns to process
        outlier_multiplier: IQR multiplier for outlier removal
        normalize_cols: columns to normalize using min-max
        standardize_cols: columns to standardize using z-score
        missing_strategy: strategy for handling missing values
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    # Handle missing values
    cleaned_data = handle_missing_values(
        cleaned_data, 
        strategy=missing_strategy, 
        columns=numeric_columns
    )
    
    # Remove outliers from numeric columns
    for col in numeric_columns:
        if col in cleaned_data.columns:
            cleaned_data = remove_outliers_iqr(
                cleaned_data, 
                col, 
                multiplier=outlier_multiplier
            )
    
    # Apply normalization
    if normalize_cols:
        for col in normalize_cols:
            if col in cleaned_data.columns:
                cleaned_data[f'{col}_normalized'] = normalize_minmax(cleaned_data, col)
    
    # Apply standardization
    if standardize_cols:
        for col in standardize_cols:
            if col in cleaned_data.columns:
                cleaned_data[f'{col}_standardized'] = standardize_zscore(cleaned_data, col)
    
    return cleaned_data

def validate_data(data, required_columns=None, min_rows=1):
    """
    Validate data structure and content.
    
    Args:
        data: pandas DataFrame
        required_columns: list of required column names
        min_rows: minimum number of rows required
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(data, pd.DataFrame):
        return False, "Data must be a pandas DataFrame"
    
    if len(data) < min_rows:
        return False, f"Data must have at least {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Data validation passed"
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
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
import pandas as pd
import numpy as np

def clean_dataset(df, strategy='mean', outlier_threshold=3):
    """
    Clean dataset by handling missing values and removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
    outlier_threshold (float): Z-score threshold for outlier detection
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if strategy == 'mean':
        cleaned_df = cleaned_df.fillna(cleaned_df.mean())
    elif strategy == 'median':
        cleaned_df = cleaned_df.fillna(cleaned_df.median())
    elif strategy == 'mode':
        cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
    elif strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    
    # Remove outliers using Z-score method
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    z_scores = np.abs((cleaned_df[numeric_cols] - cleaned_df[numeric_cols].mean()) / cleaned_df[numeric_cols].std())
    outlier_mask = (z_scores < outlier_threshold).all(axis=1)
    cleaned_df = cleaned_df[outlier_mask]
    
    return cleaned_df.reset_index(drop=True)

def normalize_data(df, method='minmax'):
    """
    Normalize numerical columns in dataframe.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    method (str): Normalization method ('minmax', 'standard')
    
    Returns:
    pd.DataFrame: Normalized dataframe
    """
    normalized_df = df.copy()
    numeric_cols = normalized_df.select_dtypes(include=[np.number]).columns
    
    if method == 'minmax':
        for col in numeric_cols:
            min_val = normalized_df[col].min()
            max_val = normalized_df[col].max()
            if max_val != min_val:
                normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
    
    elif method == 'standard':
        for col in numeric_cols:
            mean_val = normalized_df[col].mean()
            std_val = normalized_df[col].std()
            if std_val != 0:
                normalized_df[col] = (normalized_df[col] - mean_val) / std_val
    
    return normalized_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate dataframe structure and content.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Data validation passed"
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
    for col in columns:
        if col in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[col]):
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    
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
    
    stats = {
        'mean': numeric_df.mean(),
        'median': numeric_df.median(),
        'std': numeric_df.std(),
        'min': numeric_df.min(),
        'max': numeric_df.max(),
        'count': numeric_df.count()
    }
    
    return pd.DataFrame(stats)
import pandas as pd
import re
from typing import List, Optional

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        
    def remove_duplicates(self, subset: Optional[List[str]] = None) -> pd.DataFrame:
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep='first')
        removed_count = initial_count - len(self.df)
        print(f"Removed {removed_count} duplicate rows")
        return self.df
    
    def normalize_text(self, column: str) -> pd.DataFrame:
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        
        def clean_text(text: str) -> str:
            if pd.isna(text):
                return text
            text = str(text).lower().strip()
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'[^\w\s]', '', text)
            return text
        
        self.df[column] = self.df[column].apply(clean_text)
        print(f"Normalized text in column '{column}'")
        return self.df
    
    def handle_missing_values(self, column: str, strategy: str = 'mean') -> pd.DataFrame:
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        
        if strategy == 'mean':
            fill_value = self.df[column].mean()
        elif strategy == 'median':
            fill_value = self.df[column].median()
        elif strategy == 'mode':
            fill_value = self.df[column].mode()[0]
        elif strategy == 'drop':
            self.df = self.df.dropna(subset=[column])
            print(f"Dropped rows with missing values in column '{column}'")
            return self.df
        else:
            raise ValueError("Strategy must be 'mean', 'median', 'mode', or 'drop'")
        
        missing_count = self.df[column].isna().sum()
        self.df[column] = self.df[column].fillna(fill_value)
        print(f"Filled {missing_count} missing values in column '{column}' with {strategy}: {fill_value}")
        return self.df
    
    def get_dataframe(self) -> pd.DataFrame:
        return self.df.copy()

def example_usage():
    data = {
        'id': [1, 2, 3, 4, 5, 5],
        'name': ['John Doe', 'Jane Smith', 'john doe', 'Bob Johnson', 'Alice Brown', 'Alice Brown'],
        'age': [25, 30, None, 35, 28, 28],
        'email': ['john@example.com', 'jane@example.com', 'john@example.com', 'bob@example.com', 'alice@example.com', 'alice@example.com']
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaner = DataCleaner(df)
    cleaner.remove_duplicates(subset=['id', 'email'])
    cleaner.normalize_text('name')
    cleaner.handle_missing_values('age', strategy='mean')
    
    cleaned_df = cleaner.get_dataframe()
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    return cleaned_df

if __name__ == "__main__":
    example_usage()
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_method=None):
    """
    Clean a pandas DataFrame by handling missing values and removing duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to remove duplicate rows.
    fill_method (str or None): Method to fill missing values: 
                               'mean', 'median', 'mode', or None to drop rows.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if fill_method is None:
        cleaned_df = cleaned_df.dropna()
    else:
        if fill_method == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_method == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif fill_method == 'mode':
            cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
        else:
            raise ValueError("fill_method must be 'mean', 'median', 'mode', or None")
    
    # Remove duplicates
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate a DataFrame for required columns and data types.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    bool: True if validation passes, raises ValueError otherwise.
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for numeric columns that might have been incorrectly typed
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        if df[col].isnull().all():
            raise ValueError(f"Column '{col}' is entirely null")
    
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, None, 4, 4],
        'B': [5, None, 7, 8, 8],
        'C': ['x', 'y', 'z', 'x', 'x']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame (drop NA, remove duplicates):")
    cleaned = clean_dataset(df, drop_duplicates=True, fill_method=None)
    print(cleaned)
    
    try:
        validate_dataset(cleaned, required_columns=['A', 'B', 'C'])
        print("\nDataset validation passed.")
    except ValueError as e:
        print(f"\nDataset validation failed: {e}")import numpy as np
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

def calculate_summary_statistics(df):
    """
    Calculate summary statistics for numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    pd.DataFrame: Summary statistics
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    summary = df[numeric_cols].describe()
    
    return summary

def normalize_column(df, column):
    """
    Normalize a column using min-max scaling.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to normalize
    
    Returns:
    pd.DataFrame: DataFrame with normalized column
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = df[column].min()
    max_val = df[column].max()
    
    if max_val == min_val:
        df[f'{column}_normalized'] = 0.0
    else:
        df[f'{column}_normalized'] = (df[column] - min_val) / (max_val - min_val)
    
    return df

def main():
    """
    Example usage of data cleaning functions.
    """
    np.random.seed(42)
    
    data = {
        'id': range(1, 101),
        'value': np.random.normal(100, 15, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(data)
    
    print("Original DataFrame shape:", df.shape)
    print("\nOriginal summary statistics:")
    print(calculate_summary_statistics(df))
    
    cleaned_df = remove_outliers_iqr(df, 'value')
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    
    normalized_df = normalize_column(cleaned_df, 'value')
    print("\nNormalized column sample:")
    print(normalized_df[['value', 'value_normalized']].head())

if __name__ == "__main__":
    main()