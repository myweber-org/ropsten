
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
    
    return filtered_df

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
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, column)
            except Exception as e:
                print(f"Warning: Could not clean column '{column}': {e}")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    sample_df = pd.DataFrame(sample_data)
    sample_df.loc[::100, 'A'] = 500
    
    print("Original dataset shape:", sample_df.shape)
    print("Original statistics for column 'A':")
    print(calculate_summary_statistics(sample_df, 'A'))
    
    cleaned_df = clean_dataset(sample_df, ['A', 'B'])
    
    print("\nCleaned dataset shape:", cleaned_df.shape)
    print("Cleaned statistics for column 'A':")
    print(calculate_summary_statistics(cleaned_df, 'A'))
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def clean_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            initial_count = len(df)
            df = remove_outliers_iqr(df, col)
            removed_count = initial_count - len(df)
            if removed_count > 0:
                print(f"Removed {removed_count} outliers from column '{col}'")
        
        cleaned_file_path = file_path.replace('.csv', '_cleaned.csv')
        df.to_csv(cleaned_file_path, index=False)
        print(f"Cleaned data saved to: {cleaned_file_path}")
        return cleaned_file_path
    except Exception as e:
        print(f"Error cleaning dataset: {e}")
        return None

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    sample_df = pd.DataFrame(sample_data)
    sample_df.loc[::100, 'A'] = 500
    sample_df.loc[::50, 'B'] = 300
    
    sample_df.to_csv('sample_data.csv', index=False)
    clean_dataset('sample_data.csv')import pandas as pd
import numpy as np

def load_data(filepath):
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

def remove_outliers_iqr(df, column):
    """Remove outliers using the IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    removed_count = len(df) - len(filtered_df)
    print(f"Removed {removed_count} outliers from column '{column}'")
    return filtered_df

def normalize_column(df, column):
    """Normalize a column using min-max scaling."""
    min_val = df[column].min()
    max_val = df[column].max()
    if max_val - min_val == 0:
        print(f"Warning: Column '{column}' has constant values. Skipping normalization.")
        return df
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    print(f"Normalized column '{column}'")
    return df

def clean_data(df, numeric_columns):
    """Main cleaning function."""
    if df is None or df.empty:
        print("No data to clean.")
        return df
    
    original_shape = df.shape
    print(f"Original data shape: {original_shape}")
    
    for col in numeric_columns:
        if col in df.columns:
            df = remove_outliers_iqr(df, col)
        else:
            print(f"Warning: Column '{col}' not found in data.")
    
    for col in numeric_columns:
        if col in df.columns:
            df = normalize_column(df, col)
    
    print(f"Cleaned data shape: {df.shape}")
    print(f"Rows removed: {original_shape[0] - df.shape[0]}")
    return df

def save_cleaned_data(df, output_path):
    """Save cleaned data to CSV."""
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    numeric_cols = ['age', 'income', 'score']
    
    data = load_data(input_file)
    if data is not None:
        cleaned_data = clean_data(data, numeric_cols)
        save_cleaned_data(cleaned_data, output_file)
import pandas as pd
import numpy as np
from typing import Optional, List

def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: Input DataFrame
        subset: Columns to consider for identifying duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def fill_missing_values(df: pd.DataFrame, strategy: str = 'mean', columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Fill missing values in DataFrame columns.
    
    Args:
        df: Input DataFrame
        strategy: 'mean', 'median', 'mode', or 'constant'
        columns: Specific columns to fill, fills all if None
    
    Returns:
        DataFrame with missing values filled
    """
    df_copy = df.copy()
    
    if columns is None:
        columns = df_copy.columns
    
    for col in columns:
        if df_copy[col].isnull().any():
            if strategy == 'mean':
                fill_value = df_copy[col].mean()
            elif strategy == 'median':
                fill_value = df_copy[col].median()
            elif strategy == 'mode':
                fill_value = df_copy[col].mode()[0]
            elif strategy == 'constant':
                fill_value = 0
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            df_copy[col] = df_copy[col].fillna(fill_value)
    
    return df_copy

def normalize_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Normalize a column to range [0, 1].
    
    Args:
        df: Input DataFrame
        column: Column name to normalize
    
    Returns:
        DataFrame with normalized column
    """
    df_copy = df.copy()
    col_min = df_copy[column].min()
    col_max = df_copy[column].max()
    
    if col_max != col_min:
        df_copy[column] = (df_copy[column] - col_min) / (col_max - col_min)
    
    return df_copy

def filter_outliers(df: pd.DataFrame, column: str, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
    """
    Filter outliers from a DataFrame column.
    
    Args:
        df: Input DataFrame
        column: Column name to filter
        method: 'iqr' for interquartile range or 'zscore' for standard deviations
        threshold: Threshold value for filtering
    
    Returns:
        DataFrame with outliers removed
    """
    df_copy = df.copy()
    
    if method == 'iqr':
        Q1 = df_copy[column].quantile(0.25)
        Q3 = df_copy[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        mask = (df_copy[column] >= lower_bound) & (df_copy[column] <= upper_bound)
    
    elif method == 'zscore':
        mean = df_copy[column].mean()
        std = df_copy[column].std()
        if std > 0:
            z_scores = np.abs((df_copy[column] - mean) / std)
            mask = z_scores <= threshold
        else:
            mask = pd.Series([True] * len(df_copy))
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return df_copy[mask]

def clean_dataset(df: pd.DataFrame, 
                  remove_dups: bool = True,
                  fill_na: bool = True,
                  fill_strategy: str = 'mean',
                  normalize_cols: Optional[List[str]] = None,
                  filter_outlier_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: Input DataFrame
        remove_dups: Whether to remove duplicates
        fill_na: Whether to fill missing values
        fill_strategy: Strategy for filling missing values
        normalize_cols: Columns to normalize
        filter_outlier_cols: Columns to filter outliers from
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if remove_dups:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if fill_na:
        cleaned_df = fill_missing_values(cleaned_df, strategy=fill_strategy)
    
    if normalize_cols:
        for col in normalize_cols:
            if col in cleaned_df.columns:
                cleaned_df = normalize_column(cleaned_df, col)
    
    if filter_outlier_cols:
        for col in filter_outlier_cols:
            if col in cleaned_df.columns:
                cleaned_df = filter_outliers(cleaned_df, col)
    
    return cleaned_df