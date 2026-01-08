
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    data (list or np.array): The dataset.
    column (int): Index of the column to clean.
    
    Returns:
    np.array: Data with outliers removed.
    """
    data = np.array(data)
    col_data = data[:, column].astype(float)
    
    Q1 = np.percentile(col_data, 25)
    Q3 = np.percentile(col_data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    mask = (col_data >= lower_bound) & (col_data <= upper_bound)
    cleaned_data = data[mask]
    
    return cleaned_data

def calculate_statistics(data, column):
    """
    Calculate basic statistics for a column.
    
    Parameters:
    data (list or np.array): The dataset.
    column (int): Index of the column.
    
    Returns:
    dict: Dictionary containing mean, median, and std.
    """
    data = np.array(data)
    col_data = data[:, column].astype(float)
    
    stats = {
        'mean': np.mean(col_data),
        'median': np.median(col_data),
        'std': np.std(col_data)
    }
    
    return stats

if __name__ == "__main__":
    sample_data = [
        [1, 150.5],
        [2, 160.2],
        [3, 155.8],
        [4, 165.3],
        [5, 170.1],
        [6, 200.5],
        [7, 50.2]
    ]
    
    print("Original data:")
    for row in sample_data:
        print(row)
    
    cleaned = remove_outliers_iqr(sample_data, 1)
    print("\nCleaned data:")
    for row in cleaned:
        print(row)
    
    stats = calculate_statistics(cleaned, 1)
    print("\nStatistics for cleaned column:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method.
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize column using min-max scaling.
    """
    min_val = data[column].min()
    max_val = data[column].max()
    if max_val - min_val == 0:
        return data[column].apply(lambda x: 0.5)
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize column using z-score normalization.
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(df, numeric_columns, outlier_factor=1.5, normalization_method='minmax'):
    """
    Clean dataset by removing outliers and normalizing numeric columns.
    """
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col, outlier_factor)
            
            if normalization_method == 'minmax':
                cleaned_df[col] = normalize_minmax(cleaned_df, col)
            elif normalization_method == 'zscore':
                cleaned_df[col] = standardize_zscore(cleaned_df, col)
    
    return cleaned_df

def validate_dataframe(df, required_columns):
    """
    Validate dataframe structure and required columns.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    return Trueimport pandas as pd
import numpy as np
from typing import Optional

def clean_dataset(df: pd.DataFrame, 
                  drop_duplicates: bool = True, 
                  fillna_strategy: Optional[str] = 'mean',
                  columns_to_clean: Optional[list] = None) -> pd.DataFrame:
    """
    Clean a pandas DataFrame by handling duplicates and missing values.
    
    Parameters:
    df: Input DataFrame
    drop_duplicates: Whether to drop duplicate rows
    fillna_strategy: Strategy for filling NaN values ('mean', 'median', 'mode', or None)
    columns_to_clean: Specific columns to apply cleaning (None for all columns)
    
    Returns:
    Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if columns_to_clean is None:
        columns_to_clean = cleaned_df.columns.tolist()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    
    for column in columns_to_clean:
        if column in cleaned_df.columns:
            if cleaned_df[column].dtype in ['int64', 'float64']:
                if fillna_strategy == 'mean':
                    cleaned_df[column].fillna(cleaned_df[column].mean(), inplace=True)
                elif fillna_strategy == 'median':
                    cleaned_df[column].fillna(cleaned_df[column].median(), inplace=True)
                elif fillna_strategy == 'mode':
                    cleaned_df[column].fillna(cleaned_df[column].mode()[0], inplace=True)
            elif cleaned_df[column].dtype == 'object':
                cleaned_df[column].fillna('Unknown', inplace=True)
            elif cleaned_df[column].dtype == 'bool':
                cleaned_df[column].fillna(False, inplace=True)
    
    return cleaned_df

def remove_outliers_iqr(df: pd.DataFrame, 
                        columns: Optional[list] = None,
                        multiplier: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers using the Interquartile Range method.
    
    Parameters:
    df: Input DataFrame
    columns: Columns to check for outliers (None for all numeric columns)
    multiplier: IQR multiplier for outlier detection
    
    Returns:
    DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    filtered_df = df.copy()
    
    for column in columns:
        if column in filtered_df.columns and filtered_df[column].dtype in ['int64', 'float64']:
            Q1 = filtered_df[column].quantile(0.25)
            Q3 = filtered_df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            filtered_df = filtered_df[
                (filtered_df[column] >= lower_bound) & 
                (filtered_df[column] <= upper_bound)
            ]
    
    return filtered_df.reset_index(drop=True)