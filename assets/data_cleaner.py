import numpy as np
import pandas as pd

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
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data.copy()

def normalize_minmax(data, column):
    """
    Normalize data to [0, 1] range using min-max scaling.
    
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
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

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
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(data, numeric_columns=None, outlier_factor=1.5):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric columns to process (default: all numeric)
        outlier_factor: IQR factor for outlier removal
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_data = data.copy()
    
    if numeric_columns is None:
        numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns.tolist()
    
    for column in numeric_columns:
        if column in cleaned_data.columns:
            cleaned_data = remove_outliers_iqr(cleaned_data, column, outlier_factor)
    
    return cleaned_data

def process_data_pipeline(data, config):
    """
    Process data through a configurable pipeline.
    
    Args:
        data: pandas DataFrame
        config: dictionary with processing options
    
    Returns:
        Processed DataFrame
    """
    processed_data = data.copy()
    
    if config.get('remove_outliers', False):
        columns = config.get('outlier_columns', processed_data.select_dtypes(include=[np.number]).columns.tolist())
        factor = config.get('outlier_factor', 1.5)
        
        for column in columns:
            if column in processed_data.columns:
                processed_data = remove_outliers_iqr(processed_data, column, factor)
    
    if config.get('normalize', False):
        columns = config.get('normalize_columns', [])
        method = config.get('normalize_method', 'minmax')
        
        for column in columns:
            if column in processed_data.columns:
                if method == 'minmax':
                    processed_data[f'{column}_normalized'] = normalize_minmax(processed_data, column)
                elif method == 'zscore':
                    processed_data[f'{column}_standardized'] = standardize_zscore(processed_data, column)
    
    return processed_data
import pandas as pd

def clean_dataset(df, columns_to_check=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    # Create a copy to avoid modifying the original DataFrame
    df_clean = df.copy()

    # Remove duplicate rows
    initial_rows = df_clean.shape[0]
    df_clean.drop_duplicates(inplace=True)
    removed_duplicates = initial_rows - df_clean.shape[0]

    # Handle missing values
    if columns_to_check is None:
        columns_to_check = df_clean.columns

    missing_counts = {}
    for col in columns_to_check:
        if col in df_clean.columns:
            missing_count = df_clean[col].isnull().sum()
            if missing_count > 0:
                # For numeric columns, fill with median
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                # For categorical columns, fill with mode
                else:
                    df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown', inplace=True)
                missing_counts[col] = missing_count

    return df_clean, removed_duplicates, missing_counts

def validate_dataframe(df, required_columns=None):
    """
    Validate that a DataFrame meets basic requirements.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    if df.empty:
        raise ValueError("DataFrame is empty")

    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    return True

# Example usage (commented out for production)
# if __name__ == "__main__":
#     # Create sample data
#     data = {
#         'id': [1, 2, 2, 3, 4, 5],
#         'name': ['Alice', 'Bob', 'Bob', None, 'Eve', None],
#         'age': [25, 30, 30, None, 35, 40],
#         'score': [85.5, 92.0, 92.0, 78.5, None, 88.0]
#     }
#     
#     df = pd.DataFrame(data)
#     print("Original DataFrame:")
#     print(df)
#     print(f"\nOriginal shape: {df.shape}")
#     
#     # Clean the data
#     cleaned_df, duplicates_removed, missing_filled = clean_dataset(df)
#     
#     print(f"\nDuplicates removed: {duplicates_removed}")
#     print(f"Missing values filled per column: {missing_filled}")
#     print("\nCleaned DataFrame:")
#     print(cleaned_df)
#     print(f"\nCleaned shape: {cleaned_df.shape}")