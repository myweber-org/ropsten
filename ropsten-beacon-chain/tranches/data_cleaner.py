
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

def process_numerical_columns(df, columns=None):
    """
    Process multiple numerical columns for outlier removal.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to process. If None, processes all numerical columns.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed from specified columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    result_df = df.copy()
    
    for column in columns:
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            result_df = remove_outliers_iqr(result_df, column)
    
    return result_df

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[::100, 'A'] = 500  # Add some outliers
    
    print("Original DataFrame shape:", df.shape)
    print("Original statistics for column A:", calculate_summary_statistics(df, 'A'))
    
    cleaned_df = process_numerical_columns(df, ['A', 'B'])
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Cleaned statistics for column A:", calculate_summary_statistics(cleaned_df, 'A'))import pandas as pd
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

def fill_missing_values(df, column, method='mean'):
    """
    Fill missing values in a specified column.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to fill
        method (str): 'mean', 'median', 'mode', or 'constant'
    
    Returns:
        pd.DataFrame: DataFrame with filled values
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if method == 'mean':
        fill_value = df[column].mean()
    elif method == 'median':
        fill_value = df[column].median()
    elif method == 'mode':
        fill_value = df[column].mode()[0]
    elif method == 'constant':
        fill_value = 0
    else:
        raise ValueError("Method must be 'mean', 'median', 'mode', or 'constant'")
    
    df_filled = df.copy()
    df_filled[column] = df_filled[column].fillna(fill_value)
    return df_filled

def normalize_column(df, column):
    """
    Normalize a column using min-max scaling.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to normalize
    
    Returns:
        pd.DataFrame: DataFrame with normalized column
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_normalized = df.copy()
    col_min = df_normalized[column].min()
    col_max = df_normalized[column].max()
    
    if col_max == col_min:
        df_normalized[column] = 0.5
    else:
        df_normalized[column] = (df_normalized[column] - col_min) / (col_max - col_min)
    
    return df_normalized

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
    
    df_filtered = df.copy()
    
    if method == 'iqr':
        Q1 = df_filtered[column].quantile(0.25)
        Q3 = df_filtered[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        mask = (df_filtered[column] >= lower_bound) & (df_filtered[column] <= upper_bound)
    
    elif method == 'zscore':
        mean = df_filtered[column].mean()
        std = df_filtered[column].std()
        z_scores = np.abs((df_filtered[column] - mean) / std)
        mask = z_scores <= threshold
    
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    return df_filtered[mask]

def clean_dataframe(df, operations):
    """
    Apply multiple cleaning operations to a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        operations (list): List of cleaning operations
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    for operation in operations:
        if operation['type'] == 'remove_duplicates':
            cleaned_df = remove_duplicates(cleaned_df, operation.get('subset'))
        
        elif operation['type'] == 'fill_missing':
            cleaned_df = fill_missing_values(
                cleaned_df, 
                operation['column'], 
                operation.get('method', 'mean')
            )
        
        elif operation['type'] == 'normalize':
            cleaned_df = normalize_column(cleaned_df, operation['column'])
        
        elif operation['type'] == 'filter_outliers':
            cleaned_df = filter_outliers(
                cleaned_df,
                operation['column'],
                operation.get('method', 'iqr'),
                operation.get('threshold', 1.5)
            )
    
    return cleaned_df