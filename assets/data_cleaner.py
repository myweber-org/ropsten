import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method.
    
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
    Normalize data using min-max scaling to [0, 1] range.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        DataFrame with normalized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        data[column + '_normalized'] = 0.5
    else:
        data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    
    return data.copy()

def standardize_zscore(data, column):
    """
    Standardize data using z-score normalization.
    
    Args:
        data: pandas DataFrame
        column: column name to standardize
    
    Returns:
        DataFrame with standardized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        data[column + '_standardized'] = 0
    else:
        data[column + '_standardized'] = (data[column] - mean_val) / std_val
    
    return data.copy()

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame.
    
    Args:
        data: pandas DataFrame
        strategy: 'mean', 'median', 'mode', or 'drop'
        columns: list of columns to process (None for all numeric columns)
    
    Returns:
        DataFrame with handled missing values
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    result = data.copy()
    
    for col in columns:
        if col not in result.columns:
            continue
            
        if strategy == 'drop':
            result = result.dropna(subset=[col])
        elif strategy == 'mean':
            result[col] = result[col].fillna(result[col].mean())
        elif strategy == 'median':
            result[col] = result[col].fillna(result[col].median())
        elif strategy == 'mode':
            result[col] = result[col].fillna(result[col].mode()[0] if not result[col].mode().empty else 0)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    return result

def create_sample_data():
    """
    Create sample data for testing.
    
    Returns:
        pandas DataFrame with sample data
    """
    np.random.seed(42)
    data = {
        'id': range(1, 101),
        'value': np.random.normal(100, 15, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'score': np.random.uniform(0, 100, 100)
    }
    
    df = pd.DataFrame(data)
    
    df.loc[np.random.choice(100, 5, replace=False), 'value'] = np.nan
    df.loc[np.random.choice(100, 3, replace=False), 'score'] = np.nan
    
    return df

if __name__ == "__main__":
    df = create_sample_data()
    print("Original data shape:", df.shape)
    print("\nMissing values:")
    print(df.isnull().sum())
    
    df_clean = handle_missing_values(df, strategy='mean')
    print("\nAfter handling missing values:", df_clean.shape)
    
    df_no_outliers = remove_outliers_iqr(df_clean, 'value')
    print("After removing outliers:", df_no_outliers.shape)
    
    df_normalized = normalize_minmax(df_no_outliers.copy(), 'score')
    print("\nNormalized 'score' column statistics:")
    print(df_normalized['score_normalized'].describe())
    
    df_standardized = standardize_zscore(df_no_outliers.copy(), 'value')
    print("\nStandardized 'value' column statistics:")
    print(df_standardized['value_standardized'].describe())
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

def calculate_summary_stats(df, column):
    """
    Calculate summary statistics for a column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name
    
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

def example_usage():
    """
    Example usage of the data cleaning functions.
    """
    np.random.seed(42)
    
    data = {
        'values': np.concatenate([
            np.random.normal(100, 15, 90),
            np.array([250, 300, -50, 400])
        ])
    }
    
    df = pd.DataFrame(data)
    
    print("Original DataFrame shape:", df.shape)
    print("Original summary statistics:")
    print(calculate_summary_stats(df, 'values'))
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Cleaned summary statistics:")
    print(calculate_summary_stats(cleaned_df, 'values'))
    
    return cleaned_df

if __name__ == "__main__":
    cleaned_data = example_usage()