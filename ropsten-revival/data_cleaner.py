import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method.
    
    Parameters:
    data: pandas DataFrame
    column: str, column name
    factor: float, IQR multiplier
    
    Returns:
    pandas DataFrame without outliers
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def normalize_minmax(data, column):
    """
    Normalize data using min-max scaling.
    
    Parameters:
    data: pandas DataFrame
    column: str, column name
    
    Returns:
    pandas Series with normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return pd.Series([0] * len(data), index=data.index)
    
    return (data[column] - min_val) / (max_val - min_val)

def standardize_zscore(data, column):
    """
    Standardize data using z-score normalization.
    
    Parameters:
    data: pandas DataFrame
    column: str, column name
    
    Returns:
    pandas Series with standardized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    return (data[column] - mean_val) / std_val

def clean_dataset(df, numeric_columns=None, outlier_factor=1.5):
    """
    Clean dataset by removing outliers and normalizing numeric columns.
    
    Parameters:
    df: pandas DataFrame
    numeric_columns: list, numeric column names to process
    outlier_factor: float, IQR multiplier for outlier detection
    
    Returns:
    Cleaned pandas DataFrame
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col, outlier_factor)
            cleaned_df[col + '_normalized'] = normalize_minmax(cleaned_df, col)
            cleaned_df[col + '_standardized'] = standardize_zscore(cleaned_df, col)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate dataframe structure and content.
    
    Parameters:
    df: pandas DataFrame
    required_columns: list, required column names
    min_rows: int, minimum number of rows
    
    Returns:
    bool: True if valid, False otherwise
    """
    if not isinstance(df, pd.DataFrame):
        return False
    
    if len(df) < min_rows:
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False
    
    return True

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    print("Original data shape:", sample_data.shape)
    print("Original data summary:")
    print(sample_data.describe())
    
    cleaned_data = clean_dataset(sample_data, ['feature1', 'feature2'])
    
    print("\nCleaned data shape:", cleaned_data.shape)
    print("Cleaned data summary:")
    print(cleaned_data[['feature1', 'feature2']].describe())
    
    is_valid = validate_dataframe(cleaned_data, min_rows=10)
    print(f"\nData validation: {'Passed' if is_valid else 'Failed'}")