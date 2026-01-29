
import pandas as pd

def clean_dataframe(df):
    """
    Clean a pandas DataFrame by removing null values and duplicates.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    cleaned_df = df.copy()
    
    cleaned_df = cleaned_df.dropna()
    
    cleaned_df = cleaned_df.drop_duplicates()
    
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and required columns.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
    
    Returns:
        bool: True if validation passes, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        return False
    
    if df.empty:
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'name': ['Alice', 'Bob', 'Charlie', None, 'Alice'],
        'age': [25, 30, None, 40, 25],
        'score': [85.5, 92.0, 78.5, 88.0, 85.5]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nDataFrame shape:", df.shape)
    
    cleaned = clean_dataframe(df)
    print("\nCleaned DataFrame:")
    print(cleaned)
    print("\nCleaned shape:", cleaned.shape)
    
    is_valid = validate_dataframe(cleaned, ['name', 'age', 'score'])
    print(f"\nDataFrame validation: {is_valid}")import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df: pandas DataFrame to clean
        drop_duplicates: If True, remove duplicate rows
        fill_missing: Strategy for filling missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        Cleaned pandas DataFrame
    """
    df_clean = df.copy()
    
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates()
    
    if fill_missing == 'drop':
        df_clean = df_clean.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    elif fill_missing == 'mode':
        for col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else None)
    
    return df_clean

def remove_outliers(df, columns=None, method='iqr', threshold=1.5):
    """
    Remove outliers from specified columns using IQR or Z-score method.
    
    Args:
        df: pandas DataFrame
        columns: List of column names to process (None for all numeric columns)
        method: 'iqr' for interquartile range or 'zscore' for standard deviations
        threshold: Multiplier for IQR or cutoff for Z-score
    
    Returns:
        DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    
    for col in columns:
        if method == 'iqr':
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        elif method == 'zscore':
            z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
            df_clean = df_clean[z_scores < threshold]
    
    return df_clean

def normalize_data(df, columns=None, method='minmax'):
    """
    Normalize specified columns using min-max or standard scaling.
    
    Args:
        df: pandas DataFrame
        columns: List of column names to normalize (None for all numeric columns)
        method: 'minmax' for 0-1 scaling or 'standard' for z-score normalization
    
    Returns:
        DataFrame with normalized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_normalized = df.copy()
    
    for col in columns:
        if method == 'minmax':
            min_val = df_normalized[col].min()
            max_val = df_normalized[col].max()
            if max_val > min_val:
                df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
        elif method == 'standard':
            mean_val = df_normalized[col].mean()
            std_val = df_normalized[col].std()
            if std_val > 0:
                df_normalized[col] = (df_normalized[col] - mean_val) / std_val
    
    return df_normalized