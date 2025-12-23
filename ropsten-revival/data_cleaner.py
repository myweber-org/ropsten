
import pandas as pd

def clean_dataframe(df, drop_na=True, rename_columns=None):
    """
    Clean a pandas DataFrame by removing null values and optionally renaming columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_na (bool): If True, drop rows with any null values. Default is True.
        rename_columns (dict): Dictionary mapping old column names to new names. Default is None.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_na:
        cleaned_df = cleaned_df.dropna()
    
    if rename_columns:
        cleaned_df = cleaned_df.rename(columns=rename_columns)
    
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def standardize_numeric_columns(df, columns):
    """
    Standardize numeric columns by subtracting mean and dividing by standard deviation.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list): List of column names to standardize.
    
    Returns:
        pd.DataFrame: DataFrame with standardized numeric columns.
    """
    standardized_df = df.copy()
    
    for col in columns:
        if col in standardized_df.columns and pd.api.types.is_numeric_dtype(standardized_df[col]):
            mean = standardized_df[col].mean()
            std = standardized_df[col].std()
            if std > 0:
                standardized_df[col] = (standardized_df[col] - mean) / std
    
    return standardized_df

def remove_outliers(df, column, method='iqr', threshold=1.5):
    """
    Remove outliers from a DataFrame column using specified method.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to process.
        method (str): Method for outlier detection ('iqr' or 'zscore'). Default is 'iqr'.
        threshold (float): Threshold for outlier detection. Default is 1.5 for IQR.
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        return df
    
    data = df[column].dropna()
    
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)
    
    elif method == 'zscore':
        mean = data.mean()
        std = data.std()
        if std > 0:
            z_scores = abs((df[column] - mean) / std)
            mask = z_scores <= threshold
        else:
            mask = pd.Series(True, index=df.index)
    
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    return df[mask].reset_index(drop=True)