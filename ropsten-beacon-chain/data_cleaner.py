import pandas as pd
import numpy as np

def clean_dataframe(df, missing_strategy='mean', outlier_threshold=3):
    """
    Clean a pandas DataFrame by handling missing values and outliers.

    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    missing_strategy (str): Strategy for handling missing values.
        Options: 'mean', 'median', 'mode', 'drop'.
    outlier_threshold (float): Number of standard deviations to consider a point an outlier.

    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()

    # Handle missing values
    if missing_strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    else:
        for column in cleaned_df.select_dtypes(include=[np.number]).columns:
            if cleaned_df[column].isnull().any():
                if missing_strategy == 'mean':
                    fill_value = cleaned_df[column].mean()
                elif missing_strategy == 'median':
                    fill_value = cleaned_df[column].median()
                elif missing_strategy == 'mode':
                    fill_value = cleaned_df[column].mode()[0]
                else:
                    raise ValueError(f"Unsupported missing_strategy: {missing_strategy}")
                cleaned_df[column].fillna(fill_value, inplace=True)

    # Handle outliers for numeric columns
    numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        mean = cleaned_df[column].mean()
        std = cleaned_df[column].std()
        if std > 0:  # Avoid division by zero
            z_scores = np.abs((cleaned_df[column] - mean) / std)
            cleaned_df = cleaned_df[z_scores < outlier_threshold]

    return cleaned_df.reset_index(drop=True)

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    subset (list): Columns to consider for identifying duplicates.
    keep (str): Which duplicates to keep.

    Returns:
    pd.DataFrame: DataFrame with duplicates removed.
    """
    return df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)

def normalize_columns(df, columns=None, method='minmax'):
    """
    Normalize specified columns in DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns (list): Columns to normalize. If None, normalize all numeric columns.
    method (str): Normalization method. Options: 'minmax', 'zscore'.

    Returns:
    pd.DataFrame: DataFrame with normalized columns.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns

    normalized_df = df.copy()
    for column in columns:
        if column in normalized_df.columns and pd.api.types.is_numeric_dtype(normalized_df[column]):
            if method == 'minmax':
                min_val = normalized_df[column].min()
                max_val = normalized_df[column].max()
                if max_val > min_val:
                    normalized_df[column] = (normalized_df[column] - min_val) / (max_val - min_val)
            elif method == 'zscore':
                mean = normalized_df[column].mean()
                std = normalized_df[column].std()
                if std > 0:
                    normalized_df[column] = (normalized_df[column] - mean) / std
            else:
                raise ValueError(f"Unsupported normalization method: {method}")

    return normalized_df