
import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.

    Parameters:
    df (pd.DataFrame): The input DataFrame to clean.
    drop_duplicates (bool): If True, remove duplicate rows.
    fill_missing (str or dict): Method to fill missing values.
        If a string, it can be 'mean', 'median', 'mode', or a constant value.
        If a dict, specify column-wise fill values.

    Returns:
    pd.DataFrame: The cleaned DataFrame.
    """
    cleaned_df = df.copy()

    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()

    if fill_missing is not None:
        if isinstance(fill_missing, dict):
            cleaned_df = cleaned_df.fillna(fill_missing)
        elif fill_missing == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_missing == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif fill_missing == 'mode':
            cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
        else:
            cleaned_df = cleaned_df.fillna(fill_missing)

    return cleaned_df

def normalize_column(df, column, method='minmax'):
    """
    Normalize a specific column in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to normalize.
    method (str): Normalization method ('minmax' or 'zscore').

    Returns:
    pd.DataFrame: DataFrame with the normalized column.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    normalized_df = df.copy()

    if method == 'minmax':
        min_val = normalized_df[column].min()
        max_val = normalized_df[column].max()
        if max_val != min_val:
            normalized_df[column] = (normalized_df[column] - min_val) / (max_val - min_val)
        else:
            normalized_df[column] = 0
    elif method == 'zscore':
        mean_val = normalized_df[column].mean()
        std_val = normalized_df[column].std()
        if std_val != 0:
            normalized_df[column] = (normalized_df[column] - mean_val) / std_val
        else:
            normalized_df[column] = 0
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'.")

    return normalized_df