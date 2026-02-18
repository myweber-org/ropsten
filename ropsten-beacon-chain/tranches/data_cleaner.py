import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.

    Args:
        df (pd.DataFrame): The input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
        fill_missing (str or dict, optional): Method to fill missing values.
            Can be 'ffill', 'bfill', a scalar value, or a dict mapping columns to values.
            If None, missing values are not filled.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    cleaned_df = df.copy()

    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()

    if fill_missing is not None:
        if isinstance(fill_missing, dict):
            cleaned_df = cleaned_df.fillna(fill_missing)
        else:
            cleaned_df = cleaned_df.fillna(method=fill_missing)

    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for required columns and basic integrity.

    Args:
        df (pd.DataFrame): The DataFrame to validate.
        required_columns (list, optional): List of column names that must be present.

    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"

    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"

    return True, "DataFrame is valid"