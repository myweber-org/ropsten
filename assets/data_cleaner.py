import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fillna_method=None):
    """
    Clean a pandas DataFrame by handling missing values and duplicates.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows. Default True.
        fillna_method (str or None): Method to fill missing values. 
            Options: 'ffill', 'bfill', 'mean', 'median', or None to drop rows.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    # Handle missing values
    if fillna_method is None:
        df_clean = df_clean.dropna()
    elif fillna_method in ['ffill', 'bfill']:
        df_clean = df_clean.fillna(method=fillna_method)
    elif fillna_method == 'mean':
        df_clean = df_clean.fillna(df_clean.mean(numeric_only=True))
    elif fillna_method == 'median':
        df_clean = df_clean.fillna(df_clean.median(numeric_only=True))
    
    # Remove duplicates
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates()
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
    
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