import pandas as pd

def clean_dataset(df, columns=None, drop_duplicates=True, fill_na_method='drop'):
    """
    Clean a pandas DataFrame by handling missing values and removing duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    columns (list): Specific columns to clean. If None, clean all columns.
    drop_duplicates (bool): Whether to remove duplicate rows.
    fill_na_method (str): Method to handle missing values ('drop', 'mean', 'median', 'mode', 'zero').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.columns.tolist()
    
    for col in columns:
        if col in df_clean.columns:
            if fill_na_method == 'drop':
                df_clean = df_clean.dropna(subset=[col])
            elif fill_na_method == 'mean':
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
            elif fill_na_method == 'median':
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            elif fill_na_method == 'mode':
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else None)
            elif fill_na_method == 'zero':
                df_clean[col] = df_clean[col].fillna(0)
    
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates()
    
    return df_clean

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"