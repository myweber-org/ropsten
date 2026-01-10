import pandas as pd

def remove_duplicates(dataframe, subset=None, keep='first'):
    """
    Remove duplicate rows from a pandas DataFrame.
    
    Args:
        dataframe: Input DataFrame
        subset: Column label or sequence of labels to consider for duplicates
        keep: {'first', 'last', False} - Which duplicates to keep
    
    Returns:
        DataFrame with duplicates removed
    """
    if dataframe.empty:
        return dataframe
    
    cleaned_df = dataframe.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(dataframe) - len(cleaned_df)
    print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df

def clean_numeric_column(dataframe, column_name, fill_method='mean'):
    """
    Clean numeric column by handling missing values.
    
    Args:
        dataframe: Input DataFrame
        column_name: Name of column to clean
        fill_method: Method to fill missing values ('mean', 'median', 'zero')
    
    Returns:
        DataFrame with cleaned column
    """
    if column_name not in dataframe.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    df_copy = dataframe.copy()
    
    if fill_method == 'mean':
        fill_value = df_copy[column_name].mean()
    elif fill_method == 'median':
        fill_value = df_copy[column_name].median()
    elif fill_method == 'zero':
        fill_value = 0
    else:
        raise ValueError("fill_method must be 'mean', 'median', or 'zero'")
    
    missing_count = df_copy[column_name].isna().sum()
    df_copy[column_name] = df_copy[column_name].fillna(fill_value)
    
    print(f"Filled {missing_count} missing values in column '{column_name}' with {fill_method}")
    
    return df_copy

def validate_dataframe(dataframe, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        dataframe: DataFrame to validate
        required_columns: List of columns that must be present
    
    Returns:
        Boolean indicating if DataFrame is valid
    """
    if dataframe.empty:
        print("Warning: DataFrame is empty")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in dataframe.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    total_cells = dataframe.size
    null_cells = dataframe.isna().sum().sum()
    null_percentage = (null_cells / total_cells) * 100 if total_cells > 0 else 0
    
    print(f"DataFrame shape: {dataframe.shape}")
    print(f"Null values: {null_cells} ({null_percentage:.2f}%)")
    
    return null_percentage < 50