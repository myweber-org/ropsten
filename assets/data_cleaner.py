
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
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
    
    return filtered_df.reset_index(drop=True)

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    If no columns specified, processes all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[col]):
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{col}'")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return True
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    return True

def example_usage():
    """
    Example demonstrating the data cleaning functions.
    """
    np.random.seed(42)
    
    data = {
        'id': range(100),
        'value': np.random.normal(100, 15, 100),
        'temperature': np.random.normal(25, 5, 100)
    }
    
    df = pd.DataFrame(data)
    
    df.loc[10, 'value'] = 500
    df.loc[20, 'value'] = -100
    df.loc[30, 'temperature'] = 100
    
    print("Original DataFrame shape:", df.shape)
    print("\nDataFrame info:")
    print(df.info())
    
    if validate_dataframe(df, ['id', 'value', 'temperature']):
        cleaned_df = clean_numeric_data(df, ['value', 'temperature'])
        print("\nCleaned DataFrame shape:", cleaned_df.shape)
        
        print("\nSummary statistics - Original:")
        print(df[['value', 'temperature']].describe())
        
        print("\nSummary statistics - Cleaned:")
        print(cleaned_df[['value', 'temperature']].describe())
        
        return cleaned_df
    
    return None

if __name__ == "__main__":
    result = example_usage()import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        subset (list, optional): Column labels to consider for duplicates.
        keep (str, optional): Which duplicates to keep.
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    return cleaned_df

def validate_dataframe(df):
    """
    Perform basic validation on DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
    
    Returns:
        tuple: (is_valid, message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    return True, "DataFrame is valid"

def clean_numeric_column(df, column_name, fill_method='mean'):
    """
    Clean numeric column by handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column_name (str): Name of column to clean.
        fill_method (str): Method to fill missing values.
    
    Returns:
        pd.DataFrame: DataFrame with cleaned column.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    if not pd.api.types.is_numeric_dtype(df[column_name]):
        raise ValueError(f"Column '{column_name}' is not numeric")
    
    df_clean = df.copy()
    
    if fill_method == 'mean':
        fill_value = df_clean[column_name].mean()
    elif fill_method == 'median':
        fill_value = df_clean[column_name].median()
    elif fill_method == 'zero':
        fill_value = 0
    else:
        raise ValueError(f"Unsupported fill method: {fill_method}")
    
    df_clean[column_name] = df_clean[column_name].fillna(fill_value)
    return df_clean