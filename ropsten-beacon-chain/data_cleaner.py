
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    drop_duplicates (bool): Whether to remove duplicate rows
    fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    original_shape = df.shape
    
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove duplicates if requested
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
        duplicates_removed = original_shape[0] - cleaned_df.shape[0]
        print(f"Removed {duplicates_removed} duplicate rows")
    
    # Handle missing values
    missing_before = cleaned_df.isnull().sum().sum()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            mode_value = cleaned_df[col].mode()
            if not mode_value.empty:
                cleaned_df[col] = cleaned_df[col].fillna(mode_value[0])
    
    missing_after = cleaned_df.isnull().sum().sum()
    print(f"Handled {missing_before - missing_after} missing values")
    
    # Report cleaning summary
    print(f"Original shape: {original_shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    print(f"Rows removed: {original_shape[0] - cleaned_df.shape[0]}")
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the structure and content of a DataFrame.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of column names that must be present
    min_rows (int): Minimum number of rows required
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    if df.empty:
        print("Validation failed: DataFrame is empty")
        return False
    
    if len(df) < min_rows:
        print(f"Validation failed: DataFrame has fewer than {min_rows} rows")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing required columns: {missing_cols}")
            return False
    
    # Check for infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.isinf(df[col]).any():
            print(f"Validation warning: Column '{col}' contains infinite values")
    
    print("Data validation passed")
    return True

# Example usage
if __name__ == "__main__":
    # Create sample data with issues
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10.5, 20.3, 20.3, None, 40.1, 50.7],
        'category': ['A', 'B', 'B', 'C', None, 'A']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    # Validate the cleaned data
    print("\nValidating cleaned data:")
    validate_data(cleaned, required_columns=['id', 'value'], min_rows=3)
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from specified column using IQR method.
    
    Args:
        df: pandas DataFrame
        column: column name to process
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def normalize_column(df, column, method='minmax'):
    """
    Normalize specified column using given method.
    
    Args:
        df: pandas DataFrame
        column: column name to normalize
        method: 'minmax' or 'zscore'
    
    Returns:
        DataFrame with normalized column
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_copy = df.copy()
    
    if method == 'minmax':
        min_val = df_copy[column].min()
        max_val = df_copy[column].max()
        if max_val != min_val:
            df_copy[column] = (df_copy[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df_copy[column].mean()
        std_val = df_copy[column].std()
        if std_val > 0:
            df_copy[column] = (df_copy[column] - mean_val) / std_val
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    return df_copy

def handle_missing_values(df, strategy='mean'):
    """
    Handle missing values in numeric columns.
    
    Args:
        df: pandas DataFrame
        strategy: 'mean', 'median', or 'drop'
    
    Returns:
        DataFrame with handled missing values
    """
    df_copy = df.copy()
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        df_copy = df_copy.dropna(subset=numeric_cols)
    
    elif strategy == 'mean':
        for col in numeric_cols:
            if df_copy[col].isnull().any():
                df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
    
    elif strategy == 'median':
        for col in numeric_cols:
            if df_copy[col].isnull().any():
                df_copy[col] = df_copy[col].fillna(df_copy[col].median())
    
    else:
        raise ValueError("Strategy must be 'mean', 'median', or 'drop'")
    
    return df_copy

def get_data_summary(df):
    """
    Generate summary statistics for DataFrame.
    
    Args:
        df: pandas DataFrame
    
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_stats': df.describe().to_dict() if not df.select_dtypes(include=[np.number]).empty else {}
    }
    
    return summary
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_method='drop'):
    """
    Clean a pandas DataFrame by handling missing values and optionally removing duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): If True, remove duplicate rows.
    fill_method (str): Method to handle missing values. 
                       Options: 'drop' (drop rows), 'fill_mean' (fill with column mean for numeric), 
                       'fill_median' (fill with column median for numeric), 'fill_mode' (fill with mode for categorical).
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if fill_method == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_method == 'fill_mean':
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].mean())
    elif fill_method == 'fill_median':
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].median())
    elif fill_method == 'fill_mode':
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object' or cleaned_df[col].dtype.name == 'category':
                mode_val = cleaned_df[col].mode()
                if not mode_val.empty:
                    cleaned_df[col] = cleaned_df[col].fillna(mode_val.iloc[0])
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    tuple: (bool, str) indicating validation success and message.
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Data validation passed"

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, None, 4, 1],
        'B': [5, None, 7, 8, 5],
        'C': ['x', 'y', 'z', None, 'x']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame (drop missing values):")
    cleaned = clean_dataset(df, drop_duplicates=True, fill_method='drop')
    print(cleaned)
    
    is_valid, msg = validate_data(cleaned, required_columns=['A', 'B', 'C'])
    print(f"\nValidation: {is_valid} - {msg}")