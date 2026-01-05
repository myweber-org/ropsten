import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df: pandas DataFrame to clean
        drop_duplicates: If True, remove duplicate rows
        fill_missing: If True, fill missing values
        fill_value: Value to use for filling missing data
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate a dataset by checking for required columns and data types.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: List of column names that must be present
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    return True, "Dataset is valid"

def remove_outliers(df, column, threshold=3):
    """
    Remove outliers from a specific column using z-score method.
    
    Args:
        df: pandas DataFrame
        column: Column name to check for outliers
        threshold: Z-score threshold for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    from scipy import stats
    import numpy as np
    
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    filtered_entries = z_scores < threshold
    return df[filtered_entries]import csv
import re

def clean_csv(input_file, output_file, columns_to_clean=None):
    """
    Clean a CSV file by removing extra whitespace and standardizing text.
    """
    cleaned_rows = []
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        for row in reader:
            cleaned_row = {}
            for field in fieldnames:
                value = row[field]
                if value and isinstance(value, str):
                    # Remove extra whitespace
                    value = re.sub(r'\s+', ' ', value.strip())
                    # Convert to title case for name fields
                    if 'name' in field.lower():
                        value = value.title()
                cleaned_row[field] = value
            cleaned_rows.append(cleaned_row)
    
    with open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned_rows)
    
    return len(cleaned_rows)

def validate_email(email):
    """
    Validate email format using regex.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email)) if email else False

if __name__ == "__main__":
    # Example usage
    count = clean_csv('raw_data.csv', 'cleaned_data.csv')
    print(f"Cleaned {count} rows of data.")
import pandas as pd
import numpy as np

def clean_dataset(df, columns_to_check=None, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by handling duplicates and missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    columns_to_check (list): Specific columns to check for duplicates, None for all columns
    drop_duplicates (bool): Whether to drop duplicate rows
    fill_missing (str): Strategy to fill missing values - 'mean', 'median', 'mode', or 'drop'
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates(subset=columns_to_check, keep='first')
    
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
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown')
    
    return df_clean

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of columns that must be present
    min_rows (int): Minimum number of rows required
    
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
    
    return True, "Data validation passed"

def remove_outliers(df, column, method='iqr', threshold=1.5):
    """
    Remove outliers from a specific column using specified method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to check for outliers
    method (str): 'iqr' for interquartile range or 'zscore' for standard deviations
    threshold (float): Threshold value for outlier detection
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' must be numeric")
    
    df_copy = df.copy()
    
    if method == 'iqr':
        Q1 = df_copy[column].quantile(0.25)
        Q3 = df_copy[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        df_copy = df_copy[(df_copy[column] >= lower_bound) & (df_copy[column] <= upper_bound)]
    
    elif method == 'zscore':
        from scipy import stats
        z_scores = np.abs(stats.zscore(df_copy[column].dropna()))
        df_copy = df_copy[z_scores < threshold]
    
    return df_copy

def normalize_data(df, columns=None, method='minmax'):
    """
    Normalize specified columns in DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): Columns to normalize, None for all numeric columns
    method (str): Normalization method - 'minmax' or 'standard'
    
    Returns:
    pd.DataFrame: DataFrame with normalized columns
    """
    df_norm = df.copy()
    
    if columns is None:
        columns = df_norm.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df_norm.columns and pd.api.types.is_numeric_dtype(df_norm[col]):
            if method == 'minmax':
                min_val = df_norm[col].min()
                max_val = df_norm[col].max()
                if max_val > min_val:
                    df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
            elif method == 'standard':
                mean_val = df_norm[col].mean()
                std_val = df_norm[col].std()
                if std_val > 0:
                    df_norm[col] = (df_norm[col] - mean_val) / std_val
    
    return df_norm

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5, 6],
        'value': [10, 20, None, 40, 50, 50, 1000],
        'category': ['A', 'B', 'C', None, 'A', 'A', 'D']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned_df = clean_dataset(df, fill_missing='mean')
    print(cleaned_df)
    
    is_valid, message = validate_data(cleaned_df, required_columns=['id', 'value'])
    print(f"\nValidation: {is_valid} - {message}")
    
    df_no_outliers = remove_outliers(cleaned_df, 'value', method='iqr')
    print(f"\nDataFrame after removing outliers: {len(df_no_outliers)} rows")
    
    normalized_df = normalize_data(df_no_outliers, columns=['value'])
    print("\nNormalized DataFrame:")
    print(normalized_df)
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
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

def clean_dataset(df, numeric_columns=None):
    """
    Clean dataset by removing outliers from all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of numeric column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns:
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, column)
            except Exception as e:
                print(f"Error cleaning column {column}: {e}")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'id': range(1, 101),
        'value': np.random.randn(100) * 10 + 50,
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(sample_data)
    print(f"Original shape: {df.shape}")
    
    cleaned_df = clean_dataset(df, ['value'])
    print(f"Cleaned shape: {cleaned_df.shape}")
    print(f"Outliers removed: {len(df) - len(cleaned_df)}")