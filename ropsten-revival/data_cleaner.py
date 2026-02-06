import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df: pandas DataFrame to clean
        drop_duplicates: If True, remove duplicate rows
        fill_missing: Strategy for handling missing values. 
                     Can be 'mean', 'median', 'mode', or a specific value.
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing is not None:
        missing_count = cleaned_df.isnull().sum().sum()
        if missing_count > 0:
            if fill_missing == 'mean':
                cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
            elif fill_missing == 'median':
                cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
            elif fill_missing == 'mode':
                cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
            else:
                cleaned_df = cleaned_df.fillna(fill_missing)
            print(f"Filled {missing_count} missing values using {fill_missing} strategy")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate that a DataFrame meets basic requirements.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: List of column names that must be present
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"import csv
import re
from typing import List, Optional

def remove_duplicates(data: List[List[str]]) -> List[List[str]]:
    seen = set()
    unique_data = []
    for row in data:
        row_tuple = tuple(row)
        if row_tuple not in seen:
            seen.add(row_tuple)
            unique_data.append(row)
    return unique_data

def normalize_string(value: str) -> str:
    normalized = re.sub(r'\s+', ' ', value.strip())
    return normalized.lower()

def clean_csv(input_path: str, output_path: str, columns_to_clean: Optional[List[int]] = None):
    with open(input_path, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        data = list(reader)
    
    if not data:
        return
    
    headers = data[0]
    rows = data[1:]
    
    cleaned_rows = []
    for row in rows:
        cleaned_row = row.copy()
        if columns_to_clean:
            for col_index in columns_to_clean:
                if col_index < len(cleaned_row):
                    cleaned_row[col_index] = normalize_string(cleaned_row[col_index])
        cleaned_rows.append(cleaned_row)
    
    cleaned_rows = remove_duplicates(cleaned_rows)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(headers)
        writer.writerows(cleaned_rows)
    
    print(f"Cleaned data saved to {output_path}")
    print(f"Removed {len(rows) - len(cleaned_rows)} duplicate rows")

def validate_email(email: str) -> bool:
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

if __name__ == "__main__":
    sample_data = [
        ["Name", "Email", "Age"],
        ["John Doe", "john@example.com", "25"],
        ["Jane Smith", "jane@example.com", "30"],
        ["John Doe", "john@example.com", "25"],
        ["Bob Johnson", "bob@example.com", "35"]
    ]
    
    test_file = "sample_data.csv"
    output_file = "cleaned_data.csv"
    
    with open(test_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(sample_data)
    
    clean_csv(test_file, output_file, columns_to_clean=[0, 1])
    
    test_email = "test@example.com"
    print(f"Email validation for {test_email}: {validate_email(test_email)}")def remove_duplicates(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

if __name__ == "__main__":
    sample_data = [1, 2, 2, 3, 4, 4, 5]
    cleaned = remove_duplicates(sample_data)
    print(f"Original: {sample_data}")
    print(f"Cleaned: {cleaned}")
import pandas as pd
import re

def clean_dataframe(df, columns_to_clean=None, remove_duplicates=True, case_normalization='lower'):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing string columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    columns_to_clean (list, optional): List of column names to apply string normalization.
                                       If None, all object dtype columns are cleaned.
    remove_duplicates (bool): If True, remove duplicate rows.
    case_normalization (str): One of 'lower', 'upper', or None for string normalization.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if remove_duplicates:
        initial_rows = cleaned_df.shape[0]
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - cleaned_df.shape[0]
        print(f"Removed {removed} duplicate rows.")
    
    if columns_to_clean is None:
        columns_to_clean = cleaned_df.select_dtypes(include=['object']).columns.tolist()
    
    for col in columns_to_clean:
        if col in cleaned_df.columns and cleaned_df[col].dtype == 'object':
            if case_normalization == 'lower':
                cleaned_df[col] = cleaned_df[col].astype(str).str.lower()
            elif case_normalization == 'upper':
                cleaned_df[col] = cleaned_df[col].astype(str).str.upper()
            
            cleaned_df[col] = cleaned_df[col].apply(lambda x: re.sub(r'\s+', ' ', str(x)).strip())
    
    return cleaned_df

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    email_column (str): Name of the column containing email addresses.
    
    Returns:
    pd.DataFrame: DataFrame with an additional 'email_valid' boolean column.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame.")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    df['email_valid'] = df[email_column].astype(str).str.match(email_pattern)
    
    valid_count = df['email_valid'].sum()
    total_count = len(df)
    print(f"Valid emails: {valid_count}/{total_count} ({valid_count/total_count*100:.2f}%)")
    
    return df

if __name__ == "__main__":
    sample_data = {
        'name': ['John Doe', 'Jane Smith', 'John Doe', 'ALICE BROWN', 'bob  wilson  '],
        'email': ['john@example.com', 'jane@test.org', 'invalid-email', 'alice@company.co', 'bob@domain.net'],
        'age': [25, 30, 25, 35, 28]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaned = clean_dataframe(df, columns_to_clean=['name'], remove_duplicates=True)
    print("Cleaned DataFrame:")
    print(cleaned)
    print("\n")
    
    validated = validate_email_column(cleaned, 'email')
    print("DataFrame with email validation:")
    print(validated)import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (str or dict): Method to fill missing values. Can be 'mean', 'median', 
                                'mode', or a dictionary of column:value pairs. Default is None.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
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
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if df.empty:
        print("Validation failed: DataFrame is empty")
        return False
    
    if len(df) < min_rows:
        print(f"Validation failed: DataFrame has fewer than {min_rows} rows")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Validation failed: Missing required columns: {missing_columns}")
            return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, None, 5],
        'B': [10, None, 30, 40, 50],
        'C': ['x', 'y', 'y', 'z', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid = validate_data(cleaned, required_columns=['A', 'B'], min_rows=3)
    print(f"\nData validation result: {is_valid}")import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (str): Method to fill missing values. Options: 'mean', 'median', 'mode', or 'drop'. Default is 'mean'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            elif fill_missing == 'median':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate a DataFrame for required columns and data types.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names. Default is None.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return False
    
    if df.empty:
        print("DataFrame is empty")
        return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, None, 5],
        'B': [10, None, 30, 40, 50],
        'C': ['x', 'y', 'y', 'z', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid = validate_dataset(cleaned, required_columns=['A', 'B'])
    print(f"\nDataset validation: {is_valid}")
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to clean.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
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

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to analyze.
    
    Returns:
    dict: Dictionary containing summary statistics.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def example_usage():
    """
    Example usage of the data cleaning functions.
    """
    np.random.seed(42)
    data = {
        'id': range(1, 101),
        'value': np.random.normal(100, 15, 100)
    }
    
    df = pd.DataFrame(data)
    
    print("Original DataFrame shape:", df.shape)
    print("Original statistics:", calculate_summary_statistics(df, 'value'))
    
    cleaned_df = remove_outliers_iqr(df, 'value')
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Cleaned statistics:", calculate_summary_statistics(cleaned_df, 'value'))
    
    return cleaned_df

if __name__ == "__main__":
    cleaned_data = example_usage()
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specific column using the Interquartile Range method.
    
    Args:
        data: numpy array or list of numerical values
        column: index of column to process (if data is 2D)
        
    Returns:
        Cleaned data with outliers removed
    """
    if isinstance(data, list):
        data = np.array(data)
    
    # Extract column if data is 2D
    if data.ndim == 2:
        column_data = data[:, column]
    else:
        column_data = data
    
    # Calculate Q1, Q3 and IQR
    Q1 = np.percentile(column_data, 25)
    Q3 = np.percentile(column_data, 75)
    IQR = Q3 - Q1
    
    # Define outlier bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter outliers
    if data.ndim == 2:
        mask = (column_data >= lower_bound) & (column_data <= upper_bound)
        cleaned_data = data[mask]
    else:
        cleaned_data = column_data[(column_data >= lower_bound) & (column_data <= upper_bound)]
    
    return cleaned_data

def calculate_statistics(data):
    """
    Calculate basic statistics for the data.
    
    Args:
        data: numpy array of numerical values
        
    Returns:
        Dictionary containing mean, median, std, min, max
    """
    stats = {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'count': len(data)
    }
    return stats

def normalize_data(data, method='minmax'):
    """
    Normalize data using specified method.
    
    Args:
        data: numpy array of numerical values
        method: 'minmax' or 'zscore'
        
    Returns:
        Normalized data
    """
    if method == 'minmax':
        data_min = np.min(data)
        data_max = np.max(data)
        if data_max - data_min == 0:
            return np.zeros_like(data)
        return (data - data_min) / (data_max - data_min)
    
    elif method == 'zscore':
        data_mean = np.mean(data)
        data_std = np.std(data)
        if data_std == 0:
            return np.zeros_like(data)
        return (data - data_mean) / data_std
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")

def main():
    # Example usage
    sample_data = np.random.normal(100, 15, 1000)
    print("Original data statistics:")
    print(calculate_statistics(sample_data))
    
    cleaned_data = remove_outliers_iqr(sample_data, 0)
    print("\nCleaned data statistics:")
    print(calculate_statistics(cleaned_data))
    
    normalized_data = normalize_data(cleaned_data, method='zscore')
    print("\nNormalized data statistics:")
    print(calculate_statistics(normalized_data))

if __name__ == "__main__":
    main()