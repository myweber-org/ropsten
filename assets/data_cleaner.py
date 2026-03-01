
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_strategy='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to remove duplicate rows.
    fill_missing (bool): Whether to fill missing values.
    fill_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', or 'constant').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        for column in cleaned_df.columns:
            if cleaned_df[column].dtype in [np.float64, np.int64]:
                if fill_strategy == 'mean':
                    cleaned_df[column].fillna(cleaned_df[column].mean(), inplace=True)
                elif fill_strategy == 'median':
                    cleaned_df[column].fillna(cleaned_df[column].median(), inplace=True)
                elif fill_strategy == 'constant':
                    cleaned_df[column].fillna(0, inplace=True)
            elif cleaned_df[column].dtype == 'object':
                if fill_strategy == 'mode':
                    cleaned_df[column].fillna(cleaned_df[column].mode()[0], inplace=True)
                elif fill_strategy == 'constant':
                    cleaned_df[column].fillna('Unknown', inplace=True)
    
    return cleaned_df

def validate_dataset(df, check_missing=True, check_types=True):
    """
    Validate a pandas DataFrame for common data quality issues.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    check_missing (bool): Check for missing values.
    check_types (bool): Check for consistent data types.
    
    Returns:
    dict: Dictionary containing validation results.
    """
    validation_results = {}
    
    if check_missing:
        missing_counts = df.isnull().sum()
        validation_results['missing_values'] = missing_counts[missing_counts > 0].to_dict()
    
    if check_types:
        type_summary = {}
        for column in df.columns:
            type_summary[column] = str(df[column].dtype)
        validation_results['data_types'] = type_summary
    
    validation_results['shape'] = df.shape
    validation_results['columns'] = list(df.columns)
    
    return validation_results

def normalize_columns(df, columns=None, method='minmax'):
    """
    Normalize specified columns in a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns (list): List of columns to normalize. If None, normalize all numeric columns.
    method (str): Normalization method ('minmax' or 'zscore').
    
    Returns:
    pd.DataFrame: DataFrame with normalized columns.
    """
    normalized_df = df.copy()
    
    if columns is None:
        numeric_cols = normalized_df.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    for column in columns:
        if column in normalized_df.columns and normalized_df[column].dtype in [np.float64, np.int64]:
            if method == 'minmax':
                min_val = normalized_df[column].min()
                max_val = normalized_df[column].max()
                if max_val > min_val:
                    normalized_df[column] = (normalized_df[column] - min_val) / (max_val - min_val)
            elif method == 'zscore':
                mean_val = normalized_df[column].mean()
                std_val = normalized_df[column].std()
                if std_val > 0:
                    normalized_df[column] = (normalized_df[column] - mean_val) / std_val
    
    return normalized_df

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, 2, 4, 5, None],
        'B': [10.5, 20.3, 20.3, 40.1, 50.0, 60.2],
        'C': ['apple', 'banana', 'banana', 'cherry', None, 'elderberry']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaned = clean_dataset(df, fill_strategy='mean')
    print("Cleaned DataFrame:")
    print(cleaned)
    print("\n")
    
    validation = validate_dataset(cleaned)
    print("Validation Results:")
    print(validation)
    print("\n")
    
    normalized = normalize_columns(cleaned, method='minmax')
    print("Normalized DataFrame:")
    print(normalized)