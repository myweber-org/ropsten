
import pandas as pd

def clean_dataframe(df, column_name, condition_func):
    """
    Filters a pandas DataFrame based on a condition applied to a specific column.
    Removes rows where the condition function returns False.

    Args:
        df (pd.DataFrame): The input DataFrame to clean.
        column_name (str): The name of the column to apply the condition to.
        condition_func (function): A function that takes a single value from the
                                   specified column and returns a boolean.

    Returns:
        pd.DataFrame: A new DataFrame with rows filtered based on the condition.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")

    filtered_df = df[df[column_name].apply(condition_func)].copy()
    filtered_df.reset_index(drop=True, inplace=True)
    return filtered_df

def remove_outliers_iqr(df, column_name):
    """
    Removes outliers from a specified column in a DataFrame using the IQR method.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the numeric column to process.

    Returns:
        pd.DataFrame: A new DataFrame with outliers removed from the specified column.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")

    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    filtered_df = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)].copy()
    filtered_df.reset_index(drop=True, inplace=True)
    return filtered_df
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, threshold=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    outliers_removed = len(data) - len(filtered_data)
    
    return filtered_data, outliers_removed

def normalize_minmax(data, column):
    """
    Normalize data using min-max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using z-score normalization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in specified columns
    """
    if columns is None:
        columns = data.columns
    
    data_clean = data.copy()
    
    for column in columns:
        if column not in data.columns:
            continue
            
        if data[column].isnull().any():
            if strategy == 'mean':
                fill_value = data[column].mean()
            elif strategy == 'median':
                fill_value = data[column].median()
            elif strategy == 'mode':
                fill_value = data[column].mode()[0]
            elif strategy == 'drop':
                data_clean = data_clean.dropna(subset=[column])
                continue
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            data_clean[column] = data_clean[column].fillna(fill_value)
    
    return data_clean

def validate_data_types(data, expected_types):
    """
    Validate that columns have expected data types
    """
    validation_results = {}
    
    for column, expected_type in expected_types.items():
        if column not in data.columns:
            validation_results[column] = {'status': 'missing', 'actual': None}
            continue
        
        actual_type = data[column].dtype
        is_valid = actual_type == expected_type
        
        validation_results[column] = {
            'status': 'valid' if is_valid else 'invalid',
            'expected': expected_type,
            'actual': actual_type
        }
    
    return validation_results