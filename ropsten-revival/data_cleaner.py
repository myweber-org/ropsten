
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    filtered_indices = np.where(z_scores < threshold)[0]
    
    return data.iloc[filtered_indices]

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

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(df, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col not in cleaned_df.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        elif outlier_method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
        
        if normalize_method == 'minmax':
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
        elif normalize_method == 'zscore':
            cleaned_df[col] = normalize_zscore(cleaned_df, col)
    
    return cleaned_df

def validate_data(df, required_columns=None, allow_nan=False):
    """
    Validate dataset structure and content
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    if not allow_nan:
        nan_count = df.isnull().sum().sum()
        if nan_count > 0:
            raise ValueError(f"Dataset contains {nan_count} NaN values")
    
    return True
import pandas as pd

def clean_dataset(df, columns_to_check=None, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        columns_to_check (list, optional): List of column names to check for duplicates.
            If None, checks all columns. Defaults to None.
        fill_missing (str, optional): Method to fill missing values.
            Options: 'mean', 'median', 'mode', or 'drop'. Defaults to 'mean'.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    # Remove duplicates
    if columns_to_check is None:
        df_clean = df_clean.drop_duplicates()
    else:
        df_clean = df_clean.drop_duplicates(subset=columns_to_check)
    
    # Handle missing values
    if fill_missing == 'drop':
        df_clean = df_clean.dropna()
    else:
        for column in df_clean.columns:
            if df_clean[column].dtype in ['int64', 'float64']:
                if fill_missing == 'mean':
                    df_clean[column].fillna(df_clean[column].mean(), inplace=True)
                elif fill_missing == 'median':
                    df_clean[column].fillna(df_clean[column].median(), inplace=True)
                elif fill_missing == 'mode':
                    df_clean[column].fillna(df_clean[column].mode()[0], inplace=True)
            else:
                # For non-numeric columns, fill with mode or empty string
                if fill_missing == 'mode':
                    df_clean[column].fillna(df_clean[column].mode()[0], inplace=True)
                else:
                    df_clean[column].fillna('', inplace=True)
    
    return df_clean

def validate_dataset(df, required_columns=None):
    """
    Validate a DataFrame for required columns and data types.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list, optional): List of required column names.
    
    Returns:
        tuple: (bool, str) indicating validation result and message.
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    # Check for any remaining NaN values
    if df.isnull().any().any():
        return False, "Dataset contains NaN values after cleaning"
    
    return True, "Dataset validation passed"

# Example usage (commented out for production)
# if __name__ == "__main__":
#     # Create sample data
#     data = {
#         'id': [1, 2, 2, 3, 4, None],
#         'value': [10, 20, 20, None, 40, 50],
#         'category': ['A', 'B', 'B', 'C', None, 'A']
#     }
#     df = pd.DataFrame(data)
#     
#     # Clean the data
#     cleaned_df = clean_dataset(df, columns_to_check=['id'], fill_missing='mean')
#     
#     # Validate the cleaned data
#     is_valid, message = validate_dataset(cleaned_df, required_columns=['id', 'value'])
#     print(f"Validation: {is_valid}, Message: {message}")
#     print(cleaned_df)