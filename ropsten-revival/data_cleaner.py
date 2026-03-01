
import pandas as pd
import numpy as np

def clean_dataset(df, missing_strategy='mean', outlier_threshold=3):
    """
    Clean a pandas DataFrame by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'drop')
    outlier_threshold (float): Z-score threshold for outlier detection
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if missing_strategy == 'mean':
        cleaned_df = cleaned_df.fillna(cleaned_df.mean())
    elif missing_strategy == 'median':
        cleaned_df = cleaned_df.fillna(cleaned_df.median())
    elif missing_strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    
    # Remove outliers using Z-score method
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        z_scores = np.abs((cleaned_df[col] - cleaned_df[col].mean()) / cleaned_df[col].std())
        cleaned_df = cleaned_df[z_scores < outlier_threshold]
    
    # Reset index after dropping rows
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def normalize_data(df, method='minmax'):
    """
    Normalize numerical columns in DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    method (str): Normalization method ('minmax' or 'standard')
    
    Returns:
    pd.DataFrame: Normalized DataFrame
    """
    normalized_df = df.copy()
    numeric_cols = normalized_df.select_dtypes(include=[np.number]).columns
    
    if method == 'minmax':
        for col in numeric_cols:
            col_min = normalized_df[col].min()
            col_max = normalized_df[col].max()
            if col_max != col_min:
                normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)
    
    elif method == 'standard':
        for col in numeric_cols:
            col_mean = normalized_df[col].mean()
            col_std = normalized_df[col].std()
            if col_std > 0:
                normalized_df[col] = (normalized_df[col] - col_mean) / col_std
    
    return normalized_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
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
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.any(np.isinf(df[col])):
            return False, f"Column '{col}' contains infinite values"
    
    return True, "Data validation passed"

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'A': [1, 2, np.nan, 4, 100],
        'B': [5, 6, 7, np.nan, 9],
        'C': [10, 11, 12, 13, 14]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    # Clean the data
    cleaned = clean_dataset(df, missing_strategy='mean', outlier_threshold=2)
    print("Cleaned DataFrame:")
    print(cleaned)
    print()
    
    # Normalize the data
    normalized = normalize_data(cleaned, method='minmax')
    print("Normalized DataFrame:")
    print(normalized)
    print()
    
    # Validate the data
    is_valid, message = validate_data(normalized, required_columns=['A', 'B', 'C'])
    print(f"Validation: {is_valid} - {message}")import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
        fill_missing (str): Method to fill missing values. 
                           Options: 'mean', 'median', 'mode', or 'drop'. 
                           Default is 'mean'.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median', 'mode']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            if cleaned_df[col].isnull().any():
                if fill_missing == 'mean':
                    cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
                elif fill_missing == 'median':
                    cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
                elif fill_missing == 'mode':
                    cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)
    
    return cleaned_df

def validate_dataset(df, check_duplicates=True, check_missing=True):
    """
    Validate a DataFrame for common data quality issues.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        check_duplicates (bool): Check for duplicate rows.
        check_missing (bool): Check for missing values.
    
    Returns:
        dict: Dictionary containing validation results.
    """
    validation_results = {}
    
    if check_duplicates:
        duplicate_count = df.duplicated().sum()
        validation_results['duplicate_rows'] = duplicate_count
    
    if check_missing:
        missing_values = df.isnull().sum().sum()
        validation_results['missing_values'] = missing_values
    
    validation_results['total_rows'] = len(df)
    validation_results['total_columns'] = len(df.columns)
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, None, 6],
        'B': [10, 20, 20, None, 50, 60],
        'C': [100, 200, 200, 400, 500, 600]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nValidation Results:")
    print(validate_dataset(df))
    
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    print("\nValidation Results after cleaning:")
    print(validate_dataset(cleaned))
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Args:
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

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a DataFrame column.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to analyze
    
    Returns:
        dict: Dictionary containing summary statistics
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

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of column names to clean. If None, clean all numeric columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for column in columns:
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            
            if removed_count > 0:
                print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'temperature': [22, 23, 24, 25, 26, 100, 27, 28, 29, -10, 30, 31],
        'humidity': [45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 200],
        'pressure': [1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1500]
    }
    
    df = pd.DataFrame(sample_data)
    
    print("Original DataFrame:")
    print(df)
    print("\nSummary statistics for temperature:")
    print(calculate_summary_statistics(df, 'temperature'))
    
    cleaned_df = clean_numeric_data(df)
    
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print("\nSummary statistics after cleaning:")
    for column in ['temperature', 'humidity', 'pressure']:
        stats = calculate_summary_statistics(cleaned_df, column)
        print(f"\n{column}:")
        for key, value in stats.items():
            print(f"  {key}: {value:.2f}")
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(file_path, numeric_columns):
    df = pd.read_csv(file_path)
    
    for col in numeric_columns:
        if col in df.columns:
            df = remove_outliers_iqr(df, col)
            df = normalize_minmax(df, col)
    
    df = df.dropna()
    return df

if __name__ == "__main__":
    cleaned_data = clean_dataset('raw_data.csv', ['age', 'income', 'score'])
    cleaned_data.to_csv('cleaned_data.csv', index=False)
    print(f"Cleaned dataset saved. Original shape: {pd.read_csv('raw_data.csv').shape}, Cleaned shape: {cleaned_data.shape}")