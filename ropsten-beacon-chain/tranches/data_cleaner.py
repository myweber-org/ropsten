import pandas as pd

def clean_dataset(df, remove_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    remove_duplicates (bool): Whether to remove duplicate rows
    fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if remove_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
        print("Removed rows with missing values")
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        print(f"Filled missing numeric values with {fill_missing}")
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
        print("Filled missing values with mode")
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate a DataFrame for basic quality checks.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    dict: Dictionary with validation results
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        validation_results['missing_required_columns'] = missing_columns
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, None],
        'B': [5, None, 7, 8, 9],
        'C': ['x', 'y', 'y', 'z', 'x']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nValidation results:")
    print(validate_dataset(df))
    
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Parameters:
    data (list or np.array): Input data
    column (int): Column index for 2D data, or None for 1D data
    
    Returns:
    np.array: Data with outliers removed
    """
    if column is not None:
        col_data = data[:, column]
    else:
        col_data = data
    
    q1 = np.percentile(col_data, 25)
    q3 = np.percentile(col_data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    if column is not None:
        mask = (data[:, column] >= lower_bound) & (data[:, column] <= upper_bound)
        return data[mask]
    else:
        mask = (data >= lower_bound) & (data <= upper_bound)
        return data[mask]

def calculate_statistics(data):
    """
    Calculate basic statistics for the data.
    
    Parameters:
    data (np.array): Input data
    
    Returns:
    dict: Dictionary containing mean, median, std, min, max
    """
    stats = {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data)
    }
    return stats

def clean_dataset(data, columns_to_clean=None):
    """
    Clean dataset by removing outliers from specified columns.
    
    Parameters:
    data (np.array): Input dataset
    columns_to_clean (list): List of column indices to clean
    
    Returns:
    np.array: Cleaned dataset
    """
    if columns_to_clean is None:
        columns_to_clean = range(data.shape[1])
    
    cleaned_data = data.copy()
    
    for col in columns_to_clean:
        if col < data.shape[1]:
            cleaned_data = remove_outliers_iqr(cleaned_data, col)
    
    return cleaned_data

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    sample_data = np.random.randn(100, 3) * 10 + 50
    
    print("Original data shape:", sample_data.shape)
    print("Original statistics:", calculate_statistics(sample_data[:, 0]))
    
    cleaned = clean_dataset(sample_data, columns_to_clean=[0, 1, 2])
    
    print("\nCleaned data shape:", cleaned.shape)
    print("Cleaned statistics:", calculate_statistics(cleaned[:, 0]))