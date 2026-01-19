
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    data (list or np.array): The dataset.
    column (int): Index of the column to process.
    
    Returns:
    np.array: Data with outliers removed.
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    col_data = data[:, column].astype(float)
    
    Q1 = np.percentile(col_data, 25)
    Q3 = np.percentile(col_data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    mask = (col_data >= lower_bound) & (col_data <= upper_bound)
    
    return data[mask]

def calculate_statistics(data, column):
    """
    Calculate basic statistics for a column after outlier removal.
    
    Parameters:
    data (list or np.array): The dataset.
    column (int): Index of the column to analyze.
    
    Returns:
    dict: Dictionary containing mean, median, and standard deviation.
    """
    cleaned_data = remove_outliers_iqr(data, column)
    col_data = cleaned_data[:, column].astype(float)
    
    stats = {
        'mean': np.mean(col_data),
        'median': np.median(col_data),
        'std': np.std(col_data),
        'sample_count': len(col_data)
    }
    
    return stats

if __name__ == "__main__":
    sample_data = np.array([
        [1, 150.5],
        [2, 162.3],
        [3, 145.8],
        [4, 210.1],
        [5, 138.9],
        [6, 155.2],
        [7, 300.7],
        [8, 148.6],
        [9, 152.4],
        [10, 165.0]
    ])
    
    cleaned = remove_outliers_iqr(sample_data, 1)
    print("Cleaned data:")
    print(cleaned)
    
    stats = calculate_statistics(sample_data, 1)
    print("\nStatistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")
import pandas as pd
import re

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        column_mapping (dict, optional): Dictionary mapping old column names to new ones
        drop_duplicates (bool): Whether to remove duplicate rows
        normalize_text (bool): Whether to normalize text columns (strip, lower case)
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    
    if normalize_text:
        for col in cleaned_df.select_dtypes(include=['object']).columns:
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip().str.lower()
            cleaned_df[col] = cleaned_df[col].apply(lambda x: re.sub(r'\s+', ' ', x))
    
    return cleaned_df

def validate_email(email_series):
    """
    Validate email addresses in a pandas Series.
    
    Args:
        email_series (pd.Series): Series containing email addresses
    
    Returns:
        pd.Series: Boolean series indicating valid emails
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return email_series.str.match(pattern)

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers from a DataFrame column using IQR method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to check for outliers
        multiplier (float): IQR multiplier for outlier detection
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]import pandas as pd
import numpy as np

def clean_dataset(df, duplicate_threshold=0.8):
    """
    Clean dataset by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    duplicate_threshold (float): Threshold for duplicate detection (0.0 to 1.0)
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    # Create a copy to avoid modifying original
    cleaned_df = df.copy()
    
    # Remove exact duplicates
    initial_rows = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates()
    exact_duplicates = initial_rows - len(cleaned_df)
    
    # Find and remove approximate duplicates based on threshold
    if duplicate_threshold < 1.0:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Calculate similarity matrix for numeric columns
            from sklearn.metrics.pairwise import cosine_similarity
            numeric_data = cleaned_df[numeric_cols].fillna(0)
            similarity_matrix = cosine_similarity(numeric_data)
            
            # Find rows with high similarity
            duplicate_mask = np.zeros(len(cleaned_df), dtype=bool)
            for i in range(len(similarity_matrix)):
                if not duplicate_mask[i]:
                    similar_indices = np.where(similarity_matrix[i] > duplicate_threshold)[0]
                    if len(similar_indices) > 1:
                        # Keep first occurrence, mark others as duplicates
                        duplicate_mask[similar_indices[1:]] = True
            
            approx_duplicates = duplicate_mask.sum()
            cleaned_df = cleaned_df[~duplicate_mask]
        else:
            approx_duplicates = 0
    else:
        approx_duplicates = 0
    
    # Handle missing values
    missing_before = cleaned_df.isnull().sum().sum()
    
    # For numeric columns, fill with median
    for col in cleaned_df.select_dtypes(include=[np.number]).columns:
        if cleaned_df[col].isnull().any():
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    
    # For categorical columns, fill with mode
    for col in cleaned_df.select_dtypes(include=['object']).columns:
        if cleaned_df[col].isnull().any():
            mode_value = cleaned_df[col].mode()
            if not mode_value.empty:
                cleaned_df[col] = cleaned_df[col].fillna(mode_value.iloc[0])
            else:
                cleaned_df[col] = cleaned_df[col].fillna('Unknown')
    
    missing_after = cleaned_df.isnull().sum().sum()
    
    # Log cleaning statistics
    print(f"Data cleaning completed:")
    print(f"  - Exact duplicates removed: {exact_duplicates}")
    print(f"  - Approximate duplicates removed: {approx_duplicates}")
    print(f"  - Missing values handled: {missing_before - missing_after}")
    print(f"  - Final dataset shape: {cleaned_df.shape}")
    
    return cleaned_df

def validate_dataframe(df):
    """
    Validate dataframe structure and content.
    
    Parameters:
    df (pd.DataFrame): Dataframe to validate
    
    Returns:
    dict: Validation results
    """
    validation_results = {
        'has_data': len(df) > 0,
        'columns_count': len(df.columns),
        'rows_count': len(df),
        'missing_values': df.isnull().sum().sum(),
        'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': len(df.select_dtypes(include=['object']).columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
    }
    
    return validation_results

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'id': [1, 2, 3, 4, 5, 1, 2, 7, 8, 9],
        'value': [10.5, 20.3, 15.7, 10.5, 25.1, 10.5, 20.3, 18.9, None, 22.4],
        'category': ['A', 'B', 'A', 'C', 'B', 'A', 'B', None, 'D', 'E'],
        'score': [85, 92, 78, 85, 95, 85, 92, 88, 76, 90]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Validate original data
    validation = validate_dataframe(df)
    print("Validation results:")
    for key, value in validation.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned_df = clean_dataset(df, duplicate_threshold=0.95)
    
    print("\n" + "="*50 + "\n")
    print("Cleaned dataset:")
    print(cleaned_df)import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=False, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    drop_duplicates (bool): Whether to remove duplicate rows
    fill_missing (bool): Whether to fill missing values
    fill_value: Value to use for filling missing data
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing:
        missing_before = cleaned_df.isnull().sum().sum()
        cleaned_df = cleaned_df.fillna(fill_value)
        missing_after = cleaned_df.isnull().sum().sum()
        print(f"Filled {missing_before - missing_after} missing values")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate that a DataFrame meets basic requirements.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of column names that must be present
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
    
    return True

def get_data_summary(df):
    """
    Generate a summary of the DataFrame including shape, dtypes, and missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum()
    }
    return summary

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, 2, 3, None],
        'B': [4, 5, 5, None, 7],
        'C': ['x', 'y', 'y', 'z', 'z']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_value=0)
    print("Cleaned DataFrame:")
    print(cleaned)
    
    is_valid = validate_dataframe(cleaned, required_columns=['A', 'B', 'C'])
    print(f"\nDataFrame validation: {is_valid}")
    
    summary = get_data_summary(cleaned)
    print(f"\nDataFrame shape: {summary['shape']}")