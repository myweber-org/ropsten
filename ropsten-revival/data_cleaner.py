
import pandas as pd
import numpy as np

def clean_dataset(df, duplicate_threshold=0.8, missing_strategy='median'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df: Input DataFrame
        duplicate_threshold: Similarity threshold for duplicate detection (0-1)
        missing_strategy: Strategy for handling missing values ('median', 'mean', 'mode', 'drop')
    
    Returns:
        Cleaned DataFrame
    """
    original_shape = df.shape
    
    # Remove exact duplicates
    df_cleaned = df.drop_duplicates().reset_index(drop=True)
    
    # Remove approximate duplicates based on similarity threshold
    if duplicate_threshold < 1.0:
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Calculate similarity matrix for numeric columns
            numeric_data = df_cleaned[numeric_cols].fillna(0)
            similarity_matrix = cosine_similarity(numeric_data)
            
            # Find duplicates above threshold
            duplicates = set()
            for i in range(len(similarity_matrix)):
                for j in range(i + 1, len(similarity_matrix)):
                    if similarity_matrix[i, j] > duplicate_threshold:
                        duplicates.add(j)
            
            # Remove duplicate indices
            df_cleaned = df_cleaned.drop(index=list(duplicates)).reset_index(drop=True)
    
    # Handle missing values
    if missing_strategy == 'drop':
        df_cleaned = df_cleaned.dropna()
    elif missing_strategy in ['median', 'mean']:
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if missing_strategy == 'median':
                fill_value = df_cleaned[col].median()
            else:  # mean
                fill_value = df_cleaned[col].mean()
            df_cleaned[col] = df_cleaned[col].fillna(fill_value)
    elif missing_strategy == 'mode':
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype == 'object':
                fill_value = df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else ''
                df_cleaned[col] = df_cleaned[col].fillna(fill_value)
    
    # Log cleaning statistics
    print(f"Original shape: {original_shape}")
    print(f"Cleaned shape: {df_cleaned.shape}")
    print(f"Rows removed: {original_shape[0] - df_cleaned.shape[0]}")
    print(f"Columns: {df_cleaned.shape[1]}")
    
    return df_cleaned

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
    
    Returns:
        Boolean indicating if validation passed
    """
    if df is None or df.empty:
        print("Error: DataFrame is empty or None")
        return False
    
    if len(df) < min_rows:
        print(f"Error: DataFrame has fewer than {min_rows} rows")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    # Check for infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.any(np.isinf(df[col])):
            print(f"Warning: Column '{col}' contains infinite values")
    
    return True

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'id': [1, 2, 3, 4, 5, 6, 7, 8],
        'value_a': [10.5, 20.3, 10.5, 15.7, np.nan, 20.3, 25.1, 30.0],
        'value_b': [100, 200, 100, 150, 250, 200, 300, 350],
        'category': ['A', 'B', 'A', 'C', 'B', 'B', 'A', 'C']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned_df = clean_dataset(df, duplicate_threshold=0.9, missing_strategy='median')
    
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    # Validate the cleaned data
    is_valid = validate_dataframe(cleaned_df, required_columns=['id', 'value_a', 'value_b'], min_rows=1)
    print(f"\nData validation passed: {is_valid}")
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
    
    return filtered_df

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
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
        'count': df[column].count()
    }
    
    return stats

def clean_dataset(df, columns_to_clean=None):
    """
    Clean multiple columns in a DataFrame by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns_to_clean (list): List of column names to clean. If None, clean all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns_to_clean is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns_to_clean = numeric_cols
    
    cleaned_df = df.copy()
    
    for column in columns_to_clean:
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_df
import pandas as pd
import numpy as np
from scipy import stats

def normalize_data(df, columns, method='zscore'):
    """
    Normalize specified columns in DataFrame.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to normalize
        method: 'zscore' or 'minmax'
    
    Returns:
        DataFrame with normalized columns
    """
    df_copy = df.copy()
    
    for col in columns:
        if col not in df_copy.columns:
            continue
            
        if method == 'zscore':
            df_copy[f'{col}_normalized'] = stats.zscore(df_copy[col])
        elif method == 'minmax':
            col_min = df_copy[col].min()
            col_max = df_copy[col].max()
            df_copy[f'{col}_normalized'] = (df_copy[col] - col_min) / (col_max - col_min)
    
    return df_copy

def remove_outliers(df, columns, method='iqr', threshold=1.5):
    """
    Remove outliers from specified columns.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to process
        method: 'iqr' or 'zscore'
        threshold: threshold value for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    df_copy = df.copy()
    
    for col in columns:
        if col not in df_copy.columns:
            continue
            
        if method == 'iqr':
            Q1 = df_copy[col].quantile(0.25)
            Q3 = df_copy[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            mask = (df_copy[col] >= lower_bound) & (df_copy[col] <= upper_bound)
            df_copy = df_copy[mask]
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(df_copy[col]))
            df_copy = df_copy[z_scores < threshold]
    
    return df_copy.reset_index(drop=True)

def clean_missing_values(df, columns, strategy='mean'):
    """
    Handle missing values in specified columns.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to process
        strategy: 'mean', 'median', 'mode', or 'drop'
    
    Returns:
        DataFrame with handled missing values
    """
    df_copy = df.copy()
    
    for col in columns:
        if col not in df_copy.columns:
            continue
            
        if strategy == 'mean':
            df_copy[col].fillna(df_copy[col].mean(), inplace=True)
        elif strategy == 'median':
            df_copy[col].fillna(df_copy[col].median(), inplace=True)
        elif strategy == 'mode':
            df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
        elif strategy == 'drop':
            df_copy = df_copy.dropna(subset=[col])
    
    return df_copy

def validate_data(df, column_types):
    """
    Validate DataFrame columns against expected data types.
    
    Args:
        df: pandas DataFrame
        column_types: dict of column_name: expected_type
    
    Returns:
        dict with validation results
    """
    results = {}
    
    for col, expected_type in column_types.items():
        if col not in df.columns:
            results[col] = {'status': 'missing', 'message': f'Column {col} not found'}
            continue
            
        actual_type = str(df[col].dtype)
        
        if expected_type == 'numeric':
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            results[col] = {
                'status': 'valid' if is_numeric else 'invalid',
                'expected': expected_type,
                'actual': actual_type
            }
        elif expected_type == 'datetime':
            try:
                pd.to_datetime(df[col])
                results[col] = {
                    'status': 'valid',
                    'expected': expected_type,
                    'actual': actual_type
                }
            except:
                results[col] = {
                    'status': 'invalid',
                    'expected': expected_type,
                    'actual': actual_type
                }
        else:
            results[col] = {
                'status': 'valid' if actual_type == expected_type else 'invalid',
                'expected': expected_type,
                'actual': actual_type
            }
    
    return results
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
        fill_missing (str or dict): Method to fill missing values. 
            Options: 'mean', 'median', 'mode', or a dictionary of column:value pairs.
            If None, missing values are not filled.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
        print(f"Removed {len(df) - len(cleaned_df)} duplicate rows.")
    
    if fill_missing is not None:
        if fill_missing == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_missing == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif fill_missing == 'mode':
            cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
        elif isinstance(fill_missing, dict):
            cleaned_df = cleaned_df.fillna(fill_missing)
        else:
            raise ValueError("Invalid fill_missing option. Use 'mean', 'median', 'mode', or a dictionary.")
        
        print(f"Filled missing values using method: {fill_missing}")
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
        min_rows (int): Minimum number of rows required.
    
    Returns:
        bool: True if validation passes, False otherwise.
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return False
    
    if len(df) < min_rows:
        print(f"DataFrame has only {len(df)} rows, minimum required is {min_rows}")
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
    
    is_valid = validate_data(cleaned, required_columns=['A', 'B'], min_rows=3)
    print(f"\nData validation passed: {is_valid}")