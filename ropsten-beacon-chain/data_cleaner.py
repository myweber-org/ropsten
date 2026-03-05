
import pandas as pd

def remove_duplicates(dataframe, subset=None, keep='first'):
    """
    Remove duplicate rows from a pandas DataFrame.
    
    Args:
        dataframe: Input DataFrame
        subset: Column label or sequence of labels to consider for identifying duplicates
        keep: Determines which duplicates to keep ('first', 'last', False)
    
    Returns:
        DataFrame with duplicates removed
    """
    if subset is None:
        subset = dataframe.columns.tolist()
    
    cleaned_df = dataframe.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(dataframe) - len(cleaned_df)
    print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df

def validate_dataframe(dataframe, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        dataframe: DataFrame to validate
        required_columns: List of columns that must be present
    
    Returns:
        Boolean indicating if validation passed
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in dataframe.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    if dataframe.empty:
        print("Warning: DataFrame is empty")
        return False
    
    null_counts = dataframe.isnull().sum()
    if null_counts.any():
        print("Null values detected:")
        for col, count in null_counts[null_counts > 0].items():
            print(f"  {col}: {count} nulls")
    
    return True

def clean_numeric_columns(dataframe, columns=None):
    """
    Clean numeric columns by converting to appropriate types and handling errors.
    
    Args:
        dataframe: Input DataFrame
        columns: Specific columns to clean (defaults to all numeric columns)
    
    Returns:
        DataFrame with cleaned numeric columns
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=['number']).columns
    
    cleaned_df = dataframe.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            try:
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
            except Exception as e:
                print(f"Error cleaning column {col}: {e}")
    
    return cleaned_df

def get_data_summary(dataframe):
    """
    Generate summary statistics for a DataFrame.
    
    Args:
        dataframe: Input DataFrame
    
    Returns:
        Dictionary containing summary statistics
    """
    summary = {
        'total_rows': len(dataframe),
        'total_columns': len(dataframe.columns),
        'column_names': dataframe.columns.tolist(),
        'data_types': dataframe.dtypes.to_dict(),
        'memory_usage': dataframe.memory_usage(deep=True).sum(),
        'null_values': dataframe.isnull().sum().to_dict()
    }
    
    numeric_cols = dataframe.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        summary['numeric_summary'] = dataframe[numeric_cols].describe().to_dict()
    
    return summarydef remove_duplicates(input_list):
    """
    Removes duplicate items from a list while preserving the original order.
    Uses a dictionary to track seen items for O(n) time complexity.
    """
    seen = {}
    result = []
    for item in input_list:
        if item not in seen:
            seen[item] = True
            result.append(item)
    return result
def remove_duplicates_preserve_order(iterable):
    seen = set()
    result = []
    for item in iterable:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return resultimport pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to remove duplicate rows
        fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if cleaned_df.isnull().sum().sum() > 0:
        print(f"Found {cleaned_df.isnull().sum().sum()} missing values")
        
        if fill_missing == 'drop':
            cleaned_df = cleaned_df.dropna()
            print("Dropped rows with missing values")
        elif fill_missing in ['mean', 'median']:
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
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
    
    print(f"Cleaning complete. Final shape: {cleaned_df.shape}")
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    if not isinstance(df, pd.DataFrame):
        validation_results['is_valid'] = False
        validation_results['errors'].append("Input is not a pandas DataFrame")
        return validation_results
    
    if df.empty:
        validation_results['warnings'].append("DataFrame is empty")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing required columns: {missing_cols}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        validation_results['warnings'].append("No numeric columns found")
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, 5, None],
        'B': [10, 20, 20, None, 50, 60],
        'C': ['x', 'y', 'y', 'z', None, 'x']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaning with mean imputation...")
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    validation = validate_dataframe(cleaned, required_columns=['A', 'B'])
    print("\nValidation results:", validation)
import pandas as pd
import numpy as np

def clean_dataframe(df, missing_strategy='mean', outlier_method='iqr'):
    """
    Clean a pandas DataFrame by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'drop')
    outlier_method (str): Method for detecting outliers ('iqr', 'zscore')
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    
    cleaned_df = df.copy()
    
    # Handle missing values
    if missing_strategy == 'mean':
        cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
    elif missing_strategy == 'median':
        cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
    elif missing_strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    
    # Handle outliers for numeric columns
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    
    if outlier_method == 'iqr':
        for col in numeric_cols:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers
            cleaned_df[col] = cleaned_df[col].clip(lower=lower_bound, upper=upper_bound)
    
    elif outlier_method == 'zscore':
        for col in numeric_cols:
            z_scores = np.abs((cleaned_df[col] - cleaned_df[col].mean()) / cleaned_df[col].std())
            mask = z_scores > 3
            cleaned_df.loc[mask, col] = cleaned_df[col].median()
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
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
    
    return True, "DataFrame is valid"

# Example usage function
def process_example_data():
    """
    Example function demonstrating data cleaning workflow.
    """
    # Create sample data with missing values and outliers
    np.random.seed(42)
    data = {
        'A': np.random.randn(100),
        'B': np.random.randn(100),
        'C': np.random.randn(100)
    }
    
    # Introduce missing values
    for col in data:
        mask = np.random.random(100) < 0.1
        data[col][mask] = np.nan
    
    # Introduce outliers
    data['A'][0] = 100
    data['B'][1] = -50
    
    df = pd.DataFrame(data)
    
    print("Original DataFrame shape:", df.shape)
    print("Missing values per column:")
    print(df.isnull().sum())
    
    # Clean the data
    cleaned_df = clean_dataframe(df, missing_strategy='mean', outlier_method='iqr')
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Missing values after cleaning:")
    print(cleaned_df.isnull().sum())
    
    # Validate the cleaned data
    is_valid, message = validate_dataframe(cleaned_df, min_rows=50)
    print(f"\nValidation result: {is_valid}")
    print(f"Validation message: {message}")
    
    return cleaned_df

if __name__ == "__main__":
    result = process_example_data()
    print("\nData cleaning completed successfully.")