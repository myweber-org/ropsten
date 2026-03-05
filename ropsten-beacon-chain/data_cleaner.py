
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