import pandas as pd
import sys

def remove_duplicates(input_file, output_file=None, subset=None, keep='first'):
    """
    Remove duplicate rows from a CSV file.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str, optional): Path to output CSV file. If None, overwrites input file
        subset (list, optional): Columns to consider for identifying duplicates
        keep (str): Which duplicate to keep - 'first', 'last', or False to drop all duplicates
    
    Returns:
        int: Number of duplicates removed
    """
    try:
        df = pd.read_csv(input_file)
        initial_rows = len(df)
        
        df_cleaned = df.drop_duplicates(subset=subset, keep=keep)
        final_rows = len(df_cleaned)
        
        duplicates_removed = initial_rows - final_rows
        
        if output_file is None:
            output_file = input_file
        
        df_cleaned.to_csv(output_file, index=False)
        
        print(f"Removed {duplicates_removed} duplicate rows")
        print(f"Original rows: {initial_rows}, Cleaned rows: {final_rows}")
        print(f"Saved to: {output_file}")
        
        return duplicates_removed
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
        return -1
    except pd.errors.EmptyDataError:
        print(f"Error: File '{input_file}' is empty")
        return -1
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return -1

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_cleaner.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    result = remove_duplicates(input_file, output_file)
    
    if result >= 0:
        sys.exit(0)
    else:
        sys.exit(1)import pandas as pd

def clean_dataset(df):
    """
    Clean a pandas DataFrame by removing duplicate rows and
    filling missing values with appropriate defaults.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()
    
    # Fill missing values
    # For numeric columns, fill with median
    numeric_cols = df_cleaned.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
    
    # For categorical columns, fill with mode
    categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else 'Unknown')
    
    # Reset index after cleaning
    df_cleaned = df_cleaned.reset_index(drop=True)
    
    return df_cleaned

def validate_dataframe(df):
    """
    Validate that the DataFrame meets basic requirements.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 2, 3, 4, None],
        'name': ['Alice', 'Bob', 'Bob', None, 'Eve', 'Frank'],
        'age': [25, 30, 30, 35, None, 40],
        'score': [85.5, 92.0, 92.0, None, 78.5, 88.0]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned_df = clean_dataset(df)
    print(cleaned_df)
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specific column using the Interquartile Range method.
    
    Parameters:
    data (np.ndarray): Input data array
    column (int): Column index to process
    
    Returns:
    np.ndarray: Data with outliers removed
    """
    if data.size == 0:
        return data
    
    col_data = data[:, column]
    q1 = np.percentile(col_data, 25)
    q3 = np.percentile(col_data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    mask = (col_data >= lower_bound) & (col_data <= upper_bound)
    return data[mask]

def calculate_statistics(data, column):
    """
    Calculate basic statistics for a column after outlier removal.
    
    Parameters:
    data (np.ndarray): Input data array
    column (int): Column index to analyze
    
    Returns:
    dict: Dictionary containing statistical measures
    """
    if data.size == 0:
        return {}
    
    col_data = data[:, column]
    stats = {
        'mean': np.mean(col_data),
        'median': np.median(col_data),
        'std': np.std(col_data),
        'min': np.min(col_data),
        'max': np.max(col_data),
        'count': len(col_data)
    }
    return stats

def clean_dataset(data, columns_to_clean=None):
    """
    Clean dataset by removing outliers from specified columns.
    
    Parameters:
    data (np.ndarray): Input data array
    columns_to_clean (list): List of column indices to clean
    
    Returns:
    np.ndarray: Cleaned data array
    """
    if columns_to_clean is None:
        columns_to_clean = list(range(data.shape[1]))
    
    cleaned_data = data.copy()
    for column in columns_to_clean:
        if column < data.shape[1]:
            cleaned_data = remove_outliers_iqr(cleaned_data, column)
    
    return cleaned_data

def validate_data(data):
    """
    Validate data for common issues.
    
    Parameters:
    data (np.ndarray): Input data array
    
    Returns:
    bool: True if data is valid, False otherwise
    """
    if data is None:
        return False
    
    if not isinstance(data, np.ndarray):
        return False
    
    if data.size == 0:
        return False
    
    if np.any(np.isnan(data)):
        return False
    
    return True

def process_data_pipeline(data, columns=None):
    """
    Complete data processing pipeline.
    
    Parameters:
    data (np.ndarray): Input data array
    columns (list): List of column indices to process
    
    Returns:
    tuple: (cleaned_data, statistics_dict)
    """
    if not validate_data(data):
        raise ValueError("Invalid input data")
    
    cleaned_data = clean_dataset(data, columns)
    
    statistics = {}
    if columns is None:
        columns = list(range(data.shape[1]))
    
    for column in columns:
        if column < data.shape[1]:
            stats = calculate_statistics(cleaned_data, column)
            statistics[f'column_{column}'] = stats
    
    return cleaned_data, statistics