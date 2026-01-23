
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Args:
        data: List or numpy array of numerical values
        column: Column index or name (if using pandas DataFrame)
    
    Returns:
        Cleaned data with outliers removed
    """
    if isinstance(data, list):
        data_array = np.array(data)
    else:
        data_array = data
    
    q1 = np.percentile(data_array, 25)
    q3 = np.percentile(data_array, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    filtered_data = data_array[(data_array >= lower_bound) & (data_array <= upper_bound)]
    
    return filtered_data

def calculate_basic_stats(data):
    """
    Calculate basic statistics for the data.
    
    Args:
        data: List or numpy array of numerical values
    
    Returns:
        Dictionary containing mean, median, and standard deviation
    """
    if isinstance(data, list):
        data_array = np.array(data)
    else:
        data_array = data
    
    stats = {
        'mean': np.mean(data_array),
        'median': np.median(data_array),
        'std': np.std(data_array),
        'min': np.min(data_array),
        'max': np.max(data_array)
    }
    
    return stats

def clean_dataset(data, columns_to_clean=None):
    """
    Clean entire dataset by removing outliers from specified columns.
    
    Args:
        data: Dictionary or pandas-like structure with column data
        columns_to_clean: List of column names/indices to clean
    
    Returns:
        Cleaned dataset
    """
    if columns_to_clean is None:
        if isinstance(data, dict):
            columns_to_clean = list(data.keys())
        else:
            return data
    
    cleaned_data = {}
    
    for column in columns_to_clean:
        if column in data:
            original_data = data[column]
            cleaned = remove_outliers_iqr(original_data, column)
            cleaned_data[column] = cleaned
            
            print(f"Column '{column}':")
            print(f"  Original count: {len(original_data)}")
            print(f"  Cleaned count: {len(cleaned)}")
            print(f"  Removed outliers: {len(original_data) - len(cleaned)}")
            print(f"  Basic stats: {calculate_basic_stats(cleaned)}")
            print()
    
    return cleaned_data

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'temperature': [22, 23, 24, 25, 26, 100, 27, 28, 29, -10],
        'humidity': [45, 46, 47, 48, 49, 200, 50, 51, 52, -5],
        'pressure': [1013, 1014, 1015, 1016, 1017, 2000, 1018, 1019, 1020, 500]
    }
    
    print("Original dataset:")
    for col, values in sample_data.items():
        print(f"{col}: {values}")
    
    print("\nCleaning dataset...")
    cleaned = clean_dataset(sample_data)
    
    print("\nCleaned dataset:")
    for col, values in cleaned.items():
        print(f"{col}: {values}")
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
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
        elif fill_missing == 'mean':
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].mean())
            print("Filled missing numeric values with column means")
        elif fill_missing == 'median':
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].median())
            print("Filled missing numeric values with column medians")
        elif fill_missing == 'mode':
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype == 'object':
                    mode_val = cleaned_df[col].mode()
                    if not mode_val.empty:
                        cleaned_df[col] = cleaned_df[col].fillna(mode_val[0])
            print("Filled missing categorical values with column modes")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of column names that must be present
    min_rows (int): Minimum number of rows required
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if len(df) < min_rows:
        print(f"Error: DataFrame has fewer than {min_rows} rows")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    return True

def get_data_summary(df):
    """
    Generate a summary of the DataFrame.
    
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
        'numeric_stats': {}
    }
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        summary['numeric_stats'][col] = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max()
        }
    
    return summary