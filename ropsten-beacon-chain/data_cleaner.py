
import pandas as pd
import numpy as np

def clean_csv_data(file_path, fill_method='mean', output_path=None):
    """
    Load a CSV file, clean missing values, and optionally save cleaned data.
    
    Args:
        file_path (str): Path to input CSV file
        fill_method (str): Method for filling missing values ('mean', 'median', 'mode', 'zero')
        output_path (str, optional): Path to save cleaned CSV. If None, returns DataFrame
    
    Returns:
        pd.DataFrame or None: Cleaned DataFrame if output_path is None, else None
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded data with shape: {df.shape}")
        
        missing_counts = df.isnull().sum()
        if missing_counts.any():
            print("Missing values per column:")
            for col, count in missing_counts[missing_counts > 0].items():
                print(f"  {col}: {count}")
            
            for col in df.columns:
                if df[col].isnull().any():
                    if fill_method == 'mean' and pd.api.types.is_numeric_dtype(df[col]):
                        fill_value = df[col].mean()
                    elif fill_method == 'median' and pd.api.types.is_numeric_dtype(df[col]):
                        fill_value = df[col].median()
                    elif fill_method == 'mode':
                        fill_value = df[col].mode()[0] if not df[col].mode().empty else np.nan
                    elif fill_method == 'zero':
                        fill_value = 0
                    else:
                        fill_value = df[col].ffill().bfill().iloc[0] if not df[col].dropna().empty else np.nan
                    
                    df[col].fillna(fill_value, inplace=True)
        
        df = df.drop_duplicates()
        print(f"Cleaned data shape: {df.shape}")
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
            return None
        else:
            return df
            
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list, optional): List of required column names
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if df is None or df.empty:
        print("Validation failed: DataFrame is empty or None")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing required columns: {missing_cols}")
            return False
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        for col in numeric_cols:
            if df[col].isnull().any():
                print(f"Validation warning: Column '{col}' still contains NaN values")
    
    return True

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, np.nan, 5],
        'C': ['x', 'y', 'z', 'x', np.nan]
    })
    
    sample_data.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data('sample_data.csv', fill_method='mean')
    
    if cleaned_df is not None:
        is_valid = validate_dataframe(cleaned_df, required_columns=['A', 'B', 'C'])
        print(f"Data validation result: {is_valid}")
        
        import os
        os.remove('sample_data.csv')import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
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
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to analyze.
    
    Returns:
    dict: Dictionary containing summary statistics.
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

def process_dataframe(df, column):
    """
    Main function to process DataFrame by removing outliers and calculating statistics.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    tuple: (cleaned_df, original_stats, cleaned_stats)
    """
    original_stats = calculate_summary_statistics(df, column)
    cleaned_df = remove_outliers_iqr(df, column)
    cleaned_stats = calculate_summary_statistics(cleaned_df, column)
    
    return cleaned_df, original_stats, cleaned_stats

if __name__ == "__main__":
    sample_data = {
        'values': [10, 12, 12, 13, 12, 11, 14, 13, 15, 100, 12, 14, 13, 12, 11, 10, 9, 8, 12, 13]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df, original_stats, cleaned_stats = process_dataframe(df, 'values')
    
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    print("\nOriginal Statistics:")
    for key, value in original_stats.items():
        print(f"{key}: {value}")
    
    print("\nCleaned Statistics:")
    for key, value in cleaned_stats.items():
        print(f"{key}: {value}")
import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        subset (list, optional): Column labels to consider for duplicates
        keep (str, optional): Which duplicates to keep ('first', 'last', False)
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(df) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate row(s)")
    
    return cleaned_df

def clean_numeric_columns(df, columns):
    """
    Clean numeric columns by converting to appropriate types and handling errors.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of column names to clean
    
    Returns:
        pd.DataFrame: DataFrame with cleaned numeric columns
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list, optional): List of required column names
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return True
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
    
    return True