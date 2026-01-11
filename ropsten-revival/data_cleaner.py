import pandas as pd
import numpy as np
from scipy import stats

def clean_dataset(df, numeric_columns=None, z_threshold=3, fill_method='median'):
    """
    Clean dataset by handling missing values and removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    numeric_columns (list): List of numeric column names to process
    z_threshold (float): Z-score threshold for outlier detection
    fill_method (str): Method for filling missing values ('median' or 'mean')
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_clean = df.copy()
    
    for col in numeric_columns:
        if col not in df_clean.columns:
            continue
            
        col_data = df_clean[col]
        
        if fill_method == 'median':
            fill_value = col_data.median()
        else:
            fill_value = col_data.mean()
        
        df_clean[col] = col_data.fillna(fill_value)
        
        z_scores = np.abs(stats.zscore(df_clean[col]))
        outlier_mask = z_scores < z_threshold
        
        df_clean = df_clean[outlier_mask]
    
    return df_clean.reset_index(drop=True)

def normalize_data(df, numeric_columns=None, method='minmax'):
    """
    Normalize numeric columns in dataframe.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    numeric_columns (list): List of numeric column names to normalize
    method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
    pd.DataFrame: Normalized dataframe
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_normalized = df.copy()
    
    for col in numeric_columns:
        if col not in df_normalized.columns:
            continue
            
        if method == 'minmax':
            col_min = df_normalized[col].min()
            col_max = df_normalized[col].max()
            if col_max != col_min:
                df_normalized[col] = (df_normalized[col] - col_min) / (col_max - col_min)
        else:
            col_mean = df_normalized[col].mean()
            col_std = df_normalized[col].std()
            if col_std > 0:
                df_normalized[col] = (df_normalized[col] - col_mean) / col_std
    
    return df_normalized

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from dataframe.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    subset (list): Columns to consider for duplicate detection
    keep (str): Which duplicates to keep ('first', 'last', or False)
    
    Returns:
    pd.DataFrame: Dataframe with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate dataframe structure and content.
    
    Parameters:
    df (pd.DataFrame): Input dataframe to validate
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
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if not numeric_cols.empty:
        inf_values = df[numeric_cols].applymap(lambda x: np.isinf(x)).any().any()
        if inf_values:
            return False, "DataFrame contains infinite values"
    
    return True, "Data validation passed"
import pandas as pd
import sys

def remove_duplicates(input_file, output_file=None, subset=None, keep='first'):
    """
    Remove duplicate rows from a CSV file.
    
    Parameters:
    input_file (str): Path to the input CSV file.
    output_file (str, optional): Path to the output CSV file. If None, overwrites input file.
    subset (list, optional): Columns to consider for identifying duplicates.
    keep (str): Which duplicate to keep. Options: 'first', 'last', False.
    
    Returns:
    int: Number of duplicates removed.
    """
    try:
        df = pd.read_csv(input_file)
        initial_count = len(df)
        
        df_cleaned = df.drop_duplicates(subset=subset, keep=keep)
        final_count = len(df_cleaned)
        
        duplicates_removed = initial_count - final_count
        
        if output_file is None:
            output_file = input_file
            
        df_cleaned.to_csv(output_file, index=False)
        
        print(f"Processed: {input_file}")
        print(f"Initial rows: {initial_count}")
        print(f"Final rows: {final_count}")
        print(f"Duplicates removed: {duplicates_removed}")
        
        return duplicates_removed
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return -1
    except Exception as e:
        print(f"Error processing file: {e}")
        return -1

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_cleaner.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    result = remove_duplicates(input_file, output_file)
    
    if result >= 0:
        print("Data cleaning completed successfully.")
    else:
        print("Data cleaning failed.")