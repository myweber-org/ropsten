import pandas as pd

def clean_dataset(input_file, output_file):
    """
    Load a CSV file, remove rows with null values,
    drop duplicate rows, and save the cleaned data.
    """
    try:
        df = pd.read_csv(input_file)
        initial_rows = df.shape[0]
        
        df_cleaned = df.dropna()
        df_cleaned = df_cleaned.drop_duplicates()
        final_rows = df_cleaned.shape[0]
        
        df_cleaned.to_csv(output_file, index=False)
        
        print(f"Data cleaning completed.")
        print(f"Initial rows: {initial_rows}")
        print(f"Rows after cleaning: {final_rows}")
        print(f"Removed rows: {initial_rows - final_rows}")
        print(f"Cleaned data saved to: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except pd.errors.EmptyDataError:
        print("Error: The input file is empty.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    clean_dataset(input_csv, output_csv)
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        factor: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        threshold: Z-score threshold (default 3)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    mask = z_scores < threshold
    return data[mask]

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
    Args:
        data: pandas DataFrame or Series
        column: column name to normalize
    
    Returns:
        Normalized data
    """
    if isinstance(data, pd.DataFrame):
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        col_data = data[column]
    else:
        col_data = data
    
    min_val = col_data.min()
    max_val = col_data.max()
    
    if max_val == min_val:
        return col_data
    
    return (col_data - min_val) / (max_val - min_val)

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization.
    
    Args:
        data: pandas DataFrame or Series
        column: column name to normalize
    
    Returns:
        Standardized data
    """
    if isinstance(data, pd.DataFrame):
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        col_data = data[column]
    else:
        col_data = data
    
    mean_val = col_data.mean()
    std_val = col_data.std()
    
    if std_val == 0:
        return col_data
    
    return (col_data - mean_val) / std_val

def clean_dataset(data, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric columns to process (default: all numeric)
        outlier_method: 'iqr', 'zscore', or None
        normalize_method: 'minmax', 'zscore', or None
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    # Remove outliers
    if outlier_method:
        for col in numeric_columns:
            if col in cleaned_data.columns:
                if outlier_method == 'iqr':
                    cleaned_data = remove_outliers_iqr(cleaned_data, col)
                elif outlier_method == 'zscore':
                    cleaned_data = remove_outliers_zscore(cleaned_data, col)
    
    # Normalize data
    if normalize_method:
        for col in numeric_columns:
            if col in cleaned_data.columns:
                if normalize_method == 'minmax':
                    cleaned_data[col] = normalize_minmax(cleaned_data, col)
                elif normalize_method == 'zscore':
                    cleaned_data[col] = normalize_zscore(cleaned_data, col)
    
    return cleaned_data

def validate_data(data, check_missing=True, check_infinite=True):
    """
    Validate data quality.
    
    Args:
        data: pandas DataFrame
        check_missing: check for missing values
        check_infinite: check for infinite values
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {}
    
    if check_missing:
        missing_counts = data.isnull().sum()
        validation_results['missing_values'] = missing_counts[missing_counts > 0].to_dict()
    
    if check_infinite:
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        infinite_counts = {}
        for col in numeric_cols:
            infinite_mask = np.isinf(data[col])
            if infinite_mask.any():
                infinite_counts[col] = infinite_mask.sum()
        validation_results['infinite_values'] = infinite_counts
    
    validation_results['shape'] = data.shape
    validation_results['dtypes'] = data.dtypes.to_dict()
    
    return validation_results
import pandas as pd
import re

def clean_dataframe(df, columns_to_clean=None):
    """
    Clean a pandas DataFrame by removing duplicate rows and normalizing string columns.
    """
    df_clean = df.copy()
    
    # Remove duplicate rows
    initial_rows = df_clean.shape[0]
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    removed_duplicates = initial_rows - df_clean.shape[0]
    
    # If specific columns are provided, clean only those; otherwise clean all object columns
    if columns_to_clean is None:
        columns_to_clean = df_clean.select_dtypes(include=['object']).columns.tolist()
    
    # Normalize strings: strip whitespace, convert to lowercase, remove extra spaces
    for col in columns_to_clean:
        if col in df_clean.columns and df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].astype(str).apply(lambda x: re.sub(r'\s+', ' ', x.strip().lower()))
    
    return df_clean, removed_duplicates

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column and return a boolean mask.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return df[email_column].astype(str).str.match(email_pattern)

# Example usage (commented out)
# if __name__ == "__main__":
#     data = {'name': [' Alice ', 'bob', 'Alice', '  CAROL  '],
#             'email': ['alice@example.com', 'invalid', 'alice@example.com', 'carol@test.org']}
#     df = pd.DataFrame(data)
#     cleaned_df, duplicates_removed = clean_dataframe(df)
#     print(f"Removed {duplicates_removed} duplicate rows")
#     print(cleaned_df)
#     valid_emails = validate_email_column(cleaned_df, 'email')
#     print("Valid emails:", valid_emails)