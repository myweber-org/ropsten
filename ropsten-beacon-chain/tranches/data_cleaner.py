import pandas as pd
import numpy as np

def clean_csv_data(file_path, output_path=None, missing_strategy='mean'):
    """
    Clean CSV data by handling missing values and removing duplicates.
    
    Args:
        file_path (str): Path to input CSV file
        output_path (str, optional): Path for cleaned output CSV
        missing_strategy (str): Strategy for handling missing values
                               ('mean', 'median', 'drop', 'zero')
    
    Returns:
        pandas.DataFrame: Cleaned dataframe
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Original data shape: {df.shape}")
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        print(f"After removing duplicates: {df.shape}")
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if missing_strategy == 'mean':
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].mean())
        elif missing_strategy == 'median':
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].median())
        elif missing_strategy == 'zero':
            df = df.fillna(0)
        elif missing_strategy == 'drop':
            df = df.dropna()
        
        print(f"After handling missing values: {df.shape}")
        
        # Save cleaned data if output path provided
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate dataframe structure and content.
    
    Args:
        df (pandas.DataFrame): Dataframe to validate
        required_columns (list): List of required column names
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if df is None or df.empty:
        print("Validation failed: Dataframe is empty or None")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing columns {missing_cols}")
            return False
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.any(np.isinf(df[col])):
            print(f"Validation failed: Infinite values found in column {col}")
            return False
    
    print("Data validation passed")
    return True

if __name__ == "__main__":
    # Example usage
    cleaned_data = clean_csv_data(
        file_path='raw_data.csv',
        output_path='cleaned_data.csv',
        missing_strategy='mean'
    )
    
    if cleaned_data is not None:
        validation_result = validate_dataframe(
            cleaned_data,
            required_columns=['id', 'value', 'timestamp']
        )import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing=False, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows.
        fill_missing (bool): Whether to fill missing values.
        fill_value: Value to use for filling missing data.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    return True, "DataFrame is valid"

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers from a DataFrame column using IQR method.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to process.
        multiplier (float): IQR multiplier for outlier detection.
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
import pandas as pd
import re

def clean_dataframe(df, columns_to_clean=None, remove_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    """
    df_clean = df.copy()
    
    if remove_duplicates:
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed = initial_rows - len(df_clean)
        print(f"Removed {removed} duplicate rows.")
    
    if normalize_text and columns_to_clean:
        for col in columns_to_clean:
            if col in df_clean.columns and df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].apply(_normalize_string)
                print(f"Normalized text in column: {col}")
    
    return df_clean

def _normalize_string(text):
    """
    Normalize a string: lowercase, remove extra whitespace, and special characters.
    """
    if pd.isna(text):
        return text
    
    text = str(text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = re.sub(r'[^\w\s]', '', text)
    
    return text

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame.")
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    df['valid_email'] = df[email_column].apply(lambda x: bool(re.match(pattern, str(x))) if pd.notna(x) else False)
    
    valid_count = df['valid_email'].sum()
    total_count = len(df)
    
    print(f"Valid emails: {valid_count}/{total_count} ({valid_count/total_count*100:.2f}%)")
    
    return df
import csv
import hashlib
from collections import defaultdict

def generate_row_hash(row):
    """Generate a hash for a CSV row to identify duplicates."""
    row_string = ','.join(str(value) for value in row)
    return hashlib.md5(row_string.encode()).hexdigest()

def remove_duplicates(input_file, output_file, key_columns=None):
    """
    Remove duplicate rows from a CSV file.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        key_columns: List of column indices to consider for duplicate detection.
                     If None, entire row is considered.
    """
    seen_hashes = set()
    unique_rows = []
    
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        headers = next(reader)
        
        for row in reader:
            if key_columns:
                # Only consider specified columns for duplicate detection
                key_values = [row[i] for i in key_columns if i < len(row)]
                row_hash = generate_row_hash(key_values)
            else:
                # Consider entire row for duplicate detection
                row_hash = generate_row_hash(row)
            
            if row_hash not in seen_hashes:
                seen_hashes.add(row_hash)
                unique_rows.append(row)
    
    # Write unique rows to output file
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(headers)
        writer.writerows(unique_rows)
    
    return len(unique_rows)

def find_duplicate_stats(input_file, key_columns=None):
    """
    Analyze CSV file and return duplicate statistics.
    
    Returns:
        Dictionary with total_rows, unique_rows, duplicate_count
    """
    row_counts = defaultdict(int)
    
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        headers = next(reader)
        
        for row in reader:
            if key_columns:
                key_values = [row[i] for i in key_columns if i < len(row)]
                row_hash = generate_row_hash(key_values)
            else:
                row_hash = generate_row_hash(row)
            
            row_counts[row_hash] += 1
    
    total_rows = sum(row_counts.values())
    unique_rows = len(row_counts)
    duplicate_count = total_rows - unique_rows
    
    return {
        'total_rows': total_rows,
        'unique_rows': unique_rows,
        'duplicate_count': duplicate_count,
        'duplicate_percentage': (duplicate_count / total_rows * 100) if total_rows > 0 else 0
    }

if __name__ == "__main__":
    # Example usage
    input_csv = "input_data.csv"
    output_csv = "cleaned_data.csv"
    
    # Analyze duplicates
    stats = find_duplicate_stats(input_csv)
    print(f"Original rows: {stats['total_rows']}")
    print(f"Unique rows: {stats['unique_rows']}")
    print(f"Duplicates found: {stats['duplicate_count']}")
    print(f"Duplicate percentage: {stats['duplicate_percentage']:.2f}%")
    
    # Remove duplicates
    unique_count = remove_duplicates(input_csv, output_csv)
    print(f"Saved {unique_count} unique rows to {output_csv}")
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, columns):
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def normalize_minmax(df, columns):
    df_norm = df.copy()
    for col in columns:
        min_val = df_norm[col].min()
        max_val = df_norm[col].max()
        if max_val != min_val:
            df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
        else:
            df_norm[col] = 0
    return df_norm

def clean_dataset(file_path, numeric_columns):
    df = pd.read_csv(file_path)
    df_clean = remove_outliers_iqr(df, numeric_columns)
    df_normalized = normalize_minmax(df_clean, numeric_columns)
    return df_normalized

if __name__ == "__main__":
    cleaned_data = clean_dataset("sample_data.csv", ["age", "salary", "score"])
    cleaned_data.to_csv("cleaned_data.csv", index=False)
    print(f"Data cleaning complete. Original shape: {pd.read_csv('sample_data.csv').shape}, Cleaned shape: {cleaned_data.shape}")import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    if max_val - min_val == 0:
        return df[column].apply(lambda x: 0.0)
    return df[column].apply(lambda x: (x - min_val) / (max_val - min_val))

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    return cleaned_df.reset_index(drop=True)

def validate_data(df, required_columns):
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    return Trueimport numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = normalize_minmax(cleaned_df, col)
    return cleaned_df.reset_index(drop=True)

def process_dataframe(file_path):
    try:
        df = pd.read_csv(file_path)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        result_df = clean_dataset(df, numeric_cols)
        return result_df
    except Exception as e:
        print(f"Error processing file: {e}")
        return None