
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using Interquartile Range method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    outliers_removed = len(data) - len(filtered_data)
    
    return filtered_data, outliers_removed

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using Z-score normalization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def handle_missing_values(data, strategy='mean'):
    """
    Handle missing values in numerical columns
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    if strategy == 'mean':
        for col in numeric_cols:
            data[col] = data[col].fillna(data[col].mean())
    elif strategy == 'median':
        for col in numeric_cols:
            data[col] = data[col].fillna(data[col].median())
    elif strategy == 'mode':
        for col in numeric_cols:
            data[col] = data[col].fillna(data[col].mode()[0])
    elif strategy == 'drop':
        data = data.dropna(subset=numeric_cols)
    else:
        raise ValueError("Strategy must be 'mean', 'median', 'mode', or 'drop'")
    
    return data

def detect_skewness(data, column, threshold=0.5):
    """
    Detect skewness in data column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    skewness = stats.skew(data[column].dropna())
    is_skewed = abs(skewness) > threshold
    
    return {
        'skewness': skewness,
        'is_skewed': is_skewed,
        'interpretation': 'positively skewed' if skewness > 0 else 'negatively skewed' if skewness < 0 else 'symmetric'
    }

def log_transform(data, column):
    """
    Apply log transformation to reduce skewness
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if data[column].min() <= 0:
        shifted_data = data[column] - data[column].min() + 1
        transformed = np.log(shifted_data)
    else:
        transformed = np.log(data[column])
    
    return transformed

def create_summary_report(data):
    """
    Create a comprehensive data quality report
    """
    report = {
        'total_rows': len(data),
        'total_columns': len(data.columns),
        'missing_values': data.isnull().sum().to_dict(),
        'data_types': data.dtypes.to_dict(),
        'numeric_summary': {},
        'categorical_summary': {}
    }
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    
    for col in numeric_cols:
        report['numeric_summary'][col] = {
            'mean': data[col].mean(),
            'median': data[col].median(),
            'std': data[col].std(),
            'min': data[col].min(),
            'max': data[col].max(),
            'skewness': stats.skew(data[col].dropna())
        }
    
    for col in categorical_cols:
        report['categorical_summary'][col] = {
            'unique_values': data[col].nunique(),
            'top_value': data[col].mode()[0] if len(data[col].mode()) > 0 else None,
            'top_count': data[col].value_counts().iloc[0] if len(data[col]) > 0 else 0
        }
    
    return report

def validate_dataframe(data):
    """
    Validate DataFrame structure and content
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if data.empty:
        raise ValueError("DataFrame is empty")
    
    if len(data.columns) == 0:
        raise ValueError("DataFrame has no columns")
    
    return True
import pandas as pd
import re

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).strip().lower()
    text = re.sub(r'\s+', ' ', text)
    return text

def remove_duplicates(df, column_name):
    df[column_name] = df[column_name].apply(clean_text)
    df = df.drop_duplicates(subset=[column_name], keep='first')
    return df

def process_data(input_file, output_file, column_to_clean):
    try:
        df = pd.read_csv(input_file)
        df_cleaned = remove_duplicates(df, column_to_clean)
        df_cleaned.to_csv(output_file, index=False)
        print(f"Data cleaned and saved to {output_file}")
        print(f"Removed {len(df) - len(df_cleaned)} duplicate entries")
    except Exception as e:
        print(f"Error processing data: {e}")

if __name__ == "__main__":
    process_data("raw_data.csv", "cleaned_data.csv", "description")
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to clean
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
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
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to analyze
    
    Returns:
        dict: Dictionary containing summary statistics
    """
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': len(df)
    }
    
    return stats

def process_dataframe(df, column):
    """
    Main function to process DataFrame by removing outliers and calculating statistics.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to process
    
    Returns:
        tuple: (cleaned_df, original_stats, cleaned_stats)
    """
    original_stats = calculate_summary_statistics(df, column)
    cleaned_df = remove_outliers_iqr(df, column)
    cleaned_stats = calculate_summary_statistics(cleaned_df, column)
    
    return cleaned_df, original_stats, cleaned_stats
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    return cleaned_df.reset_index(drop=True)

def main():
    data = {'values': [10, 12, 12, 13, 12, 11, 10, 100, 12, 14, 15, 10, 9, 8, 200]}
    df = pd.DataFrame(data)
    print("Original data:")
    print(df)
    
    cleaned_df = clean_dataset(df, ['values'])
    print("\nCleaned data:")
    print(cleaned_df)
    
    print(f"\nRemoved {len(df) - len(cleaned_df)} outliers")

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np

def detect_outliers_iqr(data, column):
    """
    Detect outliers in a specified column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to analyze
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers_mask = (data[column] < lower_bound) | (data[column] > upper_bound)
    cleaned_data = data[~outliers_mask].copy()
    
    outliers_count = outliers_mask.sum()
    print(f"Removed {outliers_count} outliers from column '{column}'")
    
    return cleaned_data

def remove_missing_values(data, strategy='drop', fill_value=None):
    """
    Handle missing values in dataframe.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    strategy (str): 'drop' to remove rows, 'fill' to fill values
    fill_value: Value to fill missing data with (if strategy='fill')
    
    Returns:
    pd.DataFrame: Processed dataframe
    """
    if strategy == 'drop':
        cleaned_data = data.dropna()
        print(f"Removed {len(data) - len(cleaned_data)} rows with missing values")
    elif strategy == 'fill':
        if fill_value is None:
            fill_value = data.mean(numeric_only=True)
        cleaned_data = data.fillna(fill_value)
        print("Filled missing values")
    else:
        raise ValueError("Strategy must be 'drop' or 'fill'")
    
    return cleaned_data

def clean_dataset(data, numeric_columns=None, missing_strategy='drop'):
    """
    Comprehensive data cleaning function.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    numeric_columns (list): List of numeric column names for outlier detection
    missing_strategy (str): Strategy for handling missing values
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    original_shape = data.shape
    
    cleaned_data = data.copy()
    
    cleaned_data = remove_missing_values(cleaned_data, strategy=missing_strategy)
    
    if numeric_columns:
        for column in numeric_columns:
            if column in cleaned_data.columns:
                cleaned_data = detect_outliers_iqr(cleaned_data, column)
    
    final_shape = cleaned_data.shape
    rows_removed = original_shape[0] - final_shape[0]
    
    print(f"Data cleaning complete. Removed {rows_removed} rows.")
    print(f"Original shape: {original_shape}, Cleaned shape: {final_shape}")
    
    return cleaned_data

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, 3, 100, 5, 6, 7, 8, 9, 10],
        'B': [10, 20, 30, 40, 50, 60, 70, 80, 90, 1000],
        'C': [1.1, 2.2, None, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0]
    })
    
    print("Original data:")
    print(sample_data)
    print("\n")
    
    cleaned = clean_dataset(sample_data, numeric_columns=['A', 'B'], missing_strategy='fill')
    
    print("\nCleaned data:")
    print(cleaned)import csv
import sys
from pathlib import Path

def remove_duplicates(input_file, output_file=None, key_column=None):
    """
    Remove duplicate rows from a CSV file.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file (optional)
        key_column: Column name to identify duplicates (optional)
    
    Returns:
        Number of duplicates removed
    """
    if output_file is None:
        output_file = input_file.replace('.csv', '_cleaned.csv')
    
    seen = set()
    duplicates_removed = 0
    
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in reader:
                if key_column:
                    key = row[key_column]
                else:
                    key = tuple(row.values())
                
                if key not in seen:
                    seen.add(key)
                    writer.writerow(row)
                else:
                    duplicates_removed += 1
    
    return duplicates_removed

def main():
    if len(sys.argv) < 2:
        print("Usage: python data_cleaner.py <input_file> [output_file] [key_column]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    key_column = sys.argv[3] if len(sys.argv) > 3 else None
    
    if not Path(input_file).exists():
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)
    
    try:
        removed = remove_duplicates(input_file, output_file, key_column)
        output = output_file if output_file else input_file.replace('.csv', '_cleaned.csv')
        print(f"Removed {removed} duplicate rows")
        print(f"Cleaned data saved to: {output}")
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing=True, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
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
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
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

def process_numeric_columns(df, columns=None):
    """
    Process numeric columns by converting to appropriate types and handling outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns (list): Specific columns to process. If None, all numeric columns are processed.
    
    Returns:
    pd.DataFrame: DataFrame with processed numeric columns.
    """
    processed_df = df.copy()
    
    if columns is None:
        numeric_cols = processed_df.select_dtypes(include=['int64', 'float64']).columns
    else:
        numeric_cols = [col for col in columns if col in processed_df.columns]
    
    for col in numeric_cols:
        if processed_df[col].dtype in ['int64', 'float64']:
            # Convert to float for consistency
            processed_df[col] = processed_df[col].astype('float64')
    
    return processed_df