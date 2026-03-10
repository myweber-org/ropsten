
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to drop duplicate rows
        fill_missing (bool): Whether to fill missing values
        fill_value: Value to use for filling missing data
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "Data validation passed"
import pandas as pd
import numpy as np

def clean_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        strategy (str): Strategy for imputation ('mean', 'median', 'mode', 'drop')
        columns (list): List of columns to process, None for all numeric columns
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        return df_clean.dropna(subset=columns)
    
    for col in columns:
        if strategy == 'mean':
            fill_value = df_clean[col].mean()
        elif strategy == 'median':
            fill_value = df_clean[col].median()
        elif strategy == 'mode':
            fill_value = df_clean[col].mode()[0]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        df_clean[col] = df_clean[col].fillna(fill_value)
    
    return df_clean

def remove_outliers_iqr(df, columns=None, multiplier=1.5):
    """
    Remove outliers using IQR method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of columns to process, None for all numeric columns
        multiplier (float): IQR multiplier for outlier detection
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.select_dtypes(include=[np.number]).columns
    
    mask = pd.Series([True] * len(df_clean))
    
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        col_mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
        mask = mask & col_mask
    
    return df_clean[mask]

def standardize_columns(df, columns=None):
    """
    Standardize numeric columns to have zero mean and unit variance.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of columns to standardize
    
    Returns:
        pd.DataFrame: DataFrame with standardized columns
    """
    df_standardized = df.copy()
    
    if columns is None:
        columns = df_standardized.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        mean = df_standardized[col].mean()
        std = df_standardized[col].std()
        
        if std > 0:
            df_standardized[col] = (df_standardized[col] - mean) / std
    
    return df_standardized

def get_data_summary(df):
    """
    Generate summary statistics for a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        dict: Dictionary containing summary statistics
    """
    summary = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.astype(str).to_dict(),
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

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, np.nan, 4, 5, 100],
        'B': [10, 20, 30, 40, np.nan, 60],
        'C': ['a', 'b', 'c', 'd', 'e', 'f']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nSummary:")
    print(get_data_summary(df))
    
    df_clean = clean_missing_values(df, strategy='mean')
    print("\nAfter cleaning missing values:")
    print(df_clean)
    
    df_no_outliers = remove_outliers_iqr(df_clean)
    print("\nAfter removing outliers:")
    print(df_no_outliers)
    
    df_standardized = standardize_columns(df_no_outliers)
    print("\nAfter standardization:")
    print(df_standardized)
import pandas as pd
import numpy as np
from scipy import stats

def load_and_clean_data(filepath):
    """
    Load CSV data and perform cleaning operations.
    """
    df = pd.read_csv(filepath)
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Handle missing values
    for column in df.select_dtypes(include=[np.number]).columns:
        df[column].fillna(df[column].median(), inplace=True)
    
    # Remove outliers using z-score method
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    df = df[(z_scores < 3).all(axis=1)]
    
    # Normalize numeric columns
    for column in numeric_cols:
        df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    
    return df

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to CSV.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = load_and_clean_data(input_file)
    save_cleaned_data(cleaned_df, output_file)
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a dataset using the Interquartile Range (IQR) method.
    
    Args:
        data: A pandas DataFrame containing the data.
        column: The name of the column to process for outliers.
    
    Returns:
        A pandas DataFrame with outliers removed.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    return filtered_data

def calculate_summary_statistics(data, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Args:
        data: A pandas DataFrame.
        column: The name of the column to analyze.
    
    Returns:
        A dictionary containing count, mean, std, min, and max.
    """
    stats = {
        'count': data[column].count(),
        'mean': data[column].mean(),
        'std': data[column].std(),
        'min': data[column].min(),
        'max': data[column].max()
    }
    return stats

if __name__ == "__main__":
    import pandas as pd
    
    sample_data = pd.DataFrame({
        'values': [10, 12, 12, 13, 12, 11, 10, 100, 12, 14, 15, 12, 11, 10, 9, 8, 12, 13, 14, 200]
    })
    
    print("Original data:")
    print(sample_data)
    print("\nOriginal statistics:")
    print(calculate_summary_statistics(sample_data, 'values'))
    
    cleaned_data = remove_outliers_iqr(sample_data, 'values')
    
    print("\nCleaned data:")
    print(cleaned_data)
    print("\nCleaned statistics:")
    print(calculate_summary_statistics(cleaned_data, 'values'))import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    """
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data to [0, 1] range using Min-Max scaling.
    """
    min_val = data[column].min()
    max_val = data[column].max()
    data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    return data

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization.
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    data[column + '_standardized'] = (data[column] - mean_val) / std_val
    return data

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    """
    Main function to clean dataset by removing outliers and normalizing numeric columns.
    """
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if outlier_method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        elif outlier_method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
        
        if normalize_method == 'minmax':
            cleaned_df = normalize_minmax(cleaned_df, col)
        elif normalize_method == 'zscore':
            cleaned_df = normalize_zscore(cleaned_df, col)
    
    return cleaned_df

def validate_data(data, column, expected_type=np.number):
    """
    Validate data type and check for missing values.
    """
    if not np.issubdtype(data[column].dtype, expected_type):
        raise TypeError(f"Column {column} must be numeric")
    
    missing_count = data[column].isnull().sum()
    if missing_count > 0:
        print(f"Warning: Column {column} has {missing_count} missing values")
    
    return True
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows.
        fill_missing (str or dict): Method to fill missing values.
            Options: 'mean', 'median', 'mode', or a dictionary of column:value pairs.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing is not None:
        if isinstance(fill_missing, dict):
            cleaned_df = cleaned_df.fillna(fill_missing)
        elif fill_missing == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_missing == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif fill_missing == 'mode':
            cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate that a DataFrame meets basic requirements.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
        min_rows (int): Minimum number of rows required.
    
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
    
    return True, "Data validation passed"
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
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
    
    return filtered_df.reset_index(drop=True)

def calculate_summary_statistics(df):
    """
    Calculate summary statistics for numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    pd.DataFrame: Summary statistics
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    summary = df[numeric_cols].describe()
    
    summary.loc['variance'] = df[numeric_cols].var()
    summary.loc['range'] = summary.loc['max'] - summary.loc['min']
    
    return summary

def handle_missing_values(df, strategy='mean'):
    """
    Handle missing values in numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): Imputation strategy ('mean', 'median', 'mode', 'drop')
    
    Returns:
    pd.DataFrame: DataFrame with handled missing values
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        return df.dropna(subset=numeric_cols).reset_index(drop=True)
    
    df_filled = df.copy()
    
    for col in numeric_cols:
        if strategy == 'mean':
            fill_value = df[col].mean()
        elif strategy == 'median':
            fill_value = df[col].median()
        elif strategy == 'mode':
            fill_value = df[col].mode()[0] if not df[col].mode().empty else 0
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        df_filled[col] = df[col].fillna(fill_value)
    
    return df_filled

def normalize_data(df, columns=None):
    """
    Normalize specified columns using min-max scaling.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of columns to normalize. If None, normalize all numeric columns.
    
    Returns:
    pd.DataFrame: DataFrame with normalized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_normalized = df.copy()
    
    for col in columns:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            
            if max_val != min_val:
                df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                df_normalized[col] = 0
    
    return df_normalized

def validate_dataframe(df):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    dict: Validation results
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(df.select_dtypes(include=['object']).columns),
        'date_columns': list(df.select_dtypes(include=['datetime']).columns)
    }
    
    return validation_results
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas DataFrame column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed from the specified column.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def calculate_summary_statistics(data, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
    column (str): The column name to analyze.
    
    Returns:
    dict: Dictionary containing count, mean, std, min, and max.
    """
    stats = {
        'count': data[column].count(),
        'mean': data[column].mean(),
        'std': data[column].std(),
        'min': data[column].min(),
        'max': data[column].max()
    }
    return stats
import pandas as pd
import numpy as np
import sys

def clean_data(input_file, output_file):
    try:
        df = pd.read_csv(input_file)
        
        df = df.drop_duplicates()
        
        df = df.dropna(subset=['critical_column'])
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df[col] = df[col].fillna(df[col].mean())
        
        text_columns = df.select_dtypes(include=['object']).columns
        for col in text_columns:
            df[col] = df[col].fillna('unknown')
        
        df['date_column'] = pd.to_datetime(df['date_column'], errors='coerce')
        
        df = df[df['value_column'] > 0]
        
        df.to_csv(output_file, index=False)
        print(f"Data cleaned successfully. Output saved to {output_file}")
        
        return True
        
    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found.")
        return False
    except pd.errors.EmptyDataError:
        print("Error: Input file is empty.")
        return False
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python data_cleaner.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    success = clean_data(input_file, output_file)
    sys.exit(0 if success else 1)
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Args:
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

def calculate_statistics(data):
    """
    Calculate basic statistics for the data.
    
    Args:
        data (np.ndarray): Input data array
    
    Returns:
        dict: Dictionary containing mean, median, and standard deviation
    """
    if data.size == 0:
        return {'mean': 0, 'median': 0, 'std': 0}
    
    return {
        'mean': np.mean(data, axis=0),
        'median': np.median(data, axis=0),
        'std': np.std(data, axis=0)
    }

def normalize_data(data):
    """
    Normalize data using min-max scaling.
    
    Args:
        data (np.ndarray): Input data array
    
    Returns:
        np.ndarray: Normalized data
    """
    if data.size == 0:
        return data
    
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    
    # Avoid division by zero
    range_vals = data_max - data_min
    range_vals[range_vals == 0] = 1
    
    normalized = (data - data_min) / range_vals
    return normalized

def process_dataset(data, target_column):
    """
    Complete data processing pipeline.
    
    Args:
        data (np.ndarray): Input data array
        target_column (int): Column index for outlier removal
    
    Returns:
        tuple: (cleaned_data, statistics, normalized_data)
    """
    print(f"Original data shape: {data.shape}")
    
    # Remove outliers
    cleaned_data = remove_outliers_iqr(data, target_column)
    print(f"Cleaned data shape: {cleaned_data.shape}")
    
    # Calculate statistics
    stats = calculate_statistics(cleaned_data)
    print(f"Statistics calculated")
    
    # Normalize data
    normalized = normalize_data(cleaned_data)
    print(f"Data normalized")
    
    return cleaned_data, stats, normalized