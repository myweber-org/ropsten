import pandas as pd
import numpy as np

def clean_csv_data(input_file, output_file, missing_strategy='mean'):
    """
    Load a CSV file, clean missing values, and save cleaned data.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to save cleaned CSV file
        missing_strategy (str): Strategy for handling missing values.
                                Options: 'mean', 'median', 'drop', 'zero'
    """
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
        
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Found {missing_count} missing values")
            
            if missing_strategy == 'drop':
                df_cleaned = df.dropna()
                print(f"Dropped rows with missing values. New shape: {df_cleaned.shape}")
            elif missing_strategy == 'mean':
                df_cleaned = df.fillna(df.mean(numeric_only=True))
                print("Filled missing values with column means")
            elif missing_strategy == 'median':
                df_cleaned = df.fillna(df.median(numeric_only=True))
                print("Filled missing values with column medians")
            elif missing_strategy == 'zero':
                df_cleaned = df.fillna(0)
                print("Filled missing values with zeros")
            else:
                print(f"Unknown strategy: {missing_strategy}. Using 'mean' as default")
                df_cleaned = df.fillna(df.mean(numeric_only=True))
        else:
            df_cleaned = df
            print("No missing values found")
        
        df_cleaned.to_csv(output_file, index=False)
        print(f"Cleaned data saved to {output_file}")
        
        return df_cleaned
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate dataframe structure and content.
    
    Args:
        df (pd.DataFrame): Dataframe to validate
        required_columns (list): List of required column names
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if df is None or df.empty:
        print("Error: Dataframe is empty or None")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        stats = df[numeric_cols].describe()
        print("Numeric columns statistics:")
        print(stats)
    
    return True

if __name__ == "__main__":
    cleaned_df = clean_csv_data('input_data.csv', 'cleaned_data.csv', 'mean')
    if cleaned_df is not None:
        validation_passed = validate_dataframe(cleaned_df)
        if validation_passed:
            print("Data cleaning completed successfully")
        else:
            print("Data cleaning completed but validation failed")
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
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
    Remove outliers using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column]))
    return data[z_scores < threshold]

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if min_val == max_val:
        return data[column].apply(lambda x: 0.5)
    
    return (data[column] - min_val) / (max_val - min_val)

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    return (data[column] - mean_val) / std_val

def clean_dataset(df, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Main function to clean dataset by removing outliers and normalizing numeric columns
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column not in df.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
        elif outlier_method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, column)
        
        if normalize_method == 'minmax':
            cleaned_df[column] = normalize_minmax(cleaned_df, column)
        elif normalize_method == 'zscore':
            cleaned_df[column] = normalize_zscore(cleaned_df, column)
    
    return cleaned_df

def validate_data(df, required_columns=None, allow_nan=False):
    """
    Validate data structure and content
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    if not allow_nan and df.isnull().any().any():
        raise ValueError("Dataset contains NaN values")
    
    return True
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Args:
        data (list or np.array): The dataset
        column (int): Index of the column to clean
    
    Returns:
        np.array: Data with outliers removed
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    column_data = data[:, column].astype(float)
    
    Q1 = np.percentile(column_data, 25)
    Q3 = np.percentile(column_data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    mask = (column_data >= lower_bound) & (column_data <= upper_bound)
    
    return data[mask]

def calculate_statistics(data, column):
    """
    Calculate basic statistics for a column.
    
    Args:
        data (np.array): The dataset
        column (int): Index of the column
    
    Returns:
        dict: Dictionary containing statistics
    """
    column_data = data[:, column].astype(float)
    
    stats = {
        'mean': np.mean(column_data),
        'median': np.median(column_data),
        'std': np.std(column_data),
        'min': np.min(column_data),
        'max': np.max(column_data),
        'q1': np.percentile(column_data, 25),
        'q3': np.percentile(column_data, 75)
    }
    
    return stats

def clean_dataset(data, columns_to_clean):
    """
    Clean multiple columns in a dataset.
    
    Args:
        data (list or np.array): The dataset
        columns_to_clean (list): List of column indices to clean
    
    Returns:
        np.array: Cleaned dataset
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    cleaned_data = data.copy()
    
    for column in columns_to_clean:
        if column < cleaned_data.shape[1]:
            cleaned_data = remove_outliers_iqr(cleaned_data, column)
    
    return cleaned_data

if __name__ == "__main__":
    sample_data = np.array([
        [1, 10.5, 100],
        [2, 12.3, 150],
        [3, 9.8, 120],
        [4, 50.0, 130],
        [5, 11.2, 110],
        [6, 9.5, 140],
        [7, 200.0, 160]
    ])
    
    print("Original data shape:", sample_data.shape)
    print("Original data:\n", sample_data)
    
    cleaned = clean_dataset(sample_data, [1])
    
    print("\nCleaned data shape:", cleaned.shape)
    print("Cleaned data:\n", cleaned)
    
    stats = calculate_statistics(cleaned, 1)
    print("\nStatistics for column 1:", stats)