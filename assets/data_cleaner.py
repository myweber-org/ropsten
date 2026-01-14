import pandas as pd
import numpy as np
from scipy import stats

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
    cleaned_df = cleaned_df.dropna()
    return cleaned_df.reset_index(drop=True)

def calculate_statistics(df, column):
    if column not in df.columns:
        return None
    data = df[column].dropna()
    return {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'skewness': stats.skew(data),
        'kurtosis': stats.kurtosis(data)
    }
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
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
    
    return filtered_df

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

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    data = {
        'values': np.concatenate([
            np.random.normal(100, 15, 95),
            np.random.normal(300, 50, 5)
        ])
    }
    
    df = pd.DataFrame(data)
    print(f"Original data shape: {df.shape}")
    print(f"Original statistics: {calculate_summary_statistics(df, 'values')}")
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    print(f"\nCleaned data shape: {cleaned_df.shape}")
    print(f"Cleaned statistics: {calculate_summary_statistics(cleaned_df, 'values')}")import numpy as np
import pandas as pd
from scipy import stats

def normalize_data(data, method='zscore'):
    if method == 'zscore':
        return (data - np.mean(data)) / np.std(data)
    elif method == 'minmax':
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    else:
        raise ValueError("Method must be 'zscore' or 'minmax'")

def remove_outliers_iqr(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return data[(data >= lower_bound) & (data <= upper_bound)]

def remove_outliers_zscore(data, threshold=3):
    z_scores = np.abs(stats.zscore(data))
    return data[z_scores < threshold]

def clean_dataset(df, column, outlier_method='iqr', normalize=False):
    if column not in df.columns:
        raise ValueError(f"Column {column} not found in DataFrame")
    
    cleaned_data = df[column].copy()
    
    if outlier_method == 'iqr':
        cleaned_data = remove_outliers_iqr(cleaned_data)
    elif outlier_method == 'zscore':
        cleaned_data = remove_outliers_zscore(cleaned_data)
    else:
        raise ValueError("Outlier method must be 'iqr' or 'zscore'")
    
    if normalize:
        cleaned_data = normalize_data(cleaned_data)
    
    return cleaned_data

def process_dataframe(df, numeric_columns=None, outlier_method='iqr', normalize=True):
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in df.columns:
            try:
                cleaned_df[col] = clean_dataset(df, col, outlier_method, normalize)
            except Exception as e:
                print(f"Error processing column {col}: {e}")
                continue
    
    return cleaned_df.dropna()

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    })
    
    print("Original data shape:", sample_data.shape)
    print("Original data statistics:")
    print(sample_data.describe())
    
    cleaned_data = process_dataframe(sample_data)
    
    print("\nCleaned data shape:", cleaned_data.shape)
    print("Cleaned data statistics:")
    print(cleaned_data.describe())
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to remove duplicate rows. Default is True.
    fill_missing (str): Method to fill missing values. Options: 'mean', 'median', 'mode', or 'drop'. Default is 'mean'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = cleaned_df.shape[0]
        cleaned_df.drop_duplicates(inplace=True)
        removed = initial_rows - cleaned_df.shape[0]
        print(f"Removed {removed} duplicate rows.")
    
    if fill_missing == 'drop':
        cleaned_df.dropna(inplace=True)
        print("Dropped rows with missing values.")
    elif fill_missing in ['mean', 'median', 'mode']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if cleaned_df[col].isnull().any():
                if fill_missing == 'mean':
                    fill_value = cleaned_df[col].mean()
                elif fill_missing == 'median':
                    fill_value = cleaned_df[col].median()
                elif fill_missing == 'mode':
                    fill_value = cleaned_df[col].mode()[0]
                cleaned_df[col].fillna(fill_value, inplace=True)
                print(f"Filled missing values in column '{col}' with {fill_missing}: {fill_value:.2f}")
    
    categorical_cols = cleaned_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if cleaned_df[col].isnull().any():
            cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)
            print(f"Filled missing values in categorical column '{col}' with mode.")
    
    print(f"Cleaning complete. Final dataset shape: {cleaned_df.shape}")
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the DataFrame for required columns and minimum row count.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing required columns: {missing_cols}")
            return False
    
    if df.shape[0] < min_rows:
        print(f"Validation failed: Dataset has {df.shape[0]} rows, minimum required is {min_rows}")
        return False
    
    print("Data validation passed.")
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, np.nan],
        'B': [5, np.nan, 7, 8, 9],
        'C': ['x', 'y', 'y', 'z', np.nan]
    }
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    
    cleaned = clean_dataset(df, fill_missing='median')
    print("\nCleaned dataset:")
    print(cleaned)
    
    is_valid = validate_data(cleaned, required_columns=['A', 'B'], min_rows=3)
    print(f"\nDataset valid: {is_valid}")