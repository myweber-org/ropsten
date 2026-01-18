import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    z_scores = np.abs(stats.zscore(data[column]))
    return data[z_scores < threshold]

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    min_val = data[column].min()
    max_val = data[column].max()
    data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    return data

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    data[column + '_standardized'] = (data[column] - mean_val) / std_val
    return data

def clean_dataset(df, numeric_columns, method='iqr', normalization='minmax'):
    """
    Main cleaning function for datasets
    """
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        elif method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
        
        if normalization == 'minmax':
            cleaned_df = normalize_minmax(cleaned_df, col)
        elif normalization == 'zscore':
            cleaned_df = normalize_zscore(cleaned_df, col)
    
    return cleaned_df

def validate_data(df, required_columns, numeric_threshold=0.8):
    """
    Validate dataset structure and quality
    """
    validation_report = {}
    
    # Check required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    validation_report['missing_columns'] = missing_columns
    
    # Check for null values
    null_counts = df.isnull().sum()
    validation_report['null_counts'] = null_counts[null_counts > 0].to_dict()
    
    # Check numeric columns quality
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_stats = {}
    for col in numeric_cols:
        if df[col].notnull().sum() > 0:
            numeric_stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'missing_pct': df[col].isnull().mean()
            }
    validation_report['numeric_stats'] = numeric_stats
    
    return validation_report
import pandas as pd
import numpy as np

def clean_dataset(df, drop_na=True, rename_columns=True):
    """
    Clean a pandas DataFrame by handling missing values and standardizing column names.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_na (bool): If True, drop rows with any null values.
    rename_columns (bool): If True, rename columns to lowercase with underscores.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    if drop_na:
        df_clean = df_clean.dropna()
    
    if rename_columns:
        df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_')
    
    return df_clean

def validate_numeric_columns(df, columns):
    """
    Validate that specified columns contain only numeric values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns (list): List of column names to validate.
    
    Returns:
    dict: Dictionary with column names as keys and validation results as values.
    """
    validation_results = {}
    
    for col in columns:
        if col in df.columns:
            non_numeric = df[~df[col].apply(lambda x: isinstance(x, (int, float, np.number)))]
            validation_results[col] = {
                'is_valid': len(non_numeric) == 0,
                'non_numeric_count': len(non_numeric),
                'non_numeric_indices': non_numeric.index.tolist()
            }
        else:
            validation_results[col] = {
                'is_valid': False,
                'error': f"Column '{col}' not found in DataFrame"
            }
    
    return validation_results

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    subset (list): Columns to consider for identifying duplicates.
    keep (str): Which duplicates to keep ('first', 'last', False).
    
    Returns:
    pd.DataFrame: DataFrame with duplicates removed.
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def standardize_text(df, columns):
    """
    Standardize text columns by converting to lowercase and stripping whitespace.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns (list): List of text column names to standardize.
    
    Returns:
    pd.DataFrame: DataFrame with standardized text columns.
    """
    df_std = df.copy()
    
    for col in columns:
        if col in df_std.columns and df_std[col].dtype == 'object':
            df_std[col] = df_std[col].astype(str).str.lower().str.strip()
    
    return df_std

if __name__ == "__main__":
    sample_data = {
        'Name': ['Alice', 'Bob', None, 'Alice'],
        'Age': [25, 30, 35, 25],
        'Salary': [50000, 60000, 75000, 50000]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned_df = clean_dataset(df)
    print(cleaned_df)
    
    validation = validate_numeric_columns(cleaned_df, ['Age', 'Salary'])
    print("\nNumeric Validation Results:")
    print(validation)
    
    deduplicated = remove_duplicates(cleaned_df)
    print("\nDataFrame after removing duplicates:")
    print(deduplicated)