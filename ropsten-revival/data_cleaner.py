
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    mask = z_scores < threshold
    
    filtered_data = data[mask]
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

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
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(df, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Main function to clean dataset by removing outliers and normalizing numeric columns
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    stats_report = {}
    
    for col in numeric_columns:
        if col not in df.columns:
            continue
            
        original_count = len(cleaned_df)
        
        if outlier_method == 'iqr':
            cleaned_df, removed = remove_outliers_iqr(cleaned_df, col)
        elif outlier_method == 'zscore':
            cleaned_df, removed = remove_outliers_zscore(cleaned_df, col)
        else:
            removed = 0
        
        if normalize_method == 'minmax':
            cleaned_df[f'{col}_normalized'] = normalize_minmax(cleaned_df, col)
        elif normalize_method == 'zscore':
            cleaned_df[f'{col}_standardized'] = normalize_zscore(cleaned_df, col)
        
        stats_report[col] = {
            'original_samples': original_count,
            'removed_outliers': removed,
            'remaining_samples': len(cleaned_df)
        }
    
    return cleaned_df, stats_report

def validate_data(df, required_columns=None, allow_nan_ratio=0.1):
    """
    Validate dataset structure and quality
    """
    validation_results = {
        'is_valid': True,
        'missing_columns': [],
        'high_nan_columns': [],
        'validation_errors': []
    }
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_results['missing_columns'] = missing_cols
            validation_results['is_valid'] = False
            validation_results['validation_errors'].append(f"Missing required columns: {missing_cols}")
    
    for col in df.columns:
        nan_ratio = df[col].isna().sum() / len(df)
        if nan_ratio > allow_nan_ratio:
            validation_results['high_nan_columns'].append({
                'column': col,
                'nan_ratio': nan_ratio
            })
            validation_results['validation_errors'].append(
                f"Column '{col}' has {nan_ratio:.1%} missing values"
            )
    
    if validation_results['high_nan_columns']:
        validation_results['is_valid'] = False
    
    return validation_resultsimport numpy as np
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
    if max_val == min_val:
        return df[column].apply(lambda x: 0.5)
    return (df[column] - min_val) / (max_val - min_val)

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
    return True

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 200),
        'feature_b': np.random.exponential(50, 200),
        'category': np.random.choice(['A', 'B', 'C'], 200)
    })
    
    print("Original shape:", sample_data.shape)
    cleaned = clean_dataset(sample_data, ['feature_a', 'feature_b'])
    print("Cleaned shape:", cleaned.shape)
    print("Sample cleaned data:")
    print(cleaned.head())
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd
import re

def clean_string_column(series, case='lower', remove_special=True):
    """
    Standardize string series by adjusting case and removing special characters.
    
    Args:
        series (pd.Series): Input string series
        case (str): 'lower', 'upper', or 'title' for case conversion
        remove_special (bool): Whether to remove non-alphanumeric characters
    
    Returns:
        pd.Series: Cleaned string series
    """
    if not pd.api.types.is_string_dtype(series):
        series = series.astype(str)
    
    result = series.copy()
    
    if case == 'lower':
        result = result.str.lower()
    elif case == 'upper':
        result = result.str.upper()
    elif case == 'title':
        result = result.str.title()
    
    if remove_special:
        result = result.apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x) if pd.notna(x) else x)
    
    return result

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows with additional logging.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        subset (list): Columns to consider for duplicates
        keep (str): 'first', 'last', or False to drop all duplicates
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """
    initial_count = len(df)
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
    final_count = len(cleaned_df)
    
    duplicates_removed = initial_count - final_count
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate rows")
    
    return cleaned_df

def standardize_dataframe(df, string_columns=None, case='lower'):
    """
    Apply cleaning operations to multiple columns in a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        string_columns (list): Columns to clean, defaults to all object columns
        case (str): Case standardization for string columns
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if string_columns is None:
        string_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    cleaned_df = df.copy()
    
    for col in string_columns:
        if col in cleaned_df.columns:
            cleaned_df[col] = clean_string_column(cleaned_df[col], case=case)
    
    cleaned_df = remove_duplicates(cleaned_df)
    
    return cleaned_df

def validate_email_format(series):
    """
    Validate email format in a series and return boolean mask.
    
    Args:
        series (pd.Series): Series containing email addresses
    
    Returns:
        pd.Series: Boolean series indicating valid emails
    """
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return series.str.match(email_pattern, na=False)

if __name__ == "__main__":
    sample_data = {
        'name': ['John Doe', 'Jane Smith', 'john doe', 'Bob Johnson', 'Jane Smith'],
        'email': ['john@example.com', 'jane@test.org', 'invalid-email', 'bob@company.net', 'jane@test.org'],
        'age': [25, 30, 25, 35, 30]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned = standardize_dataframe(df)
    print(cleaned)
    
    print("\nEmail validation:")
    df['valid_email'] = validate_email_format(df['email'])
    print(df[['email', 'valid_email']])