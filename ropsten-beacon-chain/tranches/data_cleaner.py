
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    filtered_indices = np.where(z_scores < threshold)[0]
    filtered_data = data.iloc[filtered_indices]
    return filtered_data

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

def clean_dataset(data, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline
    """
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column not in cleaned_data.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_data = remove_outliers_iqr(cleaned_data, column)
        elif outlier_method == 'zscore':
            cleaned_data = remove_outliers_zscore(cleaned_data, column)
        
        if normalize_method == 'minmax':
            cleaned_data[f'{column}_normalized'] = normalize_minmax(cleaned_data, column)
        elif normalize_method == 'zscore':
            cleaned_data[f'{column}_standardized'] = normalize_zscore(cleaned_data, column)
    
    return cleaned_data

def get_summary_statistics(data, numeric_columns):
    """
    Generate summary statistics for numeric columns
    """
    summary = {}
    
    for column in numeric_columns:
        if column in data.columns:
            col_data = data[column].dropna()
            summary[column] = {
                'count': len(col_data),
                'mean': col_data.mean(),
                'std': col_data.std(),
                'min': col_data.min(),
                '25%': col_data.quantile(0.25),
                'median': col_data.median(),
                '75%': col_data.quantile(0.75),
                'max': col_data.max(),
                'skewness': col_data.skew(),
                'kurtosis': col_data.kurtosis()
            }
    
    return pd.DataFrame(summary).T

def detect_missing_patterns(data, threshold=0.3):
    """
    Detect missing data patterns and suggest handling strategies
    """
    missing_stats = {}
    
    for column in data.columns:
        missing_count = data[column].isnull().sum()
        missing_percentage = missing_count / len(data)
        
        missing_stats[column] = {
            'missing_count': missing_count,
            'missing_percentage': missing_percentage,
            'suggestion': 'drop' if missing_percentage > threshold else 'impute'
        }
    
    return pd.DataFrame(missing_stats).T
import pandas as pd

def clean_dataset(df, columns_to_check=None, fill_missing=True):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        columns_to_check (list, optional): Specific columns to check for duplicates.
                                          If None, checks all columns.
        fill_missing (bool): Whether to fill missing values with column mean (numeric)
                            or mode (categorical)
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove duplicates
    if columns_to_check:
        cleaned_df = cleaned_df.drop_duplicates(subset=columns_to_check)
    else:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Handle missing values
    if fill_missing:
        for column in cleaned_df.columns:
            if cleaned_df[column].dtype in ['int64', 'float64']:
                # Fill numeric columns with mean
                mean_value = cleaned_df[column].mean()
                cleaned_df[column] = cleaned_df[column].fillna(mean_value)
            else:
                # Fill categorical columns with mode
                mode_value = cleaned_df[column].mode()[0] if not cleaned_df[column].mode().empty else 'Unknown'
                cleaned_df[column] = cleaned_df[column].fillna(mode_value)
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate that DataFrame meets basic quality requirements.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list, optional): List of columns that must be present
    
    Returns:
        dict: Dictionary with validation results
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        validation_results['missing_required_columns'] = missing_columns
        validation_results['all_required_columns_present'] = len(missing_columns) == 0
    
    return validation_results

# Example usage (commented out for production)
# if __name__ == "__main__":
#     # Create sample data
#     sample_data = {
#         'id': [1, 2, 2, 3, 4, None],
#         'name': ['Alice', 'Bob', 'Bob', 'Charlie', None, 'Eve'],
#         'age': [25, 30, 30, None, 35, 28],
#         'score': [85.5, 92.0, 92.0, 78.5, None, 88.0]
#     }
#     
#     df = pd.DataFrame(sample_data)
#     print("Original DataFrame:")
#     print(df)
#     print("\nValidation Results:")
#     print(validate_data(df))
#     
#     cleaned = clean_dataset(df)
#     print("\nCleaned DataFrame:")
#     print(cleaned)
#     print("\nCleaned Validation Results:")
#     print(validate_data(cleaned))
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
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

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': len(df[column])
    }
    
    return stats

def process_dataframe(df, columns_to_clean):
    """
    Process multiple columns for outlier removal and return cleaned DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns_to_clean (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    dict: Dictionary of statistics for each cleaned column
    """
    cleaned_df = df.copy()
    statistics = {}
    
    for column in columns_to_clean:
        if column in cleaned_df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            statistics[column] = {
                'removed_outliers': removed_count,
                'summary': calculate_summary_statistics(cleaned_df, column)
            }
    
    return cleaned_df, statistics

if __name__ == "__main__":
    sample_data = {
        'temperature': [22, 23, 24, 25, 26, 27, 28, 29, 100, -10],
        'humidity': [45, 50, 55, 60, 65, 70, 75, 80, 200, 10],
        'pressure': [1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 2000, 500]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    columns_to_process = ['temperature', 'humidity', 'pressure']
    cleaned_df, stats = process_dataframe(df, columns_to_process)
    
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\n" + "="*50 + "\n")
    
    print("Processing Statistics:")
    for col, data in stats.items():
        print(f"\nColumn: {col}")
        print(f"Removed outliers: {data['removed_outliers']}")
        print(f"Summary statistics: {data['summary']}")