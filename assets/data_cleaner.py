
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, multiplier=1.5):
    """
    Remove outliers from a DataFrame column using IQR method.
    
    Args:
        dataframe: pandas DataFrame
        column: Column name to process
        multiplier: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    return filtered_df.copy()

def zscore_normalize(dataframe, column):
    """
    Normalize a column using z-score normalization.
    
    Args:
        dataframe: pandas DataFrame
        column: Column name to normalize
    
    Returns:
        Series with normalized values
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = dataframe[column].mean()
    std_val = dataframe[column].std()
    
    if std_val == 0:
        return dataframe[column]
    
    normalized = (dataframe[column] - mean_val) / std_val
    return normalized

def minmax_normalize(dataframe, column, feature_range=(0, 1)):
    """
    Normalize a column using min-max scaling.
    
    Args:
        dataframe: pandas DataFrame
        column: Column name to normalize
        feature_range: Desired range of transformed data (default 0-1)
    
    Returns:
        Series with normalized values
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = dataframe[column].min()
    max_val = dataframe[column].max()
    
    if max_val == min_val:
        return dataframe[column]
    
    normalized = (dataframe[column] - min_val) / (max_val - min_val)
    
    if feature_range != (0, 1):
        min_target, max_target = feature_range
        normalized = normalized * (max_target - min_target) + min_target
    
    return normalized

def detect_missing_patterns(dataframe, threshold=0.3):
    """
    Detect columns with high percentage of missing values.
    
    Args:
        dataframe: pandas DataFrame
        threshold: Missing value threshold (default 0.3 = 30%)
    
    Returns:
        List of column names exceeding the threshold
    """
    missing_ratios = dataframe.isnull().sum() / len(dataframe)
    high_missing_cols = missing_ratios[missing_ratios > threshold].index.tolist()
    
    return high_missing_cols

def clean_dataframe(dataframe, 
                   outlier_columns=None, 
                   normalize_columns=None, 
                   normalization_method='zscore',
                   missing_threshold=0.3):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        dataframe: pandas DataFrame to clean
        outlier_columns: List of columns for outlier removal
        normalize_columns: List of columns for normalization
        normalization_method: 'zscore' or 'minmax'
        missing_threshold: Threshold for removing high-missing columns
    
    Returns:
        Cleaned DataFrame
    """
    df_clean = dataframe.copy()
    
    if outlier_columns:
        for col in outlier_columns:
            if col in df_clean.columns:
                df_clean = remove_outliers_iqr(df_clean, col)
    
    high_missing = detect_missing_patterns(df_clean, missing_threshold)
    df_clean = df_clean.drop(columns=high_missing)
    
    if normalize_columns and normalization_method:
        for col in normalize_columns:
            if col in df_clean.columns:
                if normalization_method == 'zscore':
                    df_clean[f'{col}_normalized'] = zscore_normalize(df_clean, col)
                elif normalization_method == 'minmax':
                    df_clean[f'{col}_normalized'] = minmax_normalize(df_clean, col)
    
    return df_clean

def validate_dataframe(dataframe, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        dataframe: pandas DataFrame to validate
        required_columns: List of required column names
    
    Returns:
        Dictionary with validation results
    """
    validation_result = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    if dataframe.empty:
        validation_result['is_valid'] = False
        validation_result['errors'].append('DataFrame is empty')
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in dataframe.columns]
        if missing_cols:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f'Missing required columns: {missing_cols}')
    
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        validation_result['warnings'].append('No numeric columns found')
    
    duplicate_rows = dataframe.duplicated().sum()
    if duplicate_rows > 0:
        validation_result['warnings'].append(f'Found {duplicate_rows} duplicate rows')
    
    return validation_result

if __name__ == "__main__":
    sample_data = {
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.randint(1, 100, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    df = pd.DataFrame(sample_data)
    
    cleaned_df = clean_dataframe(
        df,
        outlier_columns=['feature_a', 'feature_b'],
        normalize_columns=['feature_a', 'feature_b'],
        normalization_method='zscore'
    )
    
    validation = validate_dataframe(cleaned_df)
    
    print(f"Original shape: {df.shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    print(f"Validation result: {validation}")