
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
        return col_data * 0  # Return zeros if all values are same
    
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
        return col_data * 0  # Return zeros if no variance
    
    return (col_data - mean_val) / std_val

def clean_dataset(df, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: Input DataFrame
        numeric_columns: List of numeric columns to process (default: all numeric)
        outlier_method: 'iqr', 'zscore', or None
        normalize_method: 'minmax', 'zscore', or None
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    # Remove outliers
    if outlier_method:
        for col in numeric_columns:
            if col in cleaned_df.columns:
                if outlier_method == 'iqr':
                    cleaned_df = remove_outliers_iqr(cleaned_df, col)
                elif outlier_method == 'zscore':
                    cleaned_df = remove_outliers_zscore(cleaned_df, col)
    
    # Normalize data
    if normalize_method:
        for col in numeric_columns:
            if col in cleaned_df.columns:
                if normalize_method == 'minmax':
                    cleaned_df[col] = normalize_minmax(cleaned_df, col)
                elif normalize_method == 'zscore':
                    cleaned_df[col] = normalize_zscore(cleaned_df, col)
    
    return cleaned_df

def validate_data(df, required_columns=None, allow_nan=True, max_nan_ratio=0.1):
    """
    Validate dataset for common issues.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required columns
        allow_nan: Whether NaN values are allowed
        max_nan_ratio: Maximum allowed ratio of NaN values per column
    
    Returns:
        Tuple of (is_valid, issues_list)
    """
    issues = []
    
    # Check required columns
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
    
    # Check for NaN values
    if not allow_nan:
        nan_columns = df.columns[df.isna().any()].tolist()
        if nan_columns:
            issues.append(f"NaN values found in columns: {nan_columns}")
    else:
        # Check for excessive NaN values
        for col in df.columns:
            nan_ratio = df[col].isna().mean()
            if nan_ratio > max_nan_ratio:
                issues.append(f"Column '{col}' has {nan_ratio:.1%} NaN values (max allowed: {max_nan_ratio:.1%})")
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.any(np.isinf(df[col])):
            issues.append(f"Infinite values found in column: {col}")
    
    # Check data types
    for col in df.columns:
        if df[col].dtype == object:
            unique_count = df[col].nunique()
            if unique_count > len(df) * 0.5:
                issues.append(f"Column '{col}' has high cardinality ({unique_count} unique values)")
    
    is_valid = len(issues) == 0
    return is_valid, issues

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.uniform(0, 1, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    # Add some outliers
    sample_data.loc[50, 'feature_a'] = 500
    sample_data.loc[150, 'feature_b'] = 1000
    
    print("Original data shape:", sample_data.shape)
    print("Original data summary:")
    print(sample_data.describe())
    
    # Clean the data
    cleaned_data = clean_dataset(
        sample_data,
        numeric_columns=['feature_a', 'feature_b', 'feature_c'],
        outlier_method='iqr',
        normalize_method='minmax'
    )
    
    print("\nCleaned data shape:", cleaned_data.shape)
    print("Cleaned data summary:")
    print(cleaned_data.describe())
    
    # Validate the cleaned data
    is_valid, issues = validate_data(cleaned_data, max_nan_ratio=0.05)
    print(f"\nData validation: {'PASS' if is_valid else 'FAIL'}")
    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  - {issue}")