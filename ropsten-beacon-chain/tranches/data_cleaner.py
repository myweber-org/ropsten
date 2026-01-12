
import numpy as np
import pandas as pd
from scipy import stats

def normalize_data(data, method='zscore'):
    """
    Normalize data using specified method.
    
    Args:
        data: numpy array or pandas Series
        method: 'zscore', 'minmax', or 'robust'
    
    Returns:
        Normalized data
    """
    if method == 'zscore':
        return (data - np.mean(data)) / np.std(data)
    elif method == 'minmax':
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    elif method == 'robust':
        median = np.median(data)
        iqr = stats.iqr(data)
        return (data - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def remove_outliers_iqr(data, factor=1.5):
    """
    Remove outliers using IQR method.
    
    Args:
        data: numpy array
        factor: multiplier for IQR (default 1.5)
    
    Returns:
        Data with outliers removed
    """
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    return data[(data >= lower_bound) & (data <= upper_bound)]

def clean_dataset(df, columns=None, outlier_method='iqr', normalize_method='zscore'):
    """
    Clean dataset by removing outliers and normalizing specified columns.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to clean (default: all numeric columns)
        outlier_method: 'iqr' or None
        normalize_method: normalization method or None
    
    Returns:
        Cleaned DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            # Remove outliers
            if outlier_method == 'iqr':
                mask = ~df[col].isna()
                clean_series = remove_outliers_iqr(df.loc[mask, col].values)
                cleaned_df.loc[mask, col] = pd.Series(clean_series, index=df.loc[mask, col].index)
            
            # Normalize
            if normalize_method:
                mask = ~cleaned_df[col].isna()
                cleaned_df.loc[mask, col] = normalize_data(cleaned_df.loc[mask, col], method=normalize_method)
    
    return cleaned_df

def validate_data(df, check_missing=True, check_infinite=True):
    """
    Validate data quality.
    
    Args:
        df: pandas DataFrame
        check_missing: flag to check for missing values
        check_infinite: flag to check for infinite values
    
    Returns:
        Dictionary with validation results
    """
    results = {}
    
    if check_missing:
        missing = df.isnull().sum()
        results['missing_values'] = missing[missing > 0].to_dict()
    
    if check_infinite:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        infinite_counts = {}
        for col in numeric_cols:
            infinite = np.isinf(df[col]).sum()
            if infinite > 0:
                infinite_counts[col] = infinite
        results['infinite_values'] = infinite_counts
    
    return results

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    data = {
        'feature1': np.random.normal(100, 15, 100),
        'feature2': np.random.exponential(50, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    # Add some outliers and missing values
    data['feature1'][10] = 500  # Outlier
    data['feature2'][20] = np.nan  # Missing value
    
    df = pd.DataFrame(data)
    
    # Clean the data
    cleaned_df = clean_dataset(df, columns=['feature1', 'feature2'])
    
    # Validate
    validation = validate_data(cleaned_df)
    
    print(f"Original shape: {df.shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    print(f"Validation results: {validation}")