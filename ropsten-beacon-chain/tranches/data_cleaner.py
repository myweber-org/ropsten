
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
    print(f"Validation results: {validation}")import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        subset (list, optional): Columns to consider for duplicates
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def fill_missing_values(df, strategy='mean', columns=None):
    """
    Fill missing values in DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        strategy (str): 'mean', 'median', 'mode', or 'constant'
        columns (list, optional): Specific columns to fill
    
    Returns:
        pd.DataFrame: DataFrame with filled values
    """
    df_filled = df.copy()
    
    if columns is None:
        columns = df.columns
    
    for col in columns:
        if df[col].isnull().any():
            if strategy == 'mean':
                df_filled[col] = df[col].fillna(df[col].mean())
            elif strategy == 'median':
                df_filled[col] = df[col].fillna(df[col].median())
            elif strategy == 'mode':
                df_filled[col] = df[col].fillna(df[col].mode()[0])
            elif strategy == 'constant':
                df_filled[col] = df[col].fillna(0)
    
    return df_filled

def normalize_columns(df, columns=None):
    """
    Normalize specified columns to range [0, 1].
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): Columns to normalize
    
    Returns:
        pd.DataFrame: DataFrame with normalized columns
    """
    df_normalized = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df.columns and df[col].dtype in [np.float64, np.int64]:
            col_min = df[col].min()
            col_max = df[col].max()
            if col_max != col_min:
                df_normalized[col] = (df[col] - col_min) / (col_max - col_min)
    
    return df_normalized

def clean_dataframe(df, remove_dups=True, fill_na=True, normalize=True):
    """
    Apply multiple cleaning operations to DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        remove_dups (bool): Whether to remove duplicates
        fill_na (bool): Whether to fill missing values
        normalize (bool): Whether to normalize numeric columns
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if remove_dups:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if fill_na:
        cleaned_df = fill_missing_values(cleaned_df, strategy='mean')
    
    if normalize:
        cleaned_df = normalize_columns(cleaned_df)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list, optional): Required column names
    
    Returns:
        dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'missing_columns': [],
        'empty_columns': [],
        'null_percentages': {}
    }
    
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            validation_results['missing_columns'] = missing
            validation_results['is_valid'] = False
    
    for col in df.columns:
        null_percent = df[col].isnull().sum() / len(df) * 100
        validation_results['null_percentages'][col] = null_percent
        
        if df[col].isnull().all():
            validation_results['empty_columns'].append(col)
            validation_results['is_valid'] = False
    
    return validation_results