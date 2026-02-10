
import pandas as pd
import hashlib

def remove_duplicates_by_hash(df, column_name):
    """
    Remove duplicate rows based on hash of specified column.
    """
    seen_hashes = set()
    indices_to_keep = []
    
    for idx, row in df.iterrows():
        content = str(row[column_name]).encode('utf-8')
        content_hash = hashlib.md5(content).hexdigest()
        
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            indices_to_keep.append(idx)
    
    return df.loc[indices_to_keep].reset_index(drop=True)

def clean_numeric_column(df, column_name, fill_method='mean'):
    """
    Clean numeric column by handling missing values.
    """
    if fill_method == 'mean':
        fill_value = df[column_name].mean()
    elif fill_method == 'median':
        fill_value = df[column_name].median()
    else:
        fill_value = 0
    
    df[column_name] = df[column_name].fillna(fill_value)
    return df

def standardize_text_column(df, column_name):
    """
    Standardize text column by converting to lowercase and stripping whitespace.
    """
    df[column_name] = df[column_name].astype(str).str.lower().str.strip()
    return df

def process_dataframe(input_file, output_file, primary_key_column):
    """
    Main function to process and clean the dataframe.
    """
    try:
        df = pd.read_csv(input_file)
        
        print(f"Original shape: {df.shape}")
        
        df = remove_duplicates_by_hash(df, primary_key_column)
        print(f"After deduplication: {df.shape}")
        
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_columns:
            df = clean_numeric_column(df, col, fill_method='median')
        
        text_columns = df.select_dtypes(include=['object']).columns
        for col in text_columns:
            df = standardize_text_column(df, col)
        
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to: {output_file}")
        
        return df
        
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        return None

if __name__ == "__main__":
    input_path = "raw_data.csv"
    output_path = "cleaned_data.csv"
    key_column = "id"
    
    cleaned_df = process_dataframe(input_path, output_path, key_column)
import numpy as np
import pandas as pd

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
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data to [0, 1] range using min-max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(data), index=data.index)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using z-score normalization.
    
    Args:
        data: pandas DataFrame
        column: column name to standardize
    
    Returns:
        Series with standardized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in specified columns.
    
    Args:
        data: pandas DataFrame
        strategy: 'mean', 'median', 'mode', or 'drop'
        columns: list of columns to process (None for all numeric columns)
    
    Returns:
        DataFrame with missing values handled
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    result = data.copy()
    
    for col in columns:
        if col not in result.columns:
            continue
            
        if strategy == 'drop':
            result = result.dropna(subset=[col])
        elif strategy == 'mean':
            result[col] = result[col].fillna(result[col].mean())
        elif strategy == 'median':
            result[col] = result[col].fillna(result[col].median())
        elif strategy == 'mode':
            result[col] = result[col].fillna(result[col].mode()[0])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    return result

def create_sample_data():
    """
    Create sample data for testing.
    
    Returns:
        DataFrame with sample data containing outliers and missing values
    """
    np.random.seed(42)
    n_samples = 100
    
    data = pd.DataFrame({
        'feature_a': np.random.normal(50, 10, n_samples),
        'feature_b': np.random.exponential(5, n_samples),
        'feature_c': np.random.uniform(0, 100, n_samples)
    })
    
    # Add some outliers
    data.loc[10, 'feature_a'] = 150
    data.loc[20, 'feature_b'] = 50
    
    # Add some missing values
    data.loc[30:35, 'feature_c'] = np.nan
    
    return data

if __name__ == "__main__":
    # Example usage
    sample_data = create_sample_data()
    print("Original data shape:", sample_data.shape)
    print("\nOriginal data summary:")
    print(sample_data.describe())
    
    # Remove outliers
    cleaned_data = remove_outliers_iqr(sample_data, 'feature_a')
    print("\nAfter outlier removal shape:", cleaned_data.shape)
    
    # Normalize a column
    normalized = normalize_minmax(cleaned_data, 'feature_b')
    print("\nNormalized feature_b (first 5 values):")
    print(normalized.head())
    
    # Handle missing values
    filled_data = handle_missing_values(sample_data, strategy='mean')
    print("\nMissing values handled. Any remaining NaN?")
    print(filled_data.isna().sum())