import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    Returns a filtered DataFrame.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_column_minmax(data, column):
    """
    Normalize a column using min-max scaling to range [0, 1].
    Returns a new DataFrame with the normalized column.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        normalized = 0.5
    else:
        normalized = (data[column] - min_val) / (max_val - min_val)
    
    result = data.copy()
    result[f'{column}_normalized'] = normalized
    return result

def calculate_basic_stats(data, column):
    """
    Calculate basic statistics for a column.
    Returns a dictionary with mean, median, std, min, and max.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': data[column].mean(),
        'median': data[column].median(),
        'std': data[column].std(),
        'min': data[column].min(),
        'max': data[column].max(),
        'count': data[column].count()
    }
    return stats

def clean_missing_values(data, strategy='mean'):
    """
    Handle missing values in numeric columns.
    Strategy can be 'mean', 'median', or 'drop'.
    Returns a cleaned DataFrame.
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        cleaned_data = data.dropna(subset=numeric_cols)
    elif strategy == 'mean':
        cleaned_data = data.copy()
        for col in numeric_cols:
            cleaned_data[col].fillna(data[col].mean(), inplace=True)
    elif strategy == 'median':
        cleaned_data = data.copy()
        for col in numeric_cols:
            cleaned_data[col].fillna(data[col].median(), inplace=True)
    else:
        raise ValueError("Strategy must be 'mean', 'median', or 'drop'")
    
    return cleaned_data

def validate_dataframe(data):
    """
    Basic validation of DataFrame structure.
    Returns True if valid, raises exceptions otherwise.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if data.empty:
        raise ValueError("DataFrame is empty")
    
    if data.isnull().all().any():
        raise ValueError("Some columns contain only null values")
    
    return Trueimport numpy as np
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
    return df[column].apply(lambda x: (x - min_val) / (max_val - min_val))

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    return cleaned_df.reset_index(drop=True)

def validate_dataframe(df):
    required_checks = [
        (lambda: not df.empty, "DataFrame is empty"),
        (lambda: df.isnull().sum().sum() == 0, "DataFrame contains null values"),
        (lambda: all(df.dtypes != object), "DataFrame contains non-numeric columns")
    ]
    for check, message in required_checks:
        if not check():
            raise ValueError(f"Validation failed: {message}")
    return True

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 200),
        'feature_b': np.random.exponential(scale=2.0, size=200),
        'feature_c': np.random.uniform(0, 1, 200)
    })
    print("Original shape:", sample_data.shape)
    cleaned = clean_dataset(sample_data, ['feature_a', 'feature_b', 'feature_c'])
    print("Cleaned shape:", cleaned.shape)
    try:
        validate_dataframe(cleaned)
        print("Data validation passed")
    except ValueError as e:
        print(f"Data validation error: {e}")