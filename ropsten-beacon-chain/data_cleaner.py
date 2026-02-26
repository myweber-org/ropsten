
import pandas as pd
import numpy as np
from scipy import stats

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Remove outliers using z-score
    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    df = df[(z_scores < 3).all(axis=1)]
    
    # Normalize numeric columns
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].min()) / (df[numeric_cols].max() - df[numeric_cols].min())
    
    return df

def save_cleaned_data(df, output_path):
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = load_and_clean_data(input_file)
    save_cleaned_data(cleaned_df, output_file)
    print(f"Data cleaning complete. Cleaned data saved to {output_file}")
import numpy as np
import pandas as pd
from scipy import stats

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using Interquartile Range method.
    Returns boolean mask for outliers.
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    return (data[column] < lower_bound) | (data[column] > upper_bound)

def remove_outliers(data, columns, threshold=1.5):
    """
    Remove outliers from specified columns using IQR method.
    """
    clean_data = data.copy()
    for col in columns:
        outlier_mask = detect_outliers_iqr(clean_data, col, threshold)
        clean_data = clean_data[~outlier_mask]
    return clean_data.reset_index(drop=True)

def normalize_minmax(data, columns):
    """
    Apply min-max normalization to specified columns.
    """
    normalized_data = data.copy()
    for col in columns:
        min_val = normalized_data[col].min()
        max_val = normalized_data[col].max()
        normalized_data[col] = (normalized_data[col] - min_val) / (max_val - min_val)
    return normalized_data

def standardize_zscore(data, columns):
    """
    Apply z-score standardization to specified columns.
    """
    standardized_data = data.copy()
    for col in columns:
        mean_val = standardized_data[col].mean()
        std_val = standardized_data[col].std()
        standardized_data[col] = (standardized_data[col] - mean_val) / std_val
    return standardized_data

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values using specified strategy.
    """
    if columns is None:
        columns = data.columns
    
    cleaned_data = data.copy()
    
    for col in columns:
        if cleaned_data[col].isnull().any():
            if strategy == 'mean':
                fill_value = cleaned_data[col].mean()
            elif strategy == 'median':
                fill_value = cleaned_data[col].median()
            elif strategy == 'mode':
                fill_value = cleaned_data[col].mode()[0]
            elif strategy == 'drop':
                cleaned_data = cleaned_data.dropna(subset=[col])
                continue
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            cleaned_data[col] = cleaned_data[col].fillna(fill_value)
    
    return cleaned_data

def clean_dataset(data, config):
    """
    Comprehensive data cleaning pipeline.
    """
    cleaned_data = data.copy()
    
    if 'missing_values' in config:
        cleaned_data = handle_missing_values(
            cleaned_data,
            strategy=config['missing_values'].get('strategy', 'mean'),
            columns=config['missing_values'].get('columns')
        )
    
    if 'outliers' in config:
        cleaned_data = remove_outliers(
            cleaned_data,
            columns=config['outliers'].get('columns', cleaned_data.select_dtypes(include=[np.number]).columns),
            threshold=config['outliers'].get('threshold', 1.5)
        )
    
    if 'normalization' in config:
        norm_type = config['normalization'].get('type', 'minmax')
        columns = config['normalization'].get('columns', cleaned_data.select_dtypes(include=[np.number]).columns)
        
        if norm_type == 'minmax':
            cleaned_data = normalize_minmax(cleaned_data, columns)
        elif norm_type == 'zscore':
            cleaned_data = standardize_zscore(cleaned_data, columns)
    
    return cleaned_data

def get_data_summary(data):
    """
    Generate comprehensive data summary statistics.
    """
    summary = pd.DataFrame({
        'dtype': data.dtypes,
        'non_null_count': data.count(),
        'null_count': data.isnull().sum(),
        'null_percentage': (data.isnull().sum() / len(data)) * 100,
        'unique_values': data.nunique()
    })
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        numeric_stats = data[numeric_cols].agg(['mean', 'median', 'std', 'min', 'max'])
        summary = summary.join(numeric_stats.T, how='left')
    
    return summary
import numpy as np
import pandas as pd
from scipy import stats

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using Interquartile Range method.
    Returns indices of outliers.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers.index.tolist()

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    Returns cleaned dataframe.
    """
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize column values to range [0, 1] using Min-Max scaling.
    Returns normalized series.
    """
    min_val = data[column].min()
    max_val = data[column].max()
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_data(data, column):
    """
    Standardize column values to have mean=0 and std=1.
    Returns standardized series.
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    standardized = (data[column] - mean_val) / std_val
    return standardized

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values using specified strategy.
    Supported strategies: 'mean', 'median', 'mode', 'drop'
    """
    if columns is None:
        columns = data.columns
    
    data_clean = data.copy()
    
    for col in columns:
        if data[col].isnull().any():
            if strategy == 'mean':
                fill_value = data[col].mean()
            elif strategy == 'median':
                fill_value = data[col].median()
            elif strategy == 'mode':
                fill_value = data[col].mode()[0]
            elif strategy == 'drop':
                data_clean = data_clean.dropna(subset=[col])
                continue
            else:
                raise ValueError(f"Unsupported strategy: {strategy}")
            
            data_clean[col] = data[col].fillna(fill_value)
    
    return data_clean

def clean_dataset(data, outlier_method='iqr', normalize=True, handle_missing=True):
    """
    Comprehensive data cleaning pipeline.
    """
    cleaned_data = data.copy()
    
    if handle_missing:
        cleaned_data = handle_missing_values(cleaned_data)
    
    numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if outlier_method == 'iqr':
            outliers = detect_outliers_iqr(cleaned_data, col)
            if outliers:
                cleaned_data = cleaned_data.drop(outliers)
        elif outlier_method == 'zscore':
            cleaned_data = remove_outliers_zscore(cleaned_data, col)
    
    if normalize:
        for col in numeric_cols:
            if col in cleaned_data.columns:
                cleaned_data[col] = normalize_minmax(cleaned_data, col)
    
    return cleaned_data.reset_index(drop=True)