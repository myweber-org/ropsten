
import pandas as pd
import numpy as np
from scipy import stats

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using Interquartile Range method
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def remove_outliers(data, column, threshold=1.5):
    """
    Remove outliers from specified column
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_column(data, column, method='minmax'):
    """
    Normalize column using specified method
    """
    if method == 'minmax':
        min_val = data[column].min()
        max_val = data[column].max()
        data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = data[column].mean()
        std_val = data[column].std()
        data[column + '_normalized'] = (data[column] - mean_val) / std_val
    
    return data

def clean_dataset(data, numeric_columns, outlier_threshold=1.5, normalize=True):
    """
    Comprehensive data cleaning pipeline
    """
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column in cleaned_data.columns:
            cleaned_data = remove_outliers(cleaned_data, column, outlier_threshold)
            
            if normalize:
                cleaned_data = normalize_column(cleaned_data, column, method='zscore')
    
    return cleaned_data

def get_data_summary(data):
    """
    Generate statistical summary of dataset
    """
    summary = {
        'total_rows': len(data),
        'total_columns': len(data.columns),
        'missing_values': data.isnull().sum().sum(),
        'numeric_columns': list(data.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(data.select_dtypes(include=['object']).columns)
    }
    
    return summary

def validate_dataframe(data):
    """
    Validate dataframe structure and content
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if data.empty:
        raise ValueError("DataFrame is empty")
    
    return True
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to process
    factor (float): Multiplier for IQR (default 1.5)
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data.copy()

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to process
    threshold (float): Z-score threshold (default 3)
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    filtered_indices = np.where(z_scores < threshold)[0]
    
    filtered_data = data.iloc[filtered_indices].copy()
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to normalize
    
    Returns:
    pd.Series: Normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(data), index=data.index)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to normalize
    
    Returns:
    pd.Series: Standardized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(data, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    numeric_columns (list): List of numeric column names to process
    outlier_method (str): 'iqr' or 'zscore' (default 'iqr')
    normalize_method (str): 'minmax' or 'zscore' (default 'minmax')
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column not in cleaned_data.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_data = remove_outliers_iqr(cleaned_data, column)
        elif outlier_method == 'zscore':
            cleaned_data = remove_outliers_zscore(cleaned_data, column)
        else:
            raise ValueError(f"Unknown outlier method: {outlier_method}")
    
    for column in numeric_columns:
        if column not in cleaned_data.columns:
            continue
            
        if normalize_method == 'minmax':
            cleaned_data[f'{column}_normalized'] = normalize_minmax(cleaned_data, column)
        elif normalize_method == 'zscore':
            cleaned_data[f'{column}_standardized'] = normalize_zscore(cleaned_data, column)
        else:
            raise ValueError(f"Unknown normalize method: {normalize_method}")
    
    return cleaned_data.reset_index(drop=True)

def validate_data(data, required_columns, numeric_columns):
    """
    Validate dataset structure and content.
    
    Parameters:
    data (pd.DataFrame): Dataframe to validate
    required_columns (list): List of required column names
    numeric_columns (list): List of numeric column names
    
    Returns:
    dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'missing_columns': [],
        'non_numeric_columns': [],
        'null_counts': {},
        'basic_stats': {}
    }
    
    for column in required_columns:
        if column not in data.columns:
            validation_results['missing_columns'].append(column)
            validation_results['is_valid'] = False
    
    for column in numeric_columns:
        if column in data.columns:
            if not pd.api.types.is_numeric_dtype(data[column]):
                validation_results['non_numeric_columns'].append(column)
                validation_results['is_valid'] = False
            
            validation_results['null_counts'][column] = data[column].isnull().sum()
            validation_results['basic_stats'][column] = {
                'mean': data[column].mean(),
                'std': data[column].std(),
                'min': data[column].min(),
                'max': data[column].max()
            }
    
    return validation_results
import pandas as pd
import numpy as np

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
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(input_file, output_file):
    df = pd.read_csv(input_file)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        df = remove_outliers_iqr(df, col)
    
    for col in numeric_columns:
        df = normalize_minmax(df, col)
    
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")
    print(f"Original shape: {pd.read_csv(input_file).shape}")
    print(f"Cleaned shape: {df.shape}")

if __name__ == "__main__":
    clean_dataset('raw_data.csv', 'cleaned_data.csv')