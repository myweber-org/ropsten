import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Args:
        data: pandas DataFrame
        column: Column name to process
        multiplier: IQR multiplier for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data.copy()

def normalize_minmax(data, column):
    """
    Normalize a column using min-max scaling to range [0, 1].
    
    Args:
        data: pandas DataFrame
        column: Column name to normalize
    
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
    Standardize a column using z-score normalization.
    
    Args:
        data: pandas DataFrame
        column: Column name to standardize
    
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
        strategy: Imputation strategy ('mean', 'median', 'mode', 'drop')
        columns: List of columns to process (None processes all numeric columns)
    
    Returns:
        DataFrame with missing values handled
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
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
            result[col] = result[col].fillna(result[col].mode()[0] if not result[col].mode().empty else 0)
    
    return result

def create_cleaned_dataset(data, config):
    """
    Apply multiple cleaning operations based on configuration.
    
    Args:
        data: Input DataFrame
        config: Dictionary with cleaning configuration
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_data = data.copy()
    
    if 'remove_outliers' in config:
        for col, params in config['remove_outliers'].items():
            multiplier = params.get('multiplier', 1.5)
            cleaned_data = remove_outliers_iqr(cleaned_data, col, multiplier)
    
    if 'normalize' in config:
        for col in config['normalize']:
            if col in cleaned_data.columns:
                cleaned_data[f'{col}_normalized'] = normalize_minmax(cleaned_data, col)
    
    if 'standardize' in config:
        for col in config['standardize']:
            if col in cleaned_data.columns:
                cleaned_data[f'{col}_standardized'] = standardize_zscore(cleaned_data, col)
    
    if 'handle_missing' in config:
        strategy = config['handle_missing'].get('strategy', 'mean')
        columns = config['handle_missing'].get('columns')
        cleaned_data = handle_missing_values(cleaned_data, strategy, columns)
    
    return cleaned_data
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers from a column using the Interquartile Range method.
    
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

def normalize_minmax(data, column):
    """
    Normalize column values to range [0, 1] using min-max scaling.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to normalize
    
    Returns:
    pd.Series: Normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    col_data = data[column].astype(float)
    min_val = col_data.min()
    max_val = col_data.max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(col_data), index=col_data.index)
    
    normalized = (col_data - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize column values using z-score normalization.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to standardize
    
    Returns:
    pd.Series: Standardized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    col_data = data[column].astype(float)
    mean_val = col_data.mean()
    std_val = col_data.std()
    
    if std_val == 0:
        return pd.Series([0] * len(col_data), index=col_data.index)
    
    standardized = (col_data - mean_val) / std_val
    return standardized

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in specified columns.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    strategy (str): Imputation strategy ('mean', 'median', 'mode', 'drop')
    columns (list): List of columns to process, None for all numeric columns
    
    Returns:
    pd.DataFrame: Dataframe with missing values handled
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
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

def validate_dataframe(data, required_columns=None, numeric_columns=None):
    """
    Validate dataframe structure and content.
    
    Parameters:
    data (pd.DataFrame): Dataframe to validate
    required_columns (list): Columns that must be present
    numeric_columns (list): Columns that must be numeric
    
    Returns:
    dict: Validation results with status and messages
    """
    validation_result = {
        'is_valid': True,
        'messages': []
    }
    
    if not isinstance(data, pd.DataFrame):
        validation_result['is_valid'] = False
        validation_result['messages'].append('Input is not a pandas DataFrame')
        return validation_result
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            validation_result['is_valid'] = False
            validation_result['messages'].append(f'Missing required columns: {missing_cols}')
    
    if numeric_columns:
        non_numeric = []
        for col in numeric_columns:
            if col in data.columns:
                try:
                    pd.to_numeric(data[col])
                except ValueError:
                    non_numeric.append(col)
        
        if non_numeric:
            validation_result['is_valid'] = False
            validation_result['messages'].append(f'Non-numeric values in columns: {non_numeric}')
    
    if data.empty:
        validation_result['is_valid'] = False
        validation_result['messages'].append('Dataframe is empty')
    
    return validation_result
import pandas as pd
import re

def clean_dataframe(df, text_column):
    """
    Clean a DataFrame by removing duplicates and normalizing text in a specified column.
    """
    # Remove duplicate rows
    df_clean = df.drop_duplicates().reset_index(drop=True)
    
    # Normalize text: lowercase and remove extra whitespace
    if text_column in df_clean.columns:
        df_clean[text_column] = df_clean[text_column].astype(str).apply(
            lambda x: re.sub(r'\s+', ' ', x.strip().lower())
        )
    
    return df_clean

def save_cleaned_data(df, output_path):
    """
    Save the cleaned DataFrame to a CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 2, 3, 4],
        'comment': ['Hello World', '  Python Code  ', 'python code', 'TEST', '   multiple   spaces   ']
    }
    df = pd.DataFrame(sample_data)
    
    cleaned_df = clean_dataframe(df, 'comment')
    print("Cleaned DataFrame:")
    print(cleaned_df)
    
    save_cleaned_data(cleaned_df, 'cleaned_data.csv')