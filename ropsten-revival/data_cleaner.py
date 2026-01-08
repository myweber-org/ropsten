
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to drop duplicate rows
        fill_missing (str): Strategy for filling missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if cleaned_df.isnull().sum().sum() > 0:
        if fill_missing == 'drop':
            cleaned_df = cleaned_df.dropna()
            print("Dropped rows with missing values")
        elif fill_missing in ['mean', 'median']:
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if fill_missing == 'mean':
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
                else:
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
            print(f"Filled missing numeric values with {fill_missing}")
        elif fill_missing == 'mode':
            for col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
            print("Filled missing values with mode")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
        min_rows (int): Minimum number of rows required
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

def remove_outliers(df, column, method='iqr', threshold=1.5):
    """
    Remove outliers from a specific column using specified method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to process
        method (str): Outlier detection method ('iqr' or 'zscore')
        threshold (float): Threshold for outlier detection
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' must be numeric")
    
    data = df[column].copy()
    
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        mask = (data >= lower_bound) & (data <= upper_bound)
    elif method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        mask = z_scores <= threshold
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    initial_count = len(df)
    filtered_df = df[mask].reset_index(drop=True)
    removed_count = initial_count - len(filtered_df)
    
    print(f"Removed {removed_count} outliers from column '{column}' using {method} method")
    return filtered_df
import pandas as pd
import numpy as np

def detect_outliers_iqr(data, column):
    """
    Detect outliers in a specified column using the Interquartile Range method.
    
    Parameters:
    data (pd.DataFrame): The input dataframe
    column (str): The column name to analyze
    
    Returns:
    pd.DataFrame: DataFrame containing outliers
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    
    return outliers

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Parameters:
    data (pd.DataFrame): The input dataframe
    column (str): The column name to clean
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    cleaned_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    return cleaned_data

def calculate_statistics(data, column):
    """
    Calculate basic statistics for a column including outlier information.
    
    Parameters:
    data (pd.DataFrame): The input dataframe
    column (str): The column name to analyze
    
    Returns:
    dict: Dictionary containing statistical measures
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    stats = {
        'mean': data[column].mean(),
        'median': data[column].median(),
        'std': data[column].std(),
        'min': data[column].min(),
        'max': data[column].max(),
        'q1': data[column].quantile(0.25),
        'q3': data[column].quantile(0.75),
        'count': len(data),
        'missing': data[column].isnull().sum()
    }
    
    outliers = detect_outliers_iqr(data, column)
    stats['outlier_count'] = len(outliers)
    stats['outlier_percentage'] = (len(outliers) / len(data)) * 100
    
    return stats

def clean_numeric_data(data, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    If no columns specified, clean all numeric columns.
    
    Parameters:
    data (pd.DataFrame): The input dataframe
    columns (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    if columns is None:
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    cleaned_data = data.copy()
    
    for column in columns:
        if column in cleaned_data.columns:
            if pd.api.types.is_numeric_dtype(cleaned_data[column]):
                original_len = len(cleaned_data)
                cleaned_data = remove_outliers_iqr(cleaned_data, column)
                removed_count = original_len - len(cleaned_data)
                print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_data