
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a specified column in a DataFrame using the IQR method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to clean.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed from the specified column.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df.reset_index(drop=True)

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to analyze.
    
    Returns:
    dict: Dictionary containing count, mean, std, min, max, and IQR.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'count': df[column].count(),
        'mean': df[column].mean(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'q1': df[column].quantile(0.25),
        'q3': df[column].quantile(0.75),
        'iqr': df[column].quantile(0.75) - df[column].quantile(0.25)
    }
    
    return stats

def example_usage():
    """
    Example usage of the data cleaning functions.
    """
    np.random.seed(42)
    data = {
        'id': range(100),
        'value': np.concatenate([
            np.random.normal(100, 10, 90),
            np.random.normal(300, 50, 10)
        ])
    }
    
    df = pd.DataFrame(data)
    print(f"Original data shape: {df.shape}")
    print(f"Original statistics:")
    original_stats = calculate_summary_statistics(df, 'value')
    for key, val in original_stats.items():
        print(f"  {key}: {val:.2f}")
    
    cleaned_df = remove_outliers_iqr(df, 'value')
    print(f"\nCleaned data shape: {cleaned_df.shape}")
    print(f"Cleaned statistics:")
    cleaned_stats = calculate_summary_statistics(cleaned_df, 'value')
    for key, val in cleaned_stats.items():
        print(f"  {key}: {val:.2f}")
    
    return cleaned_df

if __name__ == "__main__":
    cleaned_data = example_usage()
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing=True, strategy='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to remove duplicate rows.
    fill_missing (bool): Whether to fill missing values.
    strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', or 'constant').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        for column in cleaned_df.select_dtypes(include=[np.number]).columns:
            if cleaned_df[column].isnull().any():
                if strategy == 'mean':
                    cleaned_df[column].fillna(cleaned_df[column].mean(), inplace=True)
                elif strategy == 'median':
                    cleaned_df[column].fillna(cleaned_df[column].median(), inplace=True)
                elif strategy == 'mode':
                    cleaned_df[column].fillna(cleaned_df[column].mode()[0], inplace=True)
                elif strategy == 'constant':
                    cleaned_df[column].fillna(0, inplace=True)
    
    return cleaned_df

def remove_outliers(df, columns=None, method='iqr', threshold=1.5):
    """
    Remove outliers from specified columns using IQR or Z-score method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns (list): List of column names to process. If None, process all numeric columns.
    method (str): Method for outlier detection ('iqr' or 'zscore').
    threshold (float): Threshold for outlier detection.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    filtered_df = df.copy()
    
    for column in columns:
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            if method == 'iqr':
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                mask = (filtered_df[column] >= lower_bound) & (filtered_df[column] <= upper_bound)
                filtered_df = filtered_df[mask]
            elif method == 'zscore':
                z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
                filtered_df = filtered_df[z_scores <= threshold]
    
    return filtered_df

def normalize_data(df, columns=None, method='minmax'):
    """
    Normalize specified columns in the DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns (list): List of column names to normalize. If None, normalize all numeric columns.
    method (str): Normalization method ('minmax' or 'standard').
    
    Returns:
    pd.DataFrame: DataFrame with normalized columns.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    normalized_df = df.copy()
    
    for column in columns:
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            if method == 'minmax':
                min_val = df[column].min()
                max_val = df[column].max()
                if max_val != min_val:
                    normalized_df[column] = (df[column] - min_val) / (max_val - min_val)
            elif method == 'standard':
                mean_val = df[column].mean()
                std_val = df[column].std()
                if std_val != 0:
                    normalized_df[column] = (df[column] - mean_val) / std_val
    
    return normalized_df

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, 5, None, 7, 8, 9, 100],
        'B': [10, 20, 20, 40, 50, 60, 70, 80, 90, 1000],
        'C': ['a', 'b', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned = clean_dataset(df, strategy='median')
    print(cleaned)
    print("\nDataFrame without outliers:")
    no_outliers = remove_outliers(cleaned, threshold=2.0)
    print(no_outliers)
    print("\nNormalized DataFrame:")
    normalized = normalize_data(no_outliers, method='minmax')
    print(normalized)
import pandas as pd
import numpy as np

def clean_dataset(df, missing_strategy='mean', outlier_method='iqr', columns=None):
    """
    Clean a pandas DataFrame by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
    outlier_method (str): Method for detecting outliers ('iqr', 'zscore')
    columns (list): Specific columns to clean, if None clean all numeric columns
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    
    df_clean = df.copy()
    
    if columns is None:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        if df_clean[col].dtype in [np.float64, np.int64]:
            handle_missing_values(df_clean, col, missing_strategy)
            handle_outliers(df_clean, col, outlier_method)
    
    return df_clean

def handle_missing_values(df, column, strategy='mean'):
    """Handle missing values in a specific column"""
    
    if strategy == 'mean':
        fill_value = df[column].mean()
    elif strategy == 'median':
        fill_value = df[column].median()
    elif strategy == 'mode':
        fill_value = df[column].mode()[0] if not df[column].mode().empty else 0
    elif strategy == 'drop':
        df.dropna(subset=[column], inplace=True)
        return
    else:
        fill_value = 0
    
    df[column].fillna(fill_value, inplace=True)

def handle_outliers(df, column, method='iqr'):
    """Detect and cap outliers in a specific column"""
    
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
        df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    
    elif method == 'zscore':
        mean = df[column].mean()
        std = df[column].std()
        
        if std > 0:
            z_scores = np.abs((df[column] - mean) / std)
            df[column] = np.where(z_scores > 3, mean, df[column])

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (is_valid, error_message)
    """
    
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(df) < min_rows:
        return False, f"DataFrame must have at least {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

def get_data_summary(df):
    """Generate summary statistics for DataFrame"""
    
    summary = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.astype(str).to_dict(),
        'numeric_stats': {}
    }
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        summary['numeric_stats'][col] = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'median': df[col].median()
        }
    
    return summary

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 100],
        'B': [5, 6, 7, np.nan, 9],
        'C': ['x', 'y', 'z', 'x', 'y']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nData Summary:")
    print(get_data_summary(df))
    
    cleaned_df = clean_dataset(df, missing_strategy='median', outlier_method='iqr')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df.reset_index(drop=True)

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    If no columns specified, clean all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{col}'")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    bool: True if validation passes
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True

def main():
    """
    Example usage of data cleaning functions.
    """
    np.random.seed(42)
    
    sample_data = {
        'id': range(100),
        'value': np.random.normal(100, 15, 100),
        'temperature': np.random.normal(25, 5, 100)
    }
    
    sample_data['value'][10] = 500
    sample_data['value'][20] = -200
    sample_data['temperature'][30] = 100
    
    df = pd.DataFrame(sample_data)
    
    print("Original DataFrame shape:", df.shape)
    print("\nDataFrame info:")
    print(df.info())
    
    try:
        validate_dataframe(df, required_columns=['id', 'value', 'temperature'])
        
        cleaned_df = clean_numeric_data(df, columns=['value', 'temperature'])
        
        print("\nCleaned DataFrame shape:", cleaned_df.shape)
        print("\nSummary statistics:")
        print(cleaned_df[['value', 'temperature']].describe())
        
    except Exception as e:
        print(f"Error during data cleaning: {e}")

if __name__ == "__main__":
    main()