
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
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

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': len(df[column])
    }
    
    return stats

def clean_dataset(df, columns_to_clean):
    """
    Clean multiple columns in a DataFrame by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns_to_clean (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    dict: Dictionary of statistics for each cleaned column
    """
    cleaned_df = df.copy()
    statistics = {}
    
    for column in columns_to_clean:
        if column in cleaned_df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            
            stats = calculate_summary_statistics(cleaned_df, column)
            stats['outliers_removed'] = removed_count
            statistics[column] = stats
    
    return cleaned_df, statistics

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'temperature': [22, 23, 24, 25, 26, 27, 28, 29, 30, 100],
        'humidity': [45, 46, 47, 48, 49, 50, 51, 52, 53, 200],
        'pressure': [1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 500]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    columns_to_clean = ['temperature', 'humidity', 'pressure']
    cleaned_df, stats = clean_dataset(df, columns_to_clean)
    
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\n" + "="*50 + "\n")
    
    print("Summary Statistics:")
    for column, column_stats in stats.items():
        print(f"\n{column}:")
        for stat_name, stat_value in column_stats.items():
            print(f"  {stat_name}: {stat_value}")import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Args:
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
    
    return filtered_df

def clean_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in specified columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        strategy (str): Imputation strategy ('mean', 'median', 'mode', 'drop')
        columns (list): List of columns to process, None for all columns
    
    Returns:
        pd.DataFrame: DataFrame with missing values handled
    """
    if columns is None:
        columns = df.columns
    
    df_clean = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if strategy == 'drop':
            df_clean = df_clean.dropna(subset=[col])
        elif strategy == 'mean':
            df_clean[col].fillna(df_clean[col].mean(), inplace=True)
        elif strategy == 'median':
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
        elif strategy == 'mode':
            df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    return df_clean

def normalize_column(df, column, method='minmax'):
    """
    Normalize a column using specified method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to normalize
        method (str): Normalization method ('minmax', 'zscore')
    
    Returns:
        pd.DataFrame: DataFrame with normalized column
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_normalized = df.copy()
    
    if method == 'minmax':
        min_val = df_normalized[column].min()
        max_val = df_normalized[column].max()
        if max_val != min_val:
            df_normalized[column] = (df_normalized[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df_normalized[column].mean()
        std_val = df_normalized[column].std()
        if std_val > 0:
            df_normalized[column] = (df_normalized[column] - mean_val) / std_val
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return df_normalized

def get_data_summary(df):
    """
    Generate a summary statistics DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        pd.DataFrame: Summary statistics
    """
    summary = pd.DataFrame({
        'count': df.count(),
        'mean': df.mean(numeric_only=True),
        'std': df.std(numeric_only=True),
        'min': df.min(numeric_only=True),
        '25%': df.quantile(0.25, numeric_only=True),
        '50%': df.quantile(0.50, numeric_only=True),
        '75%': df.quantile(0.75, numeric_only=True),
        'max': df.max(numeric_only=True),
        'missing': df.isnull().sum(),
        'dtype': df.dtypes
    })
    
    return summary