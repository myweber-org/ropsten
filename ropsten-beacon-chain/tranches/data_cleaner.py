
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers from a DataFrame column using IQR method.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    threshold (float): IQR multiplier (default 1.5)
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    return filtered_df.copy()

def normalize_minmax(dataframe, columns=None):
    """
    Normalize specified columns using min-max scaling.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    columns (list): List of column names to normalize (default: all numeric columns)
    
    Returns:
    pd.DataFrame: DataFrame with normalized columns
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    result_df = dataframe.copy()
    
    for col in columns:
        if col not in result_df.columns:
            continue
            
        if result_df[col].dtype in [np.float64, np.int64]:
            col_min = result_df[col].min()
            col_max = result_df[col].max()
            
            if col_max != col_min:
                result_df[col] = (result_df[col] - col_min) / (col_max - col_min)
            else:
                result_df[col] = 0
    
    return result_df

def handle_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame columns.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    strategy (str): Imputation strategy ('mean', 'median', 'mode', 'drop')
    columns (list): List of column names to process (default: all columns)
    
    Returns:
    pd.DataFrame: DataFrame with handled missing values
    """
    if columns is None:
        columns = dataframe.columns.tolist()
    
    result_df = dataframe.copy()
    
    for col in columns:
        if col not in result_df.columns:
            continue
            
        if result_df[col].isnull().any():
            if strategy == 'drop':
                result_df = result_df.dropna(subset=[col])
            elif strategy == 'mean' and result_df[col].dtype in [np.float64, np.int64]:
                result_df[col] = result_df[col].fillna(result_df[col].mean())
            elif strategy == 'median' and result_df[col].dtype in [np.float64, np.int64]:
                result_df[col] = result_df[col].fillna(result_df[col].median())
            elif strategy == 'mode':
                mode_value = result_df[col].mode()
                if not mode_value.empty:
                    result_df[col] = result_df[col].fillna(mode_value.iloc[0])
    
    return result_df

def calculate_statistics(dataframe, columns=None):
    """
    Calculate descriptive statistics for specified columns.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    columns (list): List of column names (default: all numeric columns)
    
    Returns:
    dict: Dictionary containing statistics for each column
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    statistics = {}
    
    for col in columns:
        if col in dataframe.columns and dataframe[col].dtype in [np.float64, np.int64]:
            col_data = dataframe[col].dropna()
            
            if len(col_data) > 0:
                stats_dict = {
                    'mean': col_data.mean(),
                    'median': col_data.median(),
                    'std': col_data.std(),
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'count': len(col_data),
                    'missing': dataframe[col].isnull().sum()
                }
                
                if len(col_data) >= 3:
                    try:
                        stats_dict['skewness'] = stats.skew(col_data)
                        stats_dict['kurtosis'] = stats.kurtosis(col_data)
                    except:
                        stats_dict['skewness'] = None
                        stats_dict['kurtosis'] = None
                
                statistics[col] = stats_dict
    
    return statistics

def clean_dataset(dataframe, numeric_columns=None, outlier_threshold=1.5, 
                  normalize=True, missing_strategy='mean'):
    """
    Complete data cleaning pipeline.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of numeric column names
    outlier_threshold (float): IQR threshold for outlier removal
    normalize (bool): Whether to normalize numeric columns
    missing_strategy (str): Strategy for handling missing values
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = dataframe.copy()
    
    # Handle missing values
    cleaned_df = handle_missing_values(cleaned_df, strategy=missing_strategy)
    
    # Remove outliers from numeric columns
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col, outlier_threshold)
    
    # Normalize numeric columns if requested
    if normalize:
        cleaned_df = normalize_minmax(cleaned_df, numeric_columns)
    
    return cleaned_df

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    sample_data = {
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'feature_c': np.random.randint(1, 100, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    # Add some missing values and outliers
    df = pd.DataFrame(sample_data)
    df.loc[10:15, 'feature_a'] = np.nan
    df.loc[95, 'feature_b'] = 500  # Outlier
    
    print("Original DataFrame shape:", df.shape)
    print("Missing values per column:")
    print(df.isnull().sum())
    
    # Clean the dataset
    cleaned = clean_dataset(df, normalize=True)
    
    print("\nCleaned DataFrame shape:", cleaned.shape)
    print("Missing values after cleaning:")
    print(cleaned.isnull().sum())
    
    # Calculate statistics
    stats_result = calculate_statistics(cleaned)
    print("\nStatistics for cleaned data:")
    for col, col_stats in stats_result.items():
        print(f"\n{col}:")
        for stat_name, stat_value in col_stats.items():
            print(f"  {stat_name}: {stat_value:.4f}" if isinstance(stat_value, (int, float)) else f"  {stat_name}: {stat_value}")
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
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
    dict: Dictionary containing summary statistics.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

if __name__ == "__main__":
    sample_data = {
        'values': [10, 12, 12, 13, 12, 11, 14, 13, 15, 102, 12, 14, 13, 12, 11, 10, 14, 13, 12, 11, 105]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original data:")
    print(df)
    print(f"\nOriginal shape: {df.shape}")
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    print("\nCleaned data:")
    print(cleaned_df)
    print(f"\nCleaned shape: {cleaned_df.shape}")
    
    stats = calculate_summary_statistics(cleaned_df, 'values')
    print("\nSummary statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")