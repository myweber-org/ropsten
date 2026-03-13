
import numpy as np
import pandas as pd

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
    
    return filtered_df.copy()

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing statistical measures
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def normalize_column(df, column, method='minmax'):
    """
    Normalize a DataFrame column using specified method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to normalize
    method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
    pd.DataFrame: DataFrame with normalized column
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_copy = df.copy()
    
    if method == 'minmax':
        min_val = df_copy[column].min()
        max_val = df_copy[column].max()
        if max_val != min_val:
            df_copy[column] = (df_copy[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df_copy[column].mean()
        std_val = df_copy[column].std()
        if std_val > 0:
            df_copy[column] = (df_copy[column] - mean_val) / std_val
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    return df_copy

def example_usage():
    """
    Example usage of the data cleaning functions.
    """
    np.random.seed(42)
    
    data = {
        'id': range(100),
        'value': np.random.normal(100, 15, 100)
    }
    
    df = pd.DataFrame(data)
    
    print("Original DataFrame shape:", df.shape)
    print("Original statistics:", calculate_basic_stats(df, 'value'))
    
    cleaned_df = remove_outliers_iqr(df, 'value')
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Cleaned statistics:", calculate_basic_stats(cleaned_df, 'value'))
    
    normalized_df = normalize_column(cleaned_df, 'value', method='zscore')
    print("\nNormalized statistics:", calculate_basic_stats(normalized_df, 'value'))
    
    return cleaned_df, normalized_df

if __name__ == "__main__":
    cleaned, normalized = example_usage()
    print("\nData cleaning completed successfully.")
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, columns=None, threshold=1.5):
    """
    Remove outliers using Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of column names to process, None for all numeric columns
    threshold (float): IQR multiplier for outlier detection
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def normalize_minmax(df, columns=None):
    """
    Normalize data using Min-Max scaling.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of column names to normalize, None for all numeric columns
    
    Returns:
    pd.DataFrame: Dataframe with normalized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_normalized = df.copy()
    for col in columns:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:
                df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
    
    return df_normalized

def remove_outliers_zscore(df, columns=None, threshold=3):
    """
    Remove outliers using Z-score method.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of column names to process, None for all numeric columns
    threshold (float): Z-score threshold for outlier detection
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    for col in columns:
        if col in df.columns:
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            mask = z_scores < threshold
            valid_indices = df[col].dropna().index[mask]
            df_clean = df_clean[df_clean.index.isin(valid_indices) | df[col].isna()]
    
    return df_clean.reset_index(drop=True)

def clean_dataset(df, outlier_method='iqr', normalize=True, outlier_threshold=1.5):
    """
    Comprehensive data cleaning pipeline.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    outlier_method (str): 'iqr' or 'zscore' for outlier removal
    normalize (bool): Whether to apply min-max normalization
    outlier_threshold (float): Threshold for outlier detection
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if outlier_method == 'iqr':
        df_clean = remove_outliers_iqr(df, numeric_cols, outlier_threshold)
    elif outlier_method == 'zscore':
        df_clean = remove_outliers_zscore(df, numeric_cols, outlier_threshold)
    else:
        df_clean = df.copy()
    
    if normalize and len(numeric_cols) > 0:
        df_clean = normalize_minmax(df_clean, numeric_cols)
    
    return df_clean
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
    
    return filtered_df

def clean_dataset(df, numeric_columns=None):
    """
    Clean dataset by removing outliers from all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of numeric column names. If None, uses all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in df.columns:
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, column)
            except Exception as e:
                print(f"Warning: Could not process column '{column}': {e}")
    
    return cleaned_df

def get_outlier_statistics(df, column):
    """
    Get statistics about outliers in a column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing outlier statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    stats = {
        'total_rows': len(df),
        'outlier_count': len(outliers),
        'outlier_percentage': (len(outliers) / len(df)) * 100,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'min_value': df[column].min(),
        'max_value': df[column].max(),
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR
    }
    
    return statsimport numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(df, columns):
    """
    Remove outliers using IQR method.
    """
    df_clean = df.copy()
    for col in columns:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def remove_outliers_zscore(df, columns, threshold=3):
    """
    Remove outliers using Z-score method.
    """
    df_clean = df.copy()
    for col in columns:
        if col in df_clean.columns:
            z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
            df_clean = df_clean[(z_scores < threshold) | (df_clean[col].isna())]
    return df_clean

def normalize_minmax(df, columns):
    """
    Normalize data using Min-Max scaling.
    """
    df_norm = df.copy()
    for col in columns:
        if col in df_norm.columns:
            min_val = df_norm[col].min()
            max_val = df_norm[col].max()
            if max_val > min_val:
                df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
    return df_norm

def normalize_zscore(df, columns):
    """
    Normalize data using Z-score standardization.
    """
    df_norm = df.copy()
    for col in columns:
        if col in df_norm.columns:
            mean_val = df_norm[col].mean()
            std_val = df_norm[col].std()
            if std_val > 0:
                df_norm[col] = (df_norm[col] - mean_val) / std_val
    return df_norm

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    """
    Main function to clean dataset by removing outliers and normalizing.
    """
    if outlier_method == 'iqr':
        df_clean = remove_outliers_iqr(df, numeric_columns)
    elif outlier_method == 'zscore':
        df_clean = remove_outliers_zscore(df, numeric_columns)
    else:
        df_clean = df.copy()

    if normalize_method == 'minmax':
        df_final = normalize_minmax(df_clean, numeric_columns)
    elif normalize_method == 'zscore':
        df_final = normalize_zscore(df_clean, numeric_columns)
    else:
        df_final = df_clean

    return df_final
import pandas as pd
import numpy as np
from pathlib import Path

def clean_csv_data(input_path, output_path=None):
    """
    Load a CSV file, clean missing values, and save cleaned data.
    """
    try:
        df = pd.read_csv(input_path)
        print(f"Original shape: {df.shape}")
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        print(f"After removing duplicates: {df.shape}")
        
        # Fill missing numeric values with column median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        # Fill missing categorical values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        # Remove columns with too many missing values (threshold: 50%)
        missing_threshold = 0.5
        cols_to_drop = [col for col in df.columns if df[col].isnull().mean() > missing_threshold]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            print(f"Dropped columns with >{missing_threshold*100}% missing values: {cols_to_drop}")
        
        # Generate output path if not provided
        if output_path is None:
            input_file = Path(input_path)
            output_path = input_file.parent / f"{input_file.stem}_cleaned{input_file.suffix}"
        
        # Save cleaned data
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
        print(f"Final shape: {df.shape}")
        
        return df, output_path
        
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        return None, None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None, None

def validate_dataframe(df):
    """
    Perform basic validation on cleaned dataframe.
    """
    if df is None:
        return False
    
    validation_results = {
        'has_data': not df.empty,
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    print("Data Validation Results:")
    for key, value in validation_results.items():
        print(f"  {key}: {value}")
    
    return validation_results['missing_values'] == 0 and validation_results['duplicate_rows'] == 0

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 3, 4, 5, 6],
        'name': ['Alice', 'Bob', 'Charlie', None, 'Eve', 'Alice'],
        'age': [25, 30, None, 35, 40, 25],
        'score': [85.5, 92.0, 78.5, None, 88.0, 85.5],
        'department': ['HR', 'IT', 'IT', None, 'Finance', 'HR']
    }
    
    # Create temporary CSV for demonstration
    temp_df = pd.DataFrame(sample_data)
    temp_input = "temp_sample_data.csv"
    temp_df.to_csv(temp_input, index=False)
    
    print("Starting data cleaning process...")
    cleaned_df, output_file = clean_csv_data(temp_input)
    
    if cleaned_df is not None:
        is_valid = validate_dataframe(cleaned_df)
        print(f"Data validation passed: {is_valid}")
    
    # Clean up temporary file
    import os
    if os.path.exists(temp_input):
        os.remove(temp_input)
        print(f"Removed temporary file: {temp_input}")