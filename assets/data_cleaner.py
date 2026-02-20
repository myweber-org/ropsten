import pandas as pd
import numpy as np

def clean_csv_data(file_path, output_path=None, fill_strategy='mean'):
    """
    Clean a CSV file by handling missing values and removing duplicates.
    
    Args:
        file_path (str): Path to the input CSV file.
        output_path (str, optional): Path to save cleaned CSV. If None, returns DataFrame.
        fill_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', 'drop').
    
    Returns:
        pd.DataFrame or None: Cleaned DataFrame if output_path is None, else None.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None
    
    original_shape = df.shape
    print(f"Original data shape: {original_shape}")
    
    df = df.drop_duplicates()
    print(f"Removed {original_shape[0] - df.shape[0]} duplicate rows.")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if fill_strategy == 'drop':
        df = df.dropna()
        print(f"Removed rows with missing values. New shape: {df.shape}")
    elif fill_strategy in ['mean', 'median'] and numeric_cols:
        for col in numeric_cols:
            if df[col].isnull().any():
                if fill_strategy == 'mean':
                    fill_value = df[col].mean()
                else:
                    fill_value = df[col].median()
                df[col].fillna(fill_value, inplace=True)
                print(f"Filled missing values in '{col}' with {fill_strategy}: {fill_value:.2f}")
    elif fill_strategy == 'mode':
        for col in df.columns:
            if df[col].isnull().any():
                fill_value = df[col].mode()[0] if not df[col].mode().empty else 'UNKNOWN'
                df[col].fillna(fill_value, inplace=True)
                print(f"Filled missing values in '{col}' with mode: {fill_value}")
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
        return None
    else:
        return df

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for basic integrity checks.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list, optional): List of columns that must be present.
    
    Returns:
        dict: Dictionary with validation results.
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'summary': {}
    }
    
    if df is None or df.empty:
        validation_results['is_valid'] = False
        validation_results['errors'].append("DataFrame is None or empty.")
        return validation_results
    
    validation_results['summary']['shape'] = df.shape
    validation_results['summary']['columns'] = df.columns.tolist()
    validation_results['summary']['dtypes'] = df.dtypes.to_dict()
    
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        validation_results['warnings'].append(f"Found {missing_values} missing values in the dataset.")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing required columns: {missing_cols}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        validation_results['summary']['numeric_stats'] = df[numeric_cols].describe().to_dict()
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'value': [10.5, 20.3, np.nan, 40.1, 50.0, 60.7, np.nan, 80.2, 90.9, 100.0],
        'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
        'score': [85, 92, 78, np.nan, 88, 95, 76, 89, 91, 94]
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned_df = clean_csv_data('test_data.csv', fill_strategy='mean')
    
    if cleaned_df is not None:
        validation = validate_dataframe(cleaned_df, required_columns=['id', 'value', 'category'])
        print("\nValidation Results:")
        print(f"Is Valid: {validation['is_valid']}")
        print(f"Shape: {validation['summary']['shape']}")
        
        if validation['warnings']:
            print(f"Warnings: {validation['warnings']}")
        
        if validation['errors']:
            print(f"Errors: {validation['errors']}")
    
    import os
    if os.path.exists('test_data.csv'):
        os.remove('test_data.csv')import numpy as np
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
    
    return filtered_df

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a DataFrame column.
    
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
            df_copy[f'{column}_normalized'] = (df_copy[column] - min_val) / (max_val - min_val)
        else:
            df_copy[f'{column}_normalized'] = 0.5
    
    elif method == 'zscore':
        mean_val = df_copy[column].mean()
        std_val = df_copy[column].std()
        if std_val != 0:
            df_copy[f'{column}_normalized'] = (df_copy[column] - mean_val) / std_val
        else:
            df_copy[f'{column}_normalized'] = 0
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    return df_copy

def main():
    """
    Example usage of data cleaning utilities.
    """
    np.random.seed(42)
    
    data = {
        'values': np.random.normal(100, 15, 1000).tolist() + [500, -200]
    }
    
    df = pd.DataFrame(data)
    
    print("Original data shape:", df.shape)
    print("Original statistics:", calculate_summary_statistics(df, 'values'))
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    print("\nCleaned data shape:", cleaned_df.shape)
    print("Cleaned statistics:", calculate_summary_statistics(cleaned_df, 'values'))
    
    normalized_df = normalize_column(cleaned_df, 'values', method='zscore')
    print("\nNormalized column sample:")
    print(normalized_df[['values', 'values_normalized']].head())

if __name__ == "__main__":
    main()
import pandas as pd

def clean_dataset(df, sort_column=None):
    """
    Clean a pandas DataFrame by removing duplicate rows and optionally sorting.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        sort_column (str, optional): Column name to sort by. Defaults to None.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame with duplicates removed.
    """
    # Remove duplicate rows
    cleaned_df = df.drop_duplicates()
    
    # Sort by specified column if provided
    if sort_column and sort_column in cleaned_df.columns:
        cleaned_df = cleaned_df.sort_values(by=sort_column)
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def filter_numeric_columns(df):
    """
    Filter DataFrame to include only numeric columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame containing only numeric columns.
    """
    numeric_df = df.select_dtypes(include=['number'])
    return numeric_df

def handle_missing_values(df, strategy='drop'):
    """
    Handle missing values in DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        strategy (str): Strategy to handle missing values. 
                       Options: 'drop', 'fill_mean', 'fill_median'.
    
    Returns:
        pd.DataFrame: DataFrame with handled missing values.
    """
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'fill_mean':
        return df.fillna(df.mean())
    elif strategy == 'fill_median':
        return df.fillna(df.median())
    else:
        raise ValueError("Invalid strategy. Use 'drop', 'fill_mean', or 'fill_median'.")
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

if __name__ == "__main__":
    sample_data = [3, 1, 2, 1, 4, 3, 5, 2, 6]
    cleaned = remove_duplicates_preserve_order(sample_data)
    print(f"Original: {sample_data}")
    print(f"Cleaned: {cleaned}")