
import numpy as np
import pandas as pd

def remove_outliers_iqr(dataframe, column, multiplier=1.5):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Args:
        dataframe: pandas DataFrame
        column: Column name to process
        multiplier: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    return filtered_df

def normalize_column(dataframe, column, method='minmax'):
    """
    Normalize a DataFrame column using specified method.
    
    Args:
        dataframe: pandas DataFrame
        column: Column name to normalize
        method: 'minmax' or 'zscore' normalization
    
    Returns:
        Series with normalized values
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if method == 'minmax':
        col_min = dataframe[column].min()
        col_max = dataframe[column].max()
        
        if col_max == col_min:
            return dataframe[column].apply(lambda x: 0.5)
        
        normalized = (dataframe[column] - col_min) / (col_max - col_min)
        return normalized
    
    elif method == 'zscore':
        col_mean = dataframe[column].mean()
        col_std = dataframe[column].std()
        
        if col_std == 0:
            return dataframe[column].apply(lambda x: 0)
        
        normalized = (dataframe[column] - col_mean) / col_std
        return normalized
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")

def clean_dataset(dataframe, numeric_columns=None, outlier_multiplier=1.5, normalize_method='minmax'):
    """
    Clean dataset by removing outliers and normalizing numeric columns.
    
    Args:
        dataframe: Input DataFrame
        numeric_columns: List of numeric columns to process (default: all numeric)
        outlier_multiplier: Multiplier for IQR outlier detection
        normalize_method: Normalization method ('minmax' or 'zscore')
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = dataframe.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns:
            # Remove outliers
            q1 = cleaned_df[column].quantile(0.25)
            q3 = cleaned_df[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - outlier_multiplier * iqr
            upper_bound = q3 + outlier_multiplier * iqr
            
            mask = (cleaned_df[column] >= lower_bound) & (cleaned_df[column] <= upper_bound)
            cleaned_df = cleaned_df[mask]
            
            # Normalize
            if normalize_method == 'minmax':
                col_min = cleaned_df[column].min()
                col_max = cleaned_df[column].max()
                if col_max != col_min:
                    cleaned_df[column] = (cleaned_df[column] - col_min) / (col_max - col_min)
            
            elif normalize_method == 'zscore':
                col_mean = cleaned_df[column].mean()
                col_std = cleaned_df[column].std()
                if col_std != 0:
                    cleaned_df[column] = (cleaned_df[column] - col_mean) / col_std
    
    cleaned_df = cleaned_df.reset_index(drop=True)
    return cleaned_df

def validate_dataframe(dataframe, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        dataframe: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(dataframe, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(dataframe) < min_rows:
        return False, f"DataFrame must have at least {min_rows} rows"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in dataframe.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"import pandas as pd
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

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    If no columns specified, clean all numeric columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): List of column names to clean
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
            except Exception as e:
                print(f"Warning: Could not clean column '{col}': {e}")
    
    return cleaned_df

def save_cleaned_data(df, input_path, suffix='_cleaned'):
    """
    Save cleaned DataFrame to a new CSV file.
    
    Args:
        df (pd.DataFrame): Cleaned DataFrame
        input_path (str): Original file path
        suffix (str): Suffix to add to filename
    """
    if input_path.endswith('.csv'):
        output_path = input_path.replace('.csv', f'{suffix}.csv')
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
    else:
        print("Error: Only CSV files are supported")

if __name__ == "__main__":
    sample_data = {
        'id': range(1, 21),
        'value': [10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
                  100, 22, 23, 24, 25, 26, 27, 28, 29, 200]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original data shape:", df.shape)
    print("Original data summary:")
    print(df['value'].describe())
    
    cleaned_df = clean_numeric_data(df, columns=['value'])
    print("\nCleaned data shape:", cleaned_df.shape)
    print("Cleaned data summary:")
    print(cleaned_df['value'].describe())import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_method=None):
    """
    Clean a pandas DataFrame by handling missing values and duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_method (str or None): Method to fill missing values: 'mean', 'median', 'mode', or None to drop.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if fill_method is None:
        cleaned_df = cleaned_df.dropna()
    else:
        if fill_method == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_method == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif fill_method == 'mode':
            cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
        else:
            raise ValueError("fill_method must be 'mean', 'median', 'mode', or None")
    
    # Remove duplicates
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate dataset structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    dict: Dictionary with validation results.
    """
    validation_result = {
        'row_count': len(df),
        'column_count': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        validation_result['missing_required_columns'] = missing_columns
        validation_result['all_required_columns_present'] = len(missing_columns) == 0
    
    return validation_result

# Example usage
if __name__ == "__main__":
    # Create sample data
    data = {
        'A': [1, 2, None, 4, 1],
        'B': [5, None, 7, 8, 5],
        'C': ['x', 'y', 'z', 'x', 'y']
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\nValidation results:")
    print(validate_dataset(df))
    
    # Clean the data
    cleaned = clean_dataset(df, drop_duplicates=True, fill_method='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    print("\nCleaned validation results:")
    print(validate_dataset(cleaned))
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    data (pandas.DataFrame): The input dataframe.
    column (str): The column name to clean.
    
    Returns:
    pandas.DataFrame: Dataframe with outliers removed.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def calculate_basic_stats(data, column):
    """
    Calculate basic statistics for a column after outlier removal.
    
    Parameters:
    data (pandas.DataFrame): The input dataframe.
    column (str): The column name to analyze.
    
    Returns:
    dict: Dictionary containing mean, median, and standard deviation.
    """
    stats = {
        'mean': np.mean(data[column]),
        'median': np.median(data[column]),
        'std': np.std(data[column])
    }
    return stats

if __name__ == "__main__":
    import pandas as pd
    
    sample_data = pd.DataFrame({
        'values': [10, 12, 12, 13, 12, 11, 10, 100, 12, 14, 15, 10, 9, 8, 12, 11, 200]
    })
    
    print("Original data:")
    print(sample_data)
    
    cleaned_data = remove_outliers_iqr(sample_data, 'values')
    print("\nCleaned data:")
    print(cleaned_data)
    
    stats = calculate_basic_stats(cleaned_data, 'values')
    print("\nBasic statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")